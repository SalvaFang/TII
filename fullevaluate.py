
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import math
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
class MscEval(object):
    def __init__(
        self,
        model,
        dataloader,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        n_classes=9,
        lb_ignore=255,
        cropsize=1024,
        flip=False,
        *args,
        **kwargs
    ):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model
        self.net = model.eval()

    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0] - H, size[1] - W
        hst, hed = margin_h // 2, margin_h // 2 + H
        wst, wed = margin_w // 2, margin_w // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob

    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5 / 6.0
        N, C, H, W = im.size()
        long_size, short_size = (H, W) if H > W else (W, H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        else:
            stride = math.ceil(cropsize * stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W - cropsize) / stride) + 1
            n_y = math.ceil((H - cropsize) / stride) + 1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = (
                        min(H, stride * iy + cropsize),
                        min(W, stride * ix + cropsize),
                    )
                    hst, wst = hed - cropsize, wed - cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        return prob

    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob

    def get_palette(self):
        unlabelled = [0, 0, 0]
        car = [64, 0, 128]
        person = [64, 64, 0]
        bike = [0, 128, 192]
        curve = [0, 0, 192]
        car_stop = [128, 128, 0]
        guardrail = [64, 64, 128]
        color_cone = [192, 128, 128]
        bump = [192, 64, 0]
        palette = np.array(
            [
                unlabelled,
                car,
                person,
                bike,
                curve,
                car_stop,
                guardrail,
                color_cone,
                bump,
            ]
        )
        return palette

    def visualize_tsne(self,preds2, labels1, n_classes):
        # Flatten the predictions and labels tensors
        preds_flat = preds2.reshape(preds2.size(1), -1).T



        # 将labels1插值为目标大小
        labels1 = labels1.float()
        labels1_resized = F.interpolate(labels1, size=(60, 80), mode='nearest')

        # 将labels1_resized展平
        n_samples = labels1_resized.shape[0]
        labels_flat = torch.stack([labels1_resized[i, 0].flatten() for i in range(n_samples)])
        labels_flat = labels_flat.flatten()

        # Convert the predictions to a numpy array
        preds_np = preds_flat.cpu().detach().numpy()

        # 进行PCA降维
        pca = PCA(n_components=50)  # 降至50维
        features_pca = pca.fit_transform(preds_np)


        # Use TSNE to reduce the dimensionality of the predictions to 2
        tsne = TSNE(n_components=2, random_state=0)
        preds_tsne = tsne.fit_transform(preds_np)

        # Get the color map for the classes
        cmap = plt.cm.get_cmap('viridis', n_classes)

        # Plot the predicted embeddings with the class color
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(preds_tsne[:, 0], preds_tsne[:, 1], c=labels_flat, s=5,cmap=plt.cm.get_cmap('gist_ncar', 9))
        plt.colorbar(sc)

        # Set the title and axis labels
        ax.set_title('TSNE Visualization of Predictions')
        ax.set_xlabel('TSNE Dimension 1')
        ax.set_ylabel('TSNE Dimension 2')
        plt.savefig('./1.png', dpi=100)
        # Save the plot as a PDF vector image
        plt.savefig('TSNE.pdf', dpi=100, format='pdf', bbox_inches='tight')

        # Show the plot
        plt.show()

    def visualize(self, save_name, predictions):
        palette = self.get_palette()
        pred = predictions
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, int(predictions.max())):
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save(save_name+'.png')

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb == ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes ** 2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self, Method='IFCNN'):
        ## evaluate
        outputs=[]
        self.preds_list = []
        self.lable_list=[]
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        # dloader = self.dl
        for i, (imgs, label, fn) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            prob = self.net(imgs)
            probs = prob[0].data.cpu().numpy()
            preds = np.argmax(probs, axis=1)

            # #TSNE
            # self.visualize_tsne(prob[3], label, n_classes) #每张图像的TSNE

            # robs1 = prob[3].data.cpu().numpy()
            # preds1= np.argmax(robs1, axis=1)
            if i % 87 == 0:
                self.preds_list.append(prob[3])
                self.lable_list.append(label)
                # outputs.append(preds1[0])


            for i in range(1):
                outpreds = preds[i]
                name = fn[i]
                folder_path = 'D:\FAQ\eight\LY\Segmentation/test'#os.path.join('results', Method, 'Segmentation')
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path = os.path.join(folder_path, name)
                self.visualize(file_path, outpreds)
                # label visualize
                # self.visualize(file_path,label[0,0,:,:])

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once

        # np.save('conf_mat.npy', hist)
        # np.save('1.npy', outputs)
        # features=np.load('1.npy')

        preds_n = torch.cat(self.preds_list, dim=0)
        label_n = torch.cat(self.lable_list, dim=0)
        self.visualize_tsne(preds_n, label_n, n_classes)  # 37张图像的TSNE


        a=hist/len(dloader)
        # Compute the normalized confusion matrix
        row_sums = np.sum(a, axis=1)
        norm_conf_mat = a / row_sums[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        # sns.heatmap(norm_conf_mat, annot=True, cmap="Blues", fmt='g', xticklabels=[...], yticklabels=[...])
        fig, ax = plt.subplots(figsize=(10, 8))
        indices = range(len(norm_conf_mat))
        class_names = ['unlabelled',
                'car',
                'person',
                'bike',
                'curve',
                'car_stop',
                'guardrail',
                'color_cone',
                'bump',]  # 类别名称
        sns.set(font_scale=1.1)
        sns.heatmap(norm_conf_mat, annot=True, cmap='Blues', fmt='.2g', ax=ax, xticklabels=class_names,
                    yticklabels=class_names)
        # sns.set(font_scale=1.1)
        # sns.heatmap(norm_conf_mat, annot=True, cmap='Blues', fmt='.2g', ax=ax)
        plt.xlabel('Predicted Results', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.title('Confusion Matrix')
        # Save the plot as a PDF vector image
        plt.savefig('norm_conf_mat.pdf', dpi=300, format='pdf', bbox_inches='tight')
        plt.show()



        IOUs = np.diag(hist) / (
            np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist)
        )
        mIOU = np.mean(IOUs)
        mIOU = mIOU
        IoU_list = IOUs.tolist()
        IoU_list.append(mIOU)
        IoU_list = [round(i, 4) for i in IoU_list]
        print(Method, ':\tIoU:', IoU_list, '\n')

        # 创建一个DataFrame对象来保存数据
        df = pd.DataFrame({'IoU': IoU_list})
        df = df.T
        # 保存到Excel文件
        filename = 'IoU.xlsx'
        df.to_excel(filename, index=False)

        print('数据已保存到', filename)

        return mIOU
