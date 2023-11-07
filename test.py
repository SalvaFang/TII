# coding:utf-8
import os
import argparse
import time
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_TII import BiSeNet
from TaskFusion_dataset import Fusion_dataset
from FusionNet import FusionNet
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image


# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main():
    fusion_model_path = './model/Fusion/fusion_model_test.pth'
    fusionmodel = eval('FusionNet')(output=1)

    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusionmodel.cuda()
    print('fusionmodel load done!')
    ir_path = 'D:\FAQ\eight\data/TNO/IR/'
    vi_path = 'D:\FAQ\eight\data/TNO/VIS/'
    # ir_path = 'D:\FAQ\eight\data/VIFB/IR/'
    # vi_path = 'D:\FAQ\eight\data/VIFB/VIS/'
    # ir_path = 'E:\BaiduNetdiskDownload\MSRS\MSRS/Infrared/test/MSRS/'
    # vi_path = 'E:\BaiduNetdiskDownload\MSRS\MSRS/Visible/test/MSRS/'

    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,name,M) in enumerate(test_loader):
            start = time.clock()
            images_vis = Variable(images_vis).cuda()
            images_ir = Variable(images_ir).cuda()
            M = Variable(M)

            images_vis = images_vis.cuda()
            images_ir = images_ir.cuda()
            M = M.cuda()
            images_vis_ycrcb = RGB2YCrCb(images_vis)

            # AE, _ = qfusionmodel(images_vis_ycrcb, M)


            a = torch.cat((images_vis_ycrcb[:, :1], M), dim=1)
            logits, _ = fusionmodel(a, images_ir)
            print(time.clock()-start)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )

            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )

            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))

def YCrCb2RGB(input_im):
    device = torch.device("cuda:0")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def RGB2YCrCb(input_im):

    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=-1)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    n_class = 9
    seg_model_path = './model/Fusion/model_final.pth'
    fusion_model_path = './model/Fusion/fusionmodel_final.pth'
    # fused_dir = 'D:\FAQ\eight\LY\Fusion\Study\Parameter\gama0.5/TNO'
    #'D:\FAQ\venv\LY\a-HARConstrastive\SeAFusion-main\MSRS\Fusion\test\MSRS'
    fused_dir='D:\FAQ/venv\LY/a-HARConstrastive\SeAFusion-main\MSRS\Fusion/test\MSRS'
    # fused_dir = 'D:\FAQ\eight\LY\Fusion/test'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()
