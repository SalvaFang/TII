#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from torch.autograd import Variable
from FusionNet import FusionNet,Quality
from TaskFusion_dataset import Fusion_dataset,qFusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from model_TII import BiSeNet
from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss,EdgeSaliencyLoss,ssimloss,TVLossPix, FocalLoss,DiceLoss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
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

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
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

from build_data import *
from module_list import *
def train_seg(i=0, logger=None):

    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, mode=0o777, exist_ok=True)
    # if logger == None:
    #     logger = logging.getLogger()
    #     setup_logger(modelpth)

    # dataset
    n_classes = 9
    n_img_per_gpu = 8
    n_workers = 4
    cropsize = [640, 480]
    ds = CityScapes('./MSRS/', cropsize=cropsize, mode='train', Method=Method)
    dl = DataLoader(
        ds,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)
    if i>0:
        load_path = './model/Fusion/model_final_test.pth'
        net.load_state_dict(torch.load(load_path))
    else:
        load_path = './model/Fusion/model_final.pth'
        net.load_state_dict(torch.load(load_path))


    net.cuda()
    net.eval()
    for p in net.parameters():
        p.requires_grad = True
    print('Load Segmentation Model {} Sucessfully~')
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    lf1=FocalLoss()
    ld1=DiceLoss()
    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-6
    max_iter = 1000
    power = 0.9
    warmup_steps = 50
    warmup_start_lr = 1e-5
    it_start = i * 500
    iter_nums = 500

    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
        it=it_start,
    )

    # train loop
    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(iter_nums):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb, _ = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, mid, pred,rep = net(im)
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(mid, lb)
        # f1=lf1(out, lb).cuda()
        # f2=lf1(mid, lb).cuda()
        #
        # f3=ld1(out, lb).cuda()
        # f4=ld1(mid, lb).cuda()


        loss1 = lossp+0.75*loss2

        # regional contrastive loss
        with torch.no_grad():
            mask = F.interpolate((lb.unsqueeze(1) >= 0).float(), size=pred.shape[2:], mode='nearest')
            label = F.interpolate(label_onehot(lb, 9), size=pred.shape[2:],
                                  mode='nearest')
            prob = torch.softmax(pred, dim=1)

        # reco_loss = compute_reco_loss(rep, label, mask, prob, 0.97, 0.07, 256,
        #                               512)

        loss = loss1 #+ 0.2*reco_loss


        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        # print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)

            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int(( max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                [
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]
            ).format(
                it=it_start+it + 1, max_it= max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed
    # dump the final model
    save_pth = osp.join(modelpth, 'model_final_test.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info(
        'Segmentation Model Training done~, The Model is saved to: {}'.format(
            save_pth)
    )
    logger.info('\n')

def train_fusion(num=0, logger=None):
    # num: control the segmodel 
    lr_start = 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.cuda()
    fusionmodel.train()
    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)

    if num>0:
        fusionmodel.load_state_dict(torch.load('./model/Fusion/fusion_model_test.pth'))


    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    # if num>=0:
    n_classes = 9
    segmodel = BiSeNet(n_classes=n_classes)

    if num>0:
        save_pth = osp.join(modelpth, 'model_final_test.pth')
        segmodel.load_state_dict(torch.load(save_pth))
    else:
        save_pth = osp.join(modelpth, 'model_final.pth')
        segmodel.load_state_dict(torch.load(save_pth))

    if logger == None:
        logger = logging.getLogger()
        setup_logger(modelpth)

    segmodel.cuda()
    segmodel.eval()
    for p in segmodel.parameters():
        p.requires_grad = False
    print('Load Segmentation Model {} Sucessfully~'.format(save_pth))



    qmodel = eval('Quality')(output=1)
    qsave_path =osp.join(modelpth, 'quality_final.pth')
    qmodel.load_state_dict(torch.load(qsave_path))
    qmodel.cuda()
    qmodel.eval()
    for p in qmodel.parameters():
        p.requires_grad = False
    print('Load Segmentation Model {} Sucessfully~'.format(qsave_path))
    mse_loss = torch.nn.MSELoss()
    train_dataset = qFusion_dataset('train1')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    #
    # if num>0:
    score_thres = 0.7
    ignore_idx = 255
    n_min = 8 * 640 * 480 // 8
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    criteria_fusion = Fusionloss()
    tv=TVLossPix()
    edloss=EdgeSaliencyLoss(device='cuda')

    epoch = 2
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        # print('\n| epo #%s begin...' % epo)
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, (image_vis, img, M, name,label, image_ir) in enumerate(train_loader):
            fusionmodel.train()
            image_vis = Variable(image_vis).cuda()
            M = Variable(M).cuda()
            img = Variable(img).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            img = RGB2YCrCb(img)
            image_ir = Variable(image_ir).cuda()
            label = Variable(label).cuda()

            a=torch.cat((image_vis_ycrcb[:,:1], M),dim=1)

            logits,f6 = fusionmodel(a, image_ir)


            # logitsvis = fusionmodel(image_vis_ycrcb, image_vis_ycrcb)
            # quality loss

            # CD=image_vis[:,:1]
            # M = CD.repeat(1,16, 1, 1)

            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )

            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            lb = torch.squeeze(label, 1)
            optimizer.zero_grad()
            # seg loss

            # if num>=0:

            qvis,f7 = qmodel(img, M)

            qir,f7 = qmodel(image_ir, M)



            out, mid, pred,rep = segmodel(fusion_image)
            lossp = criteria_p(out, lb)
            loss2 = criteria_16(mid, lb)

            # regional contrastive loss
            with torch.no_grad():
                mask = F.interpolate((lb.unsqueeze(1) >= 0).float(), size=pred.shape[2:], mode='nearest')
                label = F.interpolate(label_onehot(lb, 9), size=pred.shape[2:],
                                      mode='nearest')
                prob = torch.softmax(pred, dim=1)

            reco_loss = compute_reco_loss(rep, label, mask, prob, 0.97, 0.07, 256,
                                          512)

            seg_loss = lossp + 0.75*loss2+0.3*reco_loss


            # fusion loss
            loss_fusion4, loss_in33, loss_grad33 = criteria_fusion(
                image_vis_ycrcb[:,:1,:,:], image_ir, label, logits, num, qvis, qvis
            )

            loss_fusion5, loss_in33, loss_grad33 = criteria_fusion(
                qvis, qir, label, logits,num, qvis, qvis
            )
            ltv=tv(logits)

            # loss_fusion5, loss_in33, loss_grad33 = criteria_fusion(
            #     image_vis_ycrcb[:,:1,:,:], image_ir, label, logits,num, qvis,qir,1
            # )
            # loss_fusion51, loss_in341, loss_grad341 = criteria_fusion(
            #     image_vis_ycrcb, image_ir, label, logits,num, qout
            # )

            # fusion loss
            # loss_fusion, loss_in, loss_grad = criteria_fusion(
            #     image_vis_ycrcb, image_ir, label, logits, num, qout
            # )
            # if num > 0:
            #     loss_total = (num) * seg_loss
            # else:
            loss_total =loss_fusion4#+0.5*(loss_fusion5+0.5*edloss(logits, qvis))+0.5*reco_loss#+loss_fusion5+0.5*edloss(logits, qvis)#+0.5*reco_loss+0.5*edloss(logits, qvis)#+loss_fusion5#+ltv#reco_loss+ 0.1*reco_loss+0.5*reco_loss++0.5*reco_loss
            # loss_total=loss_fusion4
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 2 == 0:
                if num>0:
                    loss_seg=seg_loss.item()
                else:
                    loss_seg=0
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_seg: {loss_seg:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_seg=loss_seg,
                    time=t_intv,
                    eta=eta,
                )
                print(msg)
                # logger.info(msg)
                st = ed
    fusion_model_file = os.path.join(modelpth, 'fusion_model_test.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)

    # fusion_model_file1 = os.path.join(modelpth, 'model_final_our.pth')
    # state = segmodel.module.state_dict() if hasattr(segmodel, 'module') else segmodel.state_dict()
    # torch.save(state, fusion_model_file1)

    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')

def run_fusion(type='train1'):
    fusion_model_path = './model/Fusion/fusion_model_test.pth'
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.eval()
    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    test_dataset = qFusion_dataset('train1')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, img, M, name,labels, images_ir) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            labels = Variable(labels)
            M = Variable(M).cuda()
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
            images_vis_ycrcb = RGB2YCrCb(images_vis)

            a = torch.cat((images_vis_ycrcb[:, :1], M), dim=1)
            logits,_ = fusionmodel(a, images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :,
                 :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=16)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    # modelpth = './model'
    # Method = 'Fusion'
    # modelpth = os.path.join(modelpth, Method)
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    for i in range(1):
        # train_fusion(i, logger)
        # print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        run_fusion('train')
        # print("|{0} Fusion Image Sucessfully~!".format(i + 1))
        train_seg(i, logger)
        # print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
    print("training Done!")