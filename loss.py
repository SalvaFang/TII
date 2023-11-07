#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
class TVLossPix(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLossPix, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]

        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EdgeSaliencyLoss(nn.Module):
    def __init__(self, device, alpha_sal=0.7):
        super(EdgeSaliencyLoss, self).__init__()

        self.alpha_sal = alpha_sal

        self.laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float,requires_grad=False)
        self.laplacian_kernel = self.laplacian_kernel.view((1, 1, 3, 3))  # Shape format of weight for convolution
        self.laplacian_kernel = self.laplacian_kernel.to(device)

    @staticmethod
    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (1 - target) * torch.log(
            1 - input_ + eps)
        return torch.mean(wbce_loss)

    def forward(self, y_pred, y_gt):
        # Generate edge maps
        y_gt_edges = F.relu(torch.tanh(F.conv2d(y_gt, self.laplacian_kernel, padding=(1, 1))))
        y_pred_edges = F.relu(torch.tanh(F.conv2d(y_pred, self.laplacian_kernel, padding=(1, 1))))

        # sal_loss = F.binary_cross_entropy(input=y_pred, target=y_gt)
        # sal_loss = self.weighted_bce(input_=y_pred, target=y_gt, weight_0=1.0, weight_1=1.12)
        d_loss=torch.mean(torch.abs(y_pred_edges-y_gt_edges))
        # total_loss =  torch.abs(ssim(y_gt_edges,y_pred_edges))

        return d_loss

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class NormalLoss(nn.Module):
    def __init__(self,ignore_lb=255, *args, **kwargs):
        super( NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)


import torch


def compute_info_weight(image):
    # 计算梯度
    gradient_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    gradient_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

    # 计算梯度的平均值
    mean_gradient_x = torch.mean(gradient_x)
    mean_gradient_y = torch.mean(gradient_y)

    # 计算梯度的标准差
    std_gradient_x = torch.std(gradient_x)
    std_gradient_y = torch.std(gradient_y)

    # 计算信息丰富度的权重
    info_weight = 1 - (std_gradient_x + std_gradient_y) / (2 * (mean_gradient_x + mean_gradient_y))

    return info_weight


def compute_sharpness_weight(image):
    # 计算梯度
    gradient_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    gradient_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

    # 计算梯度的平均值
    mean_gradient_x = torch.mean(gradient_x)
    mean_gradient_y = torch.mean(gradient_y)

    # 计算清晰度的权重
    sharpness_weight = 1 - F.relu(mean_gradient_x - mean_gradient_y)

    return sharpness_weight


def compute_entropy_weight(image):
    # 将图像拉成一维向量
    flatten_image = image.view(-1)

    # 计算像素值出现的频率
    histogram = torch.histc(flatten_image, bins=256, min=0, max=255)
    probabilities = histogram / torch.sum(histogram)

    # 计算信息熵
    entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-6))

    # 计算信息熵的权重
    entropy_weight = 1 - entropy / (torch.log2(torch.tensor(256.)) * flatten_image.shape[0])

    return entropy_weight


def PSNRLoss(img1, img2):
    mse_loss = torch.mean((img1 - img2)**2)
    psnr_loss = 10 * torch.log10(1 / mse_loss)
    return -psnr_loss*0.01


def entropy_loss(img):
    # 计算图像的信息熵
    eps = 1e-7
    H = -torch.mean(torch.sum(F.softmax(img, dim=1) * F.log_softmax(img + eps, dim=1), dim=1))

    # 返回负的信息熵作为损失函数，因为最小化损失等价于最大化信息熵
    return -H*0.1


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=1):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # print(mask.shape,ssim_map.shape)
    ssim_map = ssim_map * mask

    ssim_map = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)

    return ssim_map
    # if size_average:
    #     return ssim_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)


import torch
import torch.nn.functional as F


def sobel_filter(x):
    # Define Sobel filter kernels
    kernelx = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    kernely = [[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    sobelx = F.conv2d(x, weightx, padding=1)
    sobely = F.conv2d(x, weighty, padding=1)

    return torch.sqrt(sobelx ** 2 + sobely ** 2)



def three_component_ssim(ref_img, dist_img, window_size=11):
    # Compute SSIM map
    ssim_map = ssimloss(ref_img, dist_img)

    # Compute gradient magnitudes of reference and distorted images
    ref_grad_mag = sobel_filter(ref_img)
    dist_grad_mag = sobel_filter(dist_img)

    # Compute thresholds for edge, texture, and smoothness regions
    edge_threshold = ref_grad_mag.mean() * 0.5
    texture_threshold = ref_grad_mag.mean() * 0.25
    smoothness_threshold = ref_grad_mag.mean() * 0.25

    # Compute binary masks for edge, texture, and smoothness regions
    edge_mask = ((ref_grad_mag > edge_threshold) | (dist_grad_mag > edge_threshold)).float()
    texture_mask = ((ref_grad_mag <= texture_threshold) & (dist_grad_mag <= texture_threshold)).float()
    smoothness_mask = ((ref_grad_mag > smoothness_threshold) & (dist_grad_mag > smoothness_threshold)).float()

    # Compute weights for edge, texture, and smoothness components
    edge_weight = 0.5
    texture_weight = 0.25
    smoothness_weight = 0.25

    # Compute weighted SSIM values for each component
    edge_ssim = (edge_mask * ssim_map).sum() / (edge_mask.sum() + 1e-10)
    texture_ssim = (texture_mask * ssim_map).sum() / (texture_mask.sum() + 1e-10)
    smoothness_ssim = (smoothness_mask * ssim_map).sum() / (smoothness_mask.sum() + 1e-10)

    # Compute final 3C-SSIM index
    three_component = (edge_weight * edge_ssim) + (texture_weight * texture_ssim) + (
                smoothness_weight * smoothness_ssim)

    return three_component


def Contrast(img1, img2, window_size=11, channel=1):
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=1):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        mask = torch.logical_and(img1 > 0, img2 > 0).float()
        for i in range(self.window_size // 2):
            mask = (F.conv2d(mask, window, padding=self.window_size // 2, groups=channel) > 0.8).float()
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask=mask)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


ssimloss = SSIMLoss(window_size=11)


def fusion_channel_sf(f1, f2, kernel_radius=5):
    """
    Perform channel sf fusion two features
    """
    device = f1.device
    b, c, h, w = f1.shape
    r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) \
        .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) \
        .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
    f2_r_shift = F.conv2d(f2, r_shift_kernel, padding=1, groups=c)
    f2_b_shift = F.conv2d(f2, b_shift_kernel, padding=1, groups=c)

    f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
    f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)

    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().cuda(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
    f2_sf = torch.sum(F.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
    weight_zeros = torch.zeros(f1_sf.shape).cuda(device)
    weight_ones = torch.ones(f1_sf.shape).cuda(device)

    # get decision map
    dm_tensor = torch.where(f1_sf > f2_sf, f1, f2).cuda(device)


    return dm_tensor

def get_palette():
    unlabelled = [0, 0, 0]
    car = [64, 0, 128]
    #person = [64, 64, 0]
    person = [255]
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

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        self.mse_loss = torch.nn.MSELoss()



    def forward(self,image_vis,image_ir,labels,generate_img,i,qqout,qir):
        # image_y=image_vis[:,:1,:,:]


        weight_zeros = torch.zeros(image_vis.shape).cuda("cuda:0")
        weight_ones = torch.ones(image_vis.shape).cuda("cuda:0")

        # mask0 = torch.where(ir_grad > ir_grad.mean(), image_ir, weight_zeros)

        # palette = get_palette()
        # pred = labels[0,:,:]
        # img = np.zeros((pred.shape[0], pred.shape[1], 1), dtype=np.uint8)
        # for cid in range(1, int(labels[0].max())):
        #     if cid == 2:
        #         img[pred.cpu() == cid] = palette[cid]
        # t0 = torch.from_numpy(img/255.0)
        # t0 = t0.squeeze()  # add a new dimension at index 0 with size 1
        # t0 = t0.unsqueeze(0)  # add a new dimension at index 0 with size 1
        # t0 = t0.unsqueeze(1)  # add a new dimension at index 1 with size 1
        #
        #
        # pred1 = labels[1, :, :]
        # img1 = np.zeros((pred1.shape[0], pred1.shape[1], 1), dtype=np.uint8)
        # for cid in range(1, int(labels[1].max())):
        #     if cid == 2:
        #         img1[pred1.cpu() == cid] = palette[cid]
        # # convert the numpy array to a PyTorch tensor
        # t = torch.from_numpy(img1/255.0)
        # t = t.squeeze()  # add a new dimension at index 0 with size 1
        # t = t.unsqueeze(0)  # add a new dimension at index 0 with size 1
        # t = t.unsqueeze(1)  # add a new dimension at index 1 with size 1
        #
        # t1=torch.cat((t0,t),dim=0).cuda()


        x_in_max=torch.max(image_vis,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        # print(t1.max())

        y_grad=self.sobelconv(image_vis)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)#y_grad+ir_grad#
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)



        loss_total=loss_in+10*loss_grad #+ 1-(self.ssim_loss(qqout, image_vis))   #+ 2-(self.ssim_loss(generate_img, image_vis))-(self.ssim_loss(generate_img, qqout))#+self.mse_loss(generate_img, qqout)+self.mse_loss(generate_img, qir)
        return loss_total,loss_in,loss_grad


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss



class Fusionloss1(nn.Module):
    def __init__(self):
        super(Fusionloss1, self).__init__()
        self.sobelconv=Sobelxy()
        self.edge=MultiscaleLoG()

    def forward(self,image_vis,image_ir,labels,generate_img,i):
        image_y=image_vis[:,:1,:,:]


        weight_zeros = torch.zeros(image_y.shape).cuda("cuda:0")
        weight_ones = torch.ones(image_y.shape).cuda("cuda:0")
        mask = torch.where(image_y > 0.99 * weight_ones, image_ir, weight_zeros)


        # 定义一个3x3的高斯滤波器
        kernel = torch.Tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        kernel = kernel.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

        # 假设可见光图像为image_vis，红外图像为image_ir
        # 计算可见光图像的亮度信息
        # image_y1 = 0.299 * image_vis[:, 0, :, :] + 0.587 * image_vis[:, 1, :, :] + 0.114 * image_vis[:, 2, :, :]
        # image_y2 = image_y1 / image_y1.max()  # 归一化到[0,1]范围内
        # weight = image_y2.unsqueeze(1)
        # 计算掩膜
        # mask = (image_y > 0.95).float()  # threshold是一个合适的阈值，用于控制掩膜的范围

        # print(torch.sum(mask))

        # if torch.sum(mask) > 1000:
        #     # 使用卷积操作平滑掩膜
        #     mask=1-mask
        #     mask = F.conv2d(mask, kernel.cuda(), padding=1)
        #     mask = F.conv2d(mask, kernel.cuda(), padding=1)
        # else:
        #     mask = 0
        # 限制torch.max操作的影响
        # Lillu = torch.mean(mask *generate_img)

        # LS=2-ssimloss((1-labels)*generate_img, (1-labels)*image_y)-ssimloss(labels*generate_img, labels*image_ir)


        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)

        mask1 = torch.where(y_grad > y_grad.mean(), weight_ones, weight_zeros)
        mask0 = torch.where(ir_grad > ir_grad.mean(), weight_ones, weight_zeros)

        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)

        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)

        lir= 1-ssimloss(generate_img, image_ir).mean()
        lvis=1-ssimloss(image_y,generate_img).mean()
        # print(lvis)


        # le=entropy_loss(generate_img)
        # lg=-torch.mean(generate_img_grad)
        # ln=PSNRLoss(generate_img_grad, image_ir)

        loss_total=loss_in+20*loss_grad    #+0.2*lg#0.1*(le+lg+ln)0.5*illu
        return loss_total,loss_grad,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleLoG(nn.Module):
    def __init__(self, ksize_min=3, ksize_max=15, num_scales=3):
        super(MultiscaleLoG, self).__init__()
        self.ksize_min = ksize_min
        self.ksize_max = ksize_max
        self.num_scales = num_scales

        self.kernel = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).unsqueeze(0).unsqueeze(0)

    def forward(self, tensor):
        tensor_gray = tensor.mean(dim=1, keepdim=True)  # 将RGB图像转换为灰度图像
        tensor_edges = torch.zeros_like(tensor_gray)  # 存储边缘强度图像的张量
        for ksize in range(self.ksize_min, self.ksize_max + 1, 2):
            for sigma in range(1, self.num_scales + 1):
                # 计算LoG边缘强度图像
                tensor_smooth = gaussian_blur(tensor_gray, ksize, sigma)
                kernel = self.kernel.to(tensor.device)
                tensor_log = F.conv2d(tensor_smooth, kernel, padding=1)
                tensor_log = torch.abs(tensor_log)
                # 将边缘强度图像加入总的边缘强度图像张量中
                tensor_edges += tensor_log
        return tensor_edges


import torch.nn.functional as F

def gaussian_kernel(size, sigma, channels):
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel.unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel / kernel.sum()




def gaussian_blur(image, size, sigma):
    kernel = gaussian_kernel(size, sigma, channels=image.shape[1]).cuda()
    return F.conv2d(image, kernel, padding=size // 2, stride=1, groups=image.shape[1])





if __name__ == '__main__':
    pass

