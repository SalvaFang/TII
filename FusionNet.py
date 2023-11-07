# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class ConvBnLeakyRelu2d(nn.Module):
#     # convolution
#     # batch normalization
#     # leaky relu
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(ConvBnLeakyRelu2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
#         self.bn   = nn.BatchNorm2d(out_channels)
#     def forward(self, x):
#         return F.leaky_relu(self.conv(x), negative_slope=0.2)
#
# class ConvBnTanh2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(ConvBnTanh2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
#         self.bn   = nn.BatchNorm2d(out_channels)
#     def forward(self,x):
#         return torch.tanh(self.conv(x))/2+0.5
#
# class ConvLeakyRelu2d(nn.Module):
#     # convolution
#     # leaky relu
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(ConvLeakyRelu2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
#         # self.bn   = nn.BatchNorm2d(out_channels)
#     def forward(self,x):
#         # print(x.size())
#         return F.leaky_relu(self.conv(x), negative_slope=0.2)
#
# class Sobelxy(nn.Module):
#     def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(Sobelxy, self).__init__()
#         sobel_filter = np.array([[1, 0, -1],
#                                  [2, 0, -2],
#                                  [1, 0, -1]])
#         self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
#         self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
#         self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
#         self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
#     def forward(self, x):
#         sobelx = self.convx(x)
#         sobely = self.convy(x)
#         x=torch.abs(sobelx) + torch.abs(sobely)
#         return x
#
# class Conv1(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
#         super(Conv1, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
#     def forward(self,x):
#         return self.conv(x)
#
# class DenseBlock(nn.Module):
#     def __init__(self,channels):
#         super(DenseBlock, self).__init__()
#         self.conv1 = ConvLeakyRelu2d(channels, channels)
#         self.conv2 = ConvLeakyRelu2d(2*channels, channels)
#         # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
#     def forward(self,x):
#         x=torch.cat((x,self.conv1(x)),dim=1)
#         x = torch.cat((x, self.conv2(x)), dim=1)
#         # x = torch.cat((x, self.conv3(x)), dim=1)
#         return x
#
# class RGBD(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(RGBD, self).__init__()
#         self.dense =DenseBlock(in_channels)
#         self.convdown=Conv1(3*in_channels,out_channels)
#         self.sobelconv=Sobelxy(in_channels)
#         self.convup =Conv1(in_channels,out_channels)
#     def forward(self,x):
#         x1=self.dense(x)
#         x1=self.convdown(x1)
#         x2=self.sobelconv(x)
#         x2=self.convup(x2)
#         return F.leaky_relu(x1+x2,negative_slope=0.1)



import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.conv_block1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv_block2 = DepthwiseSeparableConv(in_channels + out_channels, out_channels)


    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(torch.cat([x, x1], dim=1))
        return x2

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)  # for IR and visible light inputs

    def forward(self, x1):
        x = self.conv_block(x1)
        return x

# class FusionNet(nn.Module):
#     def __init__(self, output):
#         super(FusionNet, self).__init__()
#         self.conv_block1 = ConvBlock(2, 16)  # for IR and visible light inputs
#         self.dsconv_block1 = DenseBlock(16, 32)
#         self.dsconv_block2 = DenseBlock(32, 64)
#         self.dsconv_block3 = DenseBlock(64, 128)
#         self.dblock1 = Block(128, 64)
#         self.dblock2 = Block(64, 32)
#         self.dblock3 = Block(32, 16)
#         self.final_conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, image_vis, image_ir):
#         x_vis_origin = image_vis[:, :1]
#         x_inf_origin = image_ir
#         # Encoder
#         x = torch.cat([x_vis_origin, x_inf_origin], dim=1)
#         x1 = self.conv_block1(x)
#         x2 = self.dsconv_block1(x1)
#         x3 = self.dsconv_block2(x2)
#         x4 = self.dsconv_block3(x3)
#
#         # Decoder
#         x = self.dblock1(x4)
#         x = self.dblock2(x)
#         x = self.dblock3(x)
#         x = self.final_conv(x)
#         return x



class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

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

class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()

        self.Fire = ConvBlock(18, 32)
        self.conv1 = ConvBlock(32, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 32)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)




    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis
        x_inf_origin = image_ir

        x = torch.cat((x_vis_origin, x_inf_origin), dim=1)

        f1=self.Fire(x)
        f2 = self.conv1(f1)
        f3 = self.conv2(f2)
        f4 = self.conv3(f3)
        f5 = self.conv4(f4)


        return f5,f5

# class FusionNet(nn.Module):
#     def __init__(self, output):
#         super(FusionNet, self).__init__()
#         vis_ch = [16,32,48]
#         inf_ch = [16,32,48]
#         output=1
#         self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
#         self.vis_rgbd1=RGBD(vis_ch[0], vis_ch[1])
#         self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])
#         # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
#         self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
#         self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
#         self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
#         # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
#         # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
#         self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1])
#         self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
#         self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
#         self.decode1 = ConvBnTanh2d(vis_ch[0], output)
#     def forward(self, image_vis,image_ir):
#         # split data into RGB and INF
#         x_vis_origin = image_vis[:,:1]
#         x_inf_origin = image_ir
#         # encode
#         x_vis_p=self.vis_conv(x_vis_origin)
#         x_vis_p1=self.vis_rgbd1(x_vis_p)
#         x_vis_p2=self.vis_rgbd2(x_vis_p1)
#         # x_vis_p3=self.vis_rgbd3(x_vis_p2)
#
#         x_inf_p=self.inf_conv(x_inf_origin)
#         x_inf_p1=self.inf_rgbd1(x_inf_p)
#         x_inf_p2=self.inf_rgbd2(x_inf_p1)
#         # x_inf_p3=self.inf_rgbd3(x_inf_p2)
#         # decode
#         x=self.decode4(torch.cat((x_vis_p2,x_inf_p2),dim=1))
#         # x=self.decode4(x)
#         x=self.decode3(x)
#         x=self.decode2(x)
#         x=self.decode1(x)
#         return x

class ConvBlock1(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock1, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Quality(nn.Module):
    def __init__(self, output):
        super(Quality, self).__init__()

        self.Fire = ConvBlock(17, 64)
        self.conv1 = ConvBlock(64, 64)
        self.conv2 = ConvBlock(64, 32)
        # self.conv3 = ConvBlock(32, 1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)
        # self.relu = nn.ReLU(inplace=True)


    def forward(self, image_vis, M):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        # x_inf_origin = image_ir

        x = torch.cat((x_vis_origin, M), dim=1)

        f4=self.Fire(x)
        f5 = self.conv1(f4)
        f6 = self.conv2(f5)
        # feature_visualization(f5,f5.type,5)
        # feature_vis(f5)
        f7 = self.conv3(f6)
        # f7=self.relu(f7)
        return f7,f6

def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = FusionNet(output=1)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()
