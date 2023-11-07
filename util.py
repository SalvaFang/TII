# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
from augmentation import model
from PIL import Image
import torch
import torchvision


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image


def float_to_uint8(image, scale=255.0):
    """
    Convert image from float type to uint8, meanwhile the clip between [0, 255]
    will be done.

    Parameters
    ----------
    image: numpy array image data of float type
    scale: a scale factor for image data

    Returns
    -------
    image_uint8: numpy array image data of uint8 type
    """
    image_uint8 = np.clip(np.round(image * scale), 0, 255).astype(np.uint8)
    return image_uint8


def get_noise_index(image, noise_num_ratio):
    """
    Get noise index for a certain ratio of noise number

    Parameters
    ----------
    image: numpy array image data
    noise_num_ratio: ratio of noise number with respect to the total number of
        pixels, between [0, 1]

    Returns
    -------
    row: row indexes
    col: column indexes
    """
    image_height, image_width = image.shape[0:2]
    noise_num = int(np.round(image_height * image_width * noise_num_ratio))
    row = np.random.randint(0, image_height, noise_num)
    col = np.random.randint(0, image_width, noise_num)
    return row, col


def get_noise_shape(image, speckle_size=1.0):
    """
    Get noise shape according to image shape.

    Parameters
    ----------
    image: numpy image data, shape is [H, W, C], C is optional
    speckle_size: speckle size of noise

    Returns
    -------
    noise_shape: a tuple whose length is 3
        The shape of noise. Let height, width be the image height and width.
        If image.ndim is 2, output noise_shape will be (height, width, 1),
        else (height, width, 3)
    """
    if speckle_size > np.min(image.shape[:2]):
        raise ValueError("speckle_size can NOT be larger than the min of "
                         "(image_height, image_width)")

    if image.ndim == 2:
        noise_shape = np.array(image.shape)
    else:
        noise_shape = np.array([*image.shape[:2], 3])

    if speckle_size > 1.0:
        noise_shape[0] = int(round(noise_shape[0] / speckle_size))
        noise_shape[1] = int(round(noise_shape[1] / speckle_size))
    return noise_shape


def add_noise(image, noise, noise_num_ratio=1.0):
    """
    Add noise to image.

    Parameters
    ----------
    image: numpy image data, shape is [H, W, C], C is optional
    noise: additive noise, same shape as 'image'
    noise_num_ratio: ratio of noise number with respect to the total number of
        pixels, between [0, 1]

    Returns
    -------
    noisy_image: image with noise
    """
    if not 0.0 <= noise_num_ratio <= 1.0:
        raise ValueError('noise_num_ratio must between [0, 1]')

    # preprocess noise
    if noise.ndim == 2:
        noise = np.expand_dims(noise, axis=2)
    channel = noise.shape[2]

    # initialize noisy_image
    noisy_image = image.copy().astype(np.float32)
    if noisy_image.ndim == 2:
        noisy_image = np.expand_dims(noisy_image, axis=2)

    # add noise to noisy_image
    if noise_num_ratio >= 1.0:
        noisy_image[:, :, :channel] += noise
    else:
        row, col = get_noise_index(image, noise_num_ratio)
        noisy_image[row, col, :channel] += noise[row, col, ...]

    # post processing for dtype and shape
    if image.dtype == np.uint8:
        noisy_image = float_to_uint8(noisy_image, scale=1.0)
    else:
        noisy_image = noisy_image.astype(image.dtype)
    noisy_image = np.squeeze(noisy_image)
    return noisy_image


def generate_uniform_noise(image,
                           limit,
                           is_gray=False,
                           speckle_size=1.0,
                           multiplicative=False):
    """
    Generate uniform distributed noise.

    Parameters
    ----------
    image: numpy image data, shape is [H, W, C], C is optional
    limit: limit of uniform random number
    is_gray: whether the noise is color or gray
    speckle_size: speckle size of noise
    multiplicative: whether to add additive noise or multiplicative noise

    Returns
    -------
    noise: uniform noise
    """
    noise_shape = get_noise_shape(image, speckle_size)

    # generate noise
    if is_gray:
        noise = np.random.uniform(-limit, limit, size=noise_shape[0:2])
        if image.ndim == 3:
            noise = np.expand_dims(noise, axis=2).repeat(3, axis=2)
    else:
        noise = np.random.uniform(-limit, limit, size=noise_shape)

    # resize noise to keep same size as image
    if speckle_size > 1.0:
        interp_type = np.random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR])
        noise = cv2.resize(noise, dsize=(image.shape[1], image.shape[0]), interpolation=interp_type)

    noise = noise.astype(np.float32)

    # make multiplicative noise if needed
    if multiplicative:
        if noise.ndim == 3:
            channel = noise.shape[2]
            noise *= np.float32(image[..., :channel])
        else:
            noise *= np.float32(image)

    return noise


def add_uniform_noise(image,
                      limit,
                      is_gray=False,
                      speckle_size=1.0,
                      noise_num_ratio=1.0,
                      multiplicative=False):
    """
    Add uniform noise to image.

    Parameters
    ----------
    image: numpy image data, shape is [H, W, C], C is optional
    limit: limit of uniform random number
    is_gray: whether the noise is color or gray
    speckle_size: speckle size of noise
    noise_num_ratio: ratio of noise number with respect to the total number of
        pixels, between [0, 1]
    multiplicative: whether to add additive noise or multiplicative noise

    Returns
    -------
    noisy_image: image after adding noise
    """
    if limit < 0:
        raise ValueError('limit must >= 0.0')

    noise = generate_uniform_noise(image,
                                   limit=limit,
                                   is_gray=is_gray,
                                   speckle_size=speckle_size,
                                   multiplicative=multiplicative)

    noisy_image = add_noise(image,
                            noise=noise,
                            noise_num_ratio=noise_num_ratio)
    return noisy_image


##################### 模糊 ############################
def add_gaussian_blur(image, sigmax, sigmaY):
    blur_image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=sigmax, sigmaY=sigmaY)

    return blur_image


######################### 增强 #############################
def clahe(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe


######################## 增强 ##############################

def lowlight(image):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('./augmentation/snapshots/Epoch99.pth'))
    _, enhanced_image, _ = DCE_net(image)
    return enhanced_image


if __name__ == '__main__':
    # choose one of the following parameter set
    MULTI = True
    SIGMA = 0.1
    LIMIT = 0.1

    # MULTI = False
    # SIGMA = 25

    # read original image / gray scale image / image with alpha channel
    data_dir_vis = 'D:\Work Files/a-SeAFusion-main\SeAFusion-main\MSRS/Visible/train/MSRS/00001D.png'
    image = cv2.imread(data_dir_vis)
    # image_gray = cv2.imread('lena512color.tiff', cv2.IMREAD_GRAYSCALE)

    # generate noisy image

    # noise_sigma = 200
    # noisy_image1 = add_gaussian_noise(image, noise_sigma)
    # cv2.imshow('original_image', image)
    # cv2.imshow('gaussian_noisy_image', noisy_image1)
    #
    # cv2.waitKey(0)

    # noisy_image2 = add_uniform_noise(image, LIMIT, False, 10.0, 1.0, MULTI)
    # cv2.imshow('original_image', image)
    # cv2.imshow('gaussian_noisy_image', noisy_image2)
    # cv2.waitKey(0)
    # # generate blur image
    # sigmaX = 2
    # sigmaY = 2
    # blur_image = add_gaussian_blur(image, sigmaX, sigmaY)
    # cv2.imshow('original_image', image)
    # cv2.imshow('gaussian_blur_image', blur_image)
    # cv2.waitKey(0)

    aug_image = lowlight(data_dir_vis)
    result_path = './augmentation/1.png'
    torchvision.utils.save_image(aug_image, result_path)
    cv2.waitKey(0)

    aug_image = clahe(image)
    cv2.imshow('original_image', image)
    cv2.imshow('clahe_augment_image', aug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
