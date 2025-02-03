"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    return (batch - mean) / std


def make_event_preview(events, color=False):
    event_preview = events[0, :, :, :].detach().cpu().numpy()
    event_preview = np.transpose(event_preview, (1, 2, 0))
    # event_preview = np.sum(event_preview, axis=0)

    # normalize event image to [0, 255] for display
    m, M = -10.0, 10.0
    event_preview = np.clip((255.0 * (event_preview - m) / (M - m)).astype(np.uint8), 0, 255)
    shape = event_preview.shape
    if shape[2] != 3:
        ones = np.ones((shape[0], shape[1], 3 - shape[2])) * 128
        event_preview = np.concatenate((event_preview, ones), axis = 2)
    if color:
        event_preview = np.dstack([event_preview] * 3)

    return event_preview


def tensor2im(input_image, imtype=np.uint8):
    """"
    Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array        
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        
        # from [-1,1] to [0,1]
        # image_numpy = ((image_numpy+1.0)/2.0)*255  # post-processing: tranpose and scaling
        
        image_numpy = image_numpy*255

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image[0]
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def add_PG_noise(img, sigma_s='RAN', sigma_c='RAN'):
    min_log = np.log([0.0001])
    if sigma_s == 'RAN':
        sigma_s = min_log + np.random.rand(1) * (np.log([0.001]) - min_log)
        sigma_s = np.exp(sigma_s)
    if sigma_c == 'RAN':
        sigma_c = min_log + np.random.rand(1) * (np.log([0.0005]) - min_log)
        sigma_c = np.exp(sigma_c)
    # add noise
    # print('Adding Noise: sigma_s='+str(sigma_s*255)+' sigma_c='+str(sigma_c*255))
    sigma_total = np.sqrt(sigma_s * img + sigma_c)
    '''
    noisy_img = img + \
    np.random.normal(0.0, 1.0, img.shape) * (sigma_s * img) + \
    np.random.normal(0.0, 1.0, img.shape) * sigma_c
    '''
    noisy_img = img +  \
        sigma_total * np.random.randn(img.shape[-2], img.shape[-1])
    noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img, sigma_s, sigma_c