import torch
import numpy as np
import math

def BGR2YCbCr(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))
    
    b = image[..., 0, :, :]
    g = image[..., 1, :, :]
    r = image[..., 2, :, :]

    yr = .299 * r + .587 * g + .114 * b
    cb = (b - yr) * .564
    cr = (r - yr) * .713
    return torch.stack((yr, cb, cr), -3)

def YCbCr2BGR(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))
    
    yr = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    r = yr + 1.403 * cr
    g = yr - .344 * cb - .714 * cr
    b = yr + 1.770 * cb
    return torch.stack((b, g, r), -3)

def BGR2XYZ(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))
    
    b = image[..., 0, :, :]
    g = image[..., 1, :, :]
    r = image[..., 2, :, :]
    
    X = (0.4124 * r) + (0.3576 * g) + (0.1805 * b)
    Y = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    Z = (0.0193 * r) + (0.1192 * g) + (0.9505 * b)

    return torch.stack((X, Y, Z), -3)

def XYZ2BGR(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))
    
    X = image[..., 0, :, :]
    Y = image[..., 1, :, :]
    Z = image[..., 2, :, :]

    r = (3.240625 * X) + (-1.537208 * Y) + (-0.498629 * Z)
    g = (-0.968931 * X) + (1.875756 * Y) + (0.041518 * Z)
    b = (0.055710 * X) + (-0.204021 * Y) + (1.056996 * Z)
    
    return torch.stack((b, g, r), -3)

