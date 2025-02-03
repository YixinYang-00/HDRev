"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import cv2 
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


# -------------------- Generate color dataset --------------
def multiple_crop(scale_size, crop_size, convert=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __multi_scale(img, scale_size)))
    # transform_list.append(transforms.Lambda(lambda img: __random_flip(img)))
    transform_list.append(transforms.Lambda(lambda img: __multi_crop(img, crop_size)))

    return transforms.Compose(transform_list)


# -----------get pair-wise transform----------------
def get_pairwise_transform(opt, params=None, convert=True):
    transform_list = []

    if 'resize' in opt.preprocess:
        osize = (opt.load_size, opt.load_size)
        transform_list.append(transforms.Lambda(lambda img: cv2.resize(img, osize)))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, opt.crop_size, isTrain=opt.isTrain)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_22(img, base=4)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.Lambda(lambda img: __random_flip(img)))

    if convert:
        transform_list.append(transforms.ToTensor())
        
    return transforms.Compose(transform_list)


# -----------get T transform for video regularization----------------
def T_transform(img):
    sx, sy = img.shape[:2]
    
    ang = np.deg2rad(1.0)
    tx = random.uniform(-2.0, 2.0)
    ty = random.uniform(-2.0, 2.0)
    r = random.uniform(-ang, ang)
    z = random.uniform(0.97, 1.03)
    hx = random.uniform(-ang, ang)
    hy = random.uniform(-ang, ang)
    
    a = hx - r
    b = hy + r
    T1 = np.true_divide(z*np.cos(a), np.cos(hx))
    T2 = np.true_divide(z*np.sin(a), np.cos(hx))
    T3 = np.true_divide(sx*np.cos(hx) - sx*z*np.cos(a) + 2*tx*z*np.cos(a) - sy*z*np.sin(a) + 2*ty*z*np.sin(a), 2*np.cos(hx))
    T4 = np.true_divide(z*np.sin(b), np.cos(hy))
    T5 = np.true_divide(z*np.cos(b), np.cos(hy))
    T6 = np.true_divide(sy*np.cos(hy) - sy*z*np.cos(b) + 2*ty*z*np.cos(b) - sx*z*np.sin(b) + 2*tx*z*np.sin(b), 2*np.cos(hy))
    
    T = np.array([[T1, T2, T3], 
                  [T4, T5, T6]])
    
    img_t = cv2.warpAffine(img, T, (sy,sx), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return img_t, T


def __random_flip(img, prob=0.5):
    if(random.random() > prob):
        return cv2.flip(img, 1)
    return img

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __make_power_22(img, base):
    ow, oh = img.shape[0], img.shape[1]
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return cv2.resize(img, (w, h))


def __scale_width(img, scale_range, method=cv2.INTER_LINEAR):
    range_low, range_high = scale_range[0], scale_range[-1]
    target_shorter = random.randint(range_low, range_high)
        
    oh, ow = img.shape[:2]
    if (min(oh, ow) == target_shorter):
        return img
    if ow >= oh:
        h = target_shorter
        w = int(target_shorter * ow / oh)
    else:
        w = target_shorter
        h = int(target_shorter * oh / ow)
        
    img_scaled = cv2.resize(img, (w, h), method)
    return img_scaled


def __crop(img, crop_size, isTrain=True):
    oh, ow = img.shape[:2]
    if isTrain:
        low_x = random.randint(0, oh-crop_size[0])
        low_y = random.randint(0, ow-crop_size[1])
    else:
        low_x = int((oh-crop_size[0]) / 2)
        low_y = int((ow-crop_size[1]) / 2)
    crop_img = img[low_x:low_x+crop_size[0], low_y:low_y+crop_size[1], :]
    return crop_img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __multi_crop(img, crop_size):
    oh, ow = img.shape[:2]
    longer = max(oh, ow)
    second_pos = int((longer-crop_size) / 2)

    if ow == longer:
        crop_img1 = img[0:crop_size, 0:crop_size, :]
        crop_img2 = img[0:crop_size, second_pos:second_pos+crop_size, :]
        crop_img3 = img[0:crop_size, -crop_size:, :]
        crop_img4 = img[-crop_size:, 0:crop_size, :]
        crop_img5 = img[-crop_size:, second_pos:second_pos+crop_size, :]
        crop_img6 = img[-crop_size:, -crop_size:, :]
    else:
        crop_img1 = img[0:crop_size, 0:crop_size, :]
        crop_img2 = img[second_pos:second_pos+crop_size, 0:crop_size, :]
        crop_img3 = img[-crop_size:, 0:crop_size, :]
        crop_img4 = img[0:crop_size, -crop_size: :]
        crop_img5 = img[second_pos:second_pos+crop_size, -crop_size:, :]
        crop_img6 = img[-crop_size:, -crop_size:, :]

    return [crop_img1, crop_img2, crop_img3, crop_img4, crop_img5, crop_img6]

def __multi_scale(img, scale_size):
    oh, ow = img.shape[:2]
    if (min(oh, ow) == scale_size):
        return img
    if ow >= oh:
        h = scale_size
        w = int(scale_size * ow / oh)
    else:
        w = scale_size
        h = int(scale_size * oh / ow)
        
    img_scaled = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
    return img_scaled

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
