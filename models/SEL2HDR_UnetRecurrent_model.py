import os
import torch
import math
import torchvision.transforms as transforms
import scipy.stats as st
import numpy as np
import cv2
from .base_model import BaseModel
from networks import networks
from util.colorConvert import BGR2YCbCr, BGR2XYZ, XYZ2BGR
from util.util import tensor2im, make_event_preview

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis = 0)
    return out_filter

class SEL2HDRUnetRecurrentModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['ldr', 'images', 'tm_images']
        self.model_names = ['Encoder_E', 'Encoder_I', 'Merge', 'Decoder']
        
        self.num_bins = opt.num_bins
        
        self.statei = None
        self.statee = None

        weights = opt.load_weights
        assert weights in ['Unet3R', 'Unet4R']

        self.netEncoder_I = networks.define_Encoder(Etype=weights, in_channels = 3,
                                                    net_path = os.path.join('weights', weights, 'Encoder_I.pth'), gpu_ids=self.gpu_ids)
        self.netEncoder_E = networks.define_Encoder(Etype=weights, in_channels = opt.num_bins, 
                                                    net_path = os.path.join('weights', weights, 'Encoder_E.pth'), gpu_ids=self.gpu_ids)
        self.netMerge = networks.define_Fusion(n_features = int(weights[-2]),
                                                    net_path = os.path.join('weights', weights, 'Merge.pth'), gpu_ids=self.gpu_ids)
        self.netDecoder  = networks.define_Decoder(Dtype=weights, out_channels = 3,
                                                    net_path = os.path.join('weights', weights, 'Decoder.pth'), gpu_ids=self.gpu_ids)
        
        self.transform = transforms.ToTensor()
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.evs = input['ev']
        self.ldr = input['ldr']
        self.t = input['t']
        self.gt = input['hdr']

    def blur(self, x, kernel = 21, channels = 3, stride = 1, padding = 'same'):
        kernel_var = torch.from_numpy(gauss_kernel(kernel, 3, channels)).to(self.device).float()
        return torch.nn.functional.conv2d(x, kernel_var, stride = stride, padding = padding, groups = channels)
    
    def forward_each(self, evs, ldr, t):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        Y_ldr = BGR2YCbCr(ldr)[:, 0:1, :, :]
        mask_ldr = (0.5 - torch.maximum(torch.abs(Y_ldr - 0.5), torch.ones_like(Y_ldr) * (0.8 - 0.5))) / (1 - 0.8)
        mask_evs = torch.clamp(self.blur(self.blur(torch.sum(torch.abs(evs), axis = 1, keepdim = True), kernel = 7, channels = 1), 
                                            kernel = 7, channels = 1), 0, 1)
        ldr = torch.pow(ldr, 2.2) / t
        feat_i, self.statei = self.netEncoder_I(ldr, self.statei)
        feat_e, self.statee = self.netEncoder_E(evs, self.statee)
        
        feat_ec, feat_ic = [], []
        for i in range(len(feat_e)):
            mask_ldr = self.blur(mask_ldr, kernel = 3, channels = 1, padding = 1, stride = 1)
            mask_ldr = self.blur(mask_ldr, kernel = 3, channels = 1, padding = 1, stride = 1)

            mask_evs = self.blur(mask_evs, kernel = 3, channels = 1, padding = 1, stride = 1)
            mask_evs = self.blur(mask_evs, kernel = 3, channels = 1, padding = 1, stride = 1)

            feat_ec.append(mask_evs * feat_e[i])
            feat_ic.append(mask_ldr * feat_i[i])

            mask_ldr = self.blur(mask_ldr, kernel = 5, channels = 1, padding = 2, stride = 2)
            mask_evs = self.blur(mask_evs, kernel = 5, channels = 1, padding = 2, stride = 2)   
        
        feat = self.netMerge(feat_ec, feat_ic)

        img = self.netDecoder(feat)
        return img

    def forward(self):
        self.images = []
        self.statei, self.statee, self.pevs = None, None, 0
        for i in range(len(self.evs)):
            self.statei, self.statee, self.pevs = None, None, 0
            for j in range(max(0, i - self.opt.num_img + 1), i):
                img = self.forward_each(self.evs[j].to(self.device).float(), self.ldr[j].to(self.device).float(), self.t[j].to(self.device).float())
                self.images[j] += img
            img = self.forward_each(self.evs[i].to(self.device).float(), self.ldr[i].to(self.device).float(), self.t[i].to(self.device).float())
            self.images.append(img)
        for i in range(len(self.images)):
            self.images[i] /= min(self.opt.num_img, len(self.images) - i)

    def tonemap(self, img, log_sum_prev=None):
        key_fac, epsilon, tm_gamma = 0.5, 1e-6, 1.4
        XYZ = BGR2XYZ(img)
        b, c, h, w = XYZ.shape
        if log_sum_prev is None:
            log_sum_prev = torch.log(epsilon + XYZ[:, 0, :,:]).sum((1, 2), keepdim=True)
            log_avg_cur = torch.exp(log_sum_prev / (h * w))
            key = key_fac
        else:
            log_sum_cur = torch.log(XYZ[:, 1, :,:] + epsilon).sum((1, 2), keepdim=True)
            log_avg_cur = torch.exp(log_sum_cur / (h * w))
            log_avg_temp = torch.exp((log_sum_cur + log_sum_prev) / (2.0 * h * w))
            key = key_fac * log_avg_cur / log_avg_temp
            log_sum_prev = log_sum_cur
        Y = XYZ[:, 1, :, :]
        Y = Y / log_avg_cur * key
        Lmax =  torch.max(torch.max(Y, 1, keepdim=True)[0], 2, keepdim=True)[0]
        L_white2 = Lmax * Lmax
        L = Y * (1 + Y / L_white2) / (1 + Y)
        XYZ *= (L / XYZ[:, 1, :, :]).unsqueeze(1)
        image = XYZ2BGR(XYZ)
        image = torch.clamp(image, 0, 1) ** (1 / tm_gamma)
        return image, log_sum_prev

    def compute_visuals(self):
        t_im = None
        t_gt = None
        self.tm_images, self.tm_gt = self.images, self.gt
        for i in range(len(self.images)):
            self.images[i] = torch.clamp(self.images[i], 0, 1)   
            self.tm_images[i], t_im = self.tonemap(self.images[i], t_im)
            self.tm_gt[i], t_gt = self.tonemap(self.gt[i], t_gt)
            self.gt[i] = torch.clamp(self.gt[i], 0, 1)    
        
