import os
import torch
import math
import torchvision.transforms as transforms
import scipy.stats as st
import numpy as np
from .base_model import BaseModel
from networks import networks
from .vgg import Vgg16
from util.util import tensor2im
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from util.colorConvert import *

def normalize_batch(batch, final_func='sig'):
    # normalize using imagenet mean and std
    if final_func == 'tanh':
        batch = (batch + 1.0) / 2.0
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    return (batch - mean) / std

def Gram_matrix(input):
    a,b,c,d = input.size()
    features = input.view(a*b, c*d)
    G = torch.mm(features, features.t())

    return G.div(a*b*c*d)

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
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        if 'l2' in self.opt.loss_type:
            self.loss_names.append('L2')
        if 'vgg' in self.opt.loss_type:
            self.loss_names.append('perc')
        if 'color' in self.opt.loss_type:
            self.loss_names.append('color')

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['evs', 'ldr', 'images']
        if self.isTrain:
            self.visual_names.append('gt')

        self.metric_names = ['psnr', 'ssim']
        self.model_names = ['Encoder_E', 'Encoder_I', 'Merge', 'Decoder']
        
        self.statei = None
        self.statee = None

        weights = opt.load_weights
        assert weights in ['Unet3R', 'Unet4R']

        self.netEncoder_I = networks.define_Encoder(Etype=weights, in_channels = 3, train = True,
                                                    net_path = os.path.join('weights', weights, 'Encoder_I.pth'), gpu_ids=self.gpu_ids)
        self.netEncoder_E = networks.define_Encoder(Etype=weights, in_channels = opt.num_bins, train = True,
                                                    net_path = os.path.join('weights', weights, 'Encoder_E.pth'), gpu_ids=self.gpu_ids)
        self.netMerge = networks.define_Fusion(n_features = int(weights[-2]), train = True,
                                                    net_path = os.path.join('weights', weights, 'Merge.pth'), gpu_ids=self.gpu_ids)
        self.netDecoder  = networks.define_Decoder(Dtype=weights, out_channels = 3, train = True,
                                                    net_path = os.path.join('weights', weights, 'Decoder.pth'), gpu_ids=self.gpu_ids)
        
        self.transform = transforms.ToTensor()
        if self.isTrain:
            # define loss functions
            if 'l2' in self.opt.loss_type:
                self.criterionL2 = torch.nn.MSELoss()
            if 'vgg' in self.opt.loss_type:
                self.vgg = Vgg16(requires_grad=False).to(self.device)

            self.optimizer_Encoder_I = torch.optim.Adam(self.netEncoder_I.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Encoder_I)
            self.optimizer_Encoder_E = torch.optim.Adam(self.netEncoder_E.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Encoder_E)
            self.optimizer_Merge = torch.optim.Adam(self.netMerge.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Merge)
            self.optimizer_Decoder = torch.optim.Adam(self.netDecoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Decoder)

    def toneMapping(self, img, tmo_name = None) :
        u = 5000
        return torch.log(1 + u * img) / math.log(1 + u)

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
    
    def backward(self, imgs, gts):
        """Calculate loss"""
        img = self.toneMapping(imgs[-1])
        gt = self.toneMapping(gts[-1])
        self.loss_total = 0
            
        if 'l2' in self.opt.loss_type:
            self.loss_L2 = self.criterionL2(img, gt) * self.opt.lambda_L2
            self.loss_total += self.loss_L2
        
        if 'vgg' in self.opt.loss_type:
            self.loss_perc = 0
            predict_features = self.vgg(normalize_batch(img))
            target_features = self.vgg(normalize_batch(gt))
            for f_x, f_y in zip(predict_features, target_features):
                self.loss_perc += torch.mean((f_x - f_y)**2)
                G_x = Gram_matrix(f_x)
                G_y = Gram_matrix(f_y)
                self.loss_perc += torch.mean((G_x - G_y)**2)
            self.loss_perc = self.loss_perc * self.opt.lambda_perc
            self.loss_total += self.loss_perc
        
        if 'color' in self.opt.loss_type:
            img_b = self.blur(img)
            gt_b  = self.blur(gt)
            self.loss_color = torch.mean(torch.sum(torch.pow(gt_b - img_b, 2), (-1, -2, -3)) / 2)  * self.opt.lambda_color
            self.loss_total += self.loss_color

        self.loss_total.backward()

    def forward(self):
        self.images = []
        self.statei, self.statee = None, None
        for i in range(len(self.evs)):
            self.statei, self.statee = None, None
            for j in range(max(0, i - self.opt.num_img + 1), i):
                img = self.forward_each(self.evs[j].to(self.device).float(), self.ldr[j].to(self.device).float(), self.t[j].to(self.device).float())
                self.images[j] += img
            img = self.forward_each(self.evs[i].to(self.device).float(), self.ldr[i].to(self.device).float(), self.t[i].to(self.device).float())
            self.images.append(img)
        for i in range(len(self.images)):
            self.images[i] /= min(self.opt.num_img, len(self.images) - i)

    def optimize_parameters(self):
        self.statei, self.statee = None, None
        imgs, gts = [], []
        for evs, ldr, gt, t in zip(self.evs, self.ldr, self.gt, self.t):
            self.images = self.forward_each(evs.to(self.device).float(), ldr.to(self.device).float(), t.to(self.device).float())
            imgs.append(self.images)
            gts.append(gt.to(self.device).float())

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        self.backward(imgs, gts)

        for optimizer in self.optimizers:
            optimizer.step()
        
        self.images = imgs
        self.gt = self.gt
        
    def tonemap(self, img, log_sum_prev=None):
        key_fac, epsilon, tm_gamma = 0.5, 1e-6, 1.4
        # print(img.shape)
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
            self.tm_images[i], t_im = self.tonemap(self.images[i], t_im)
            self.images[i] = torch.clamp(self.images[i], 0, 1)   
            self.tm_gt[i], t_gt = self.tonemap(self.gt[i], t_gt)
            self.gt[i] = torch.clamp(self.gt[i], 0, 1)    

    def compute_metrics(self):
        hdr_np = tensor2im(self.images[-1])
        gt_np = tensor2im(self.gt[-1])
        self.metric_psnr = compare_psnr(gt_np, hdr_np)
        self.metric_ssim = compare_ssim(gt_np, hdr_np, gaussian_weights=True, channel_axis=2)
        
    
        
