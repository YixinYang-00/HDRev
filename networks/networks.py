import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math 
from .submodules import *
from .Unet import *
###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02, pretrain=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if m.__class__ == list :
            print('no')
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def define_Encoder(Etype, in_channels, net_path = None, train=True, init_type='normal', init_gain=0.02, gpu_ids=[]) :
    if Etype == 'Unet4R':
        net = UEncoder4Recurrent(in_channels)
    elif Etype == 'Unet3R':
        net = UEncoder3Recurrent(in_channels)
    if net_path is not None:
        print('load ' + net_path)
        net.load_state_dict(torch.load(net_path))
    else:
        init_weights(net, init_type, init_gain)
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if not train:
        for p in net.parameters():
            p.requires_grad = False
    return net

def define_Decoder(Dtype, out_channels, net_path = None, train=True, init_type='normal', init_gain=0.02, gpu_ids=[]) :
    if Dtype == 'Unet3R':
        net = UDecoder(out_channels)
    elif Dtype == 'Unet4R':
        net = UDecoder4(out_channels)
    
    if net_path is not None:
        print('load ' + net_path)
        net.load_state_dict(torch.load(net_path))
    else:
        init_weights(net, init_type, init_gain)
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if not train:
        for p in net.parameters():
            p.requires_grad = False
    return net

def define_Fusion(n_features, net_path = None, train=True, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = FusionLayer_Unet(n_features)
    
    if net_path is not None:
        print('load ' + net_path)
        net.load_state_dict(torch.load(net_path))
    else:
        init_weights(net, init_type, init_gain)
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    
    if not train:
        for p in net.parameters():
            p.requires_grad = False
    return net

##############################################################################
# Classes
##############################################################################

class WeightBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, downsample=None, norm=None):
        super(WeightBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=bias)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)
        return self.act(out)

class FusionLayer_Unet(nn.Module):
    def __init__(self, n_features):
        super(FusionLayer_Unet, self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.n_features = n_features
        self.w1 = nn.ModuleList()
        self.w2 = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(n_features):
            self.w1.append(WeightBlock(in_channels = filters[i] * 2, out_channels = filters[i]))
            self.w2.append(WeightBlock(in_channels = filters[i] * 2, out_channels = filters[i]))
            self.fuse.append(SELayer(channel = filters[i] * 2))
            self.conv.append(ConvLayer(in_channels = filters[i] * 2, out_channels = filters[i], kernel_size = 3, padding = 1, activation = None))
        
    def forward(self, fe, fi):
        if self.n_features != len(fe) or self.n_features != len(fi):
            print("The number of features does not match the input features of FusionLayer")
        out = []
        for i in range(self.n_features):
            cat = self.fuse[i](torch.cat([fe[i], fi[i]], dim = 1))
            f1 = fe[i] * self.w1[i](cat)
            f2 = fi[i] * self.w2[i](cat)
        
            out.append(self.conv[i](torch.cat([f1, f2], dim = 1)))
        return out

class FusionLayer_Conv(nn.Module):
    def __init__(self, n_features):
        super(FusionLayer_Conv, self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.n_features = n_features
        self.fuse = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(n_features):
            self.fuse.append(SELayer(channel = filters[i] * 2))
            self.conv.append(ConvLayer(in_channels = filters[i] * 2, out_channels = filters[i], kernel_size = 3, padding = 1, activation = None))
        
    def forward(self, fe, fi):
        if self.n_features != len(fe) or self.n_features != len(fi):
            print("The number of features does not match the input features of FusionLayer")
        out = []
        for i in range(self.n_features):
            out.append(self.conv[i](torch.cat([fe[i], fi[i]], dim = 1)))
        return out