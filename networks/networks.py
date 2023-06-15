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

def define_Encoder(Etype, in_channels, net_path = None, gpu_ids=[]) :
    if Etype == 'Unet4R':
        net = UEncoder4Recurrent(in_channels)
    elif Etype == 'Unet3R':
        net = UEncoder3Recurrent(in_channels)
    print('load ' + net_path)
    net.load_state_dict(torch.load(net_path))
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net

def define_Decoder(Dtype, out_channels, net_path = None, gpu_ids=[]) :
    if Dtype == 'Unet3R':
        net = UDecoder(out_channels)
    elif Dtype == 'Unet4R':
        net = UDecoder4(out_channels)
    
    print('load ' + net_path)
    net.load_state_dict(torch.load(net_path))
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net

def define_Fusion(n_features, net_path = None, gpu_ids=[]):
    net = FusionLayer_Unet(n_features)
    
    print('load ' + net_path)
    net.load_state_dict(torch.load(net_path))
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    
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