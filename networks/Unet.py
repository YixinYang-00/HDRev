from .submodules import *
import torch.nn as nn

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding=1, activation='relu', norm=None) :
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm),
            ConvLayer(out_channels, out_channels, kernel_size, stride, padding, activation, norm)
        )
    def forward(self, x):
        return self.conv(x)

class UDecoder(nn.Module):
    def __init__(self, out_channels=3):
        super(UDecoder, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up3 = UpsampleConvLayer(filters[2], filters[1], kernel_size=3, padding=1, norm='IN')
        self.Up_conv3 = Conv2DBlock(filters[2], filters[1], padding=1, norm='IN')

        self.Up2 = UpsampleConvLayer(filters[1], filters[0], kernel_size=3, padding=1, norm='IN')
        self.Up_conv2 = Conv2DBlock(filters[1], filters[0], padding=1, norm='IN')

        self.Conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()


    def forward(self, e):
        e1, e2, e3 = e[0], e[1], e[2]
        
        d3 = self.Up3(e3, e2.shape[2:])
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3, e1.shape[2:])
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return self.act(out)

class UDecoder4(nn.Module):
    def __init__(self, out_channels=3):
        super(UDecoder4, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up4 = UpsampleConvLayer(filters[3], filters[2], kernel_size=5, padding=2, norm='IN')
        self.Up_conv4 = Conv2DBlock(filters[3], filters[2], norm='IN')

        self.Up3 = UpsampleConvLayer(filters[2], filters[1], kernel_size=5, padding=2, norm='IN')
        self.Up_conv3 = Conv2DBlock(filters[2], filters[1], norm='IN')

        self.Up2 = UpsampleConvLayer(filters[1], filters[0], kernel_size=5, padding=2, norm='IN')
        self.Up_conv2 = Conv2DBlock(filters[1], filters[0], norm='IN')

        self.Conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()


    def forward(self, e):
        e1, e2, e3, e4 = e[0], e[1], e[2], e[3]

        d4 = self.Up4(e4, e3.shape[2:])
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4, e2.shape[2:])
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3, e1.shape[2:])
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return self.act(out)


class UEncoder3Recurrent(nn.Module):
    def __init__(self, in_channels=3):
        super(UEncoder3Recurrent, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = RecurrentConvLayer(filters[0], filters[0], kernel_size=5, padding=2, stride=2, recurrent_block_type='convlstm')#nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = RecurrentConvLayer(filters[1], filters[1], kernel_size=5, padding=2, stride=2, recurrent_block_type='convlstm')#nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv2DBlock(in_channels, filters[0], norm='IN')
        self.Conv2 = Conv2DBlock(filters[0], filters[1], norm='IN')
        self.Conv3 = Conv2DBlock(filters[1], filters[2], norm='IN')


    def forward(self, x, prev_states):

        e1 = self.Conv1(x)
        if prev_states is None:
            prev_states = [None] * 2

        e2, state2 = self.Maxpool1(e1, prev_states[0])
        # e2 = nn.functional.relu(e2)
        e2 = self.Conv2(e2)

        e3, state3 = self.Maxpool2(e2, prev_states[1])
        # e3 = nn.functional.relu(e3)
        e3 = self.Conv3(e3)

        return [e1, e2, e3 ], [state2, state3]


class UEncoder4Recurrent(nn.Module):
    def __init__(self, in_channels=3):
        super(UEncoder4Recurrent, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = RecurrentConvLayer(filters[0], filters[0], kernel_size=5, padding=2, stride=2, recurrent_block_type='convlstm')#nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = RecurrentConvLayer(filters[1], filters[1], kernel_size=5, padding=2, stride=2, recurrent_block_type='convlstm')#nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = RecurrentConvLayer(filters[2], filters[2], kernel_size=5, padding=2, stride=2, recurrent_block_type='convlstm')#nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv2DBlock(in_channels, filters[0], norm='IN')
        self.Conv2 = Conv2DBlock(filters[0], filters[1], norm='IN')
        self.Conv3 = Conv2DBlock(filters[1], filters[2], norm='IN')
        self.Conv4 = Conv2DBlock(filters[2], filters[3], norm='IN')


    def forward(self, x, prev_states):

        e1 = self.Conv1(x)
        if prev_states is None:
            prev_states = [None] * 4

        e2, state2 = self.Maxpool1(e1, prev_states[0])
        e2 = self.Conv2(e2)

        e3, state3 = self.Maxpool2(e2, prev_states[1])
        e3 = self.Conv3(e3)
        
        e4, state4 = self.Maxpool3(e3, prev_states[2])
        e4 = self.Conv4(e4)

        return [e1, e2, e3, e4], [state2, state3, state4]
