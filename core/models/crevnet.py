__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.ConvLSTMCell import ConvLSTMCell
from mmcv.cnn import constant_init, kaiming_init
from typing import Sequence, Optional

import torch
from torch import nn, Tensor

from core.layers.PixelShuffle import PixelShuffle
from core.layers.ReversiblePredictiveModule import ReversiblePredictiveModule
from core.layers.i_RevNet_Block import i_RevNet_Block




class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        print("crevnet initial")

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        tcn = []
        transformer = []
        context_block = []
        #if channels_list is None:
        channels_list = [6, 3*8, 3*32]
        self.in_channels = self.frame_channel
        self.channels_list = channels_list
        self.n_blocks = len(channels_list)
        self.n_layers = 6
        self.conv_first = nn.Conv2d(self.frame_channel, 2,
                                    kernel_size=1, stride=1, padding=0, bias=False)

        self.auto_encoder = nn.ModuleList([])
        for i in range(self.n_blocks):
            self.auto_encoder.append(i_RevNet_Block(channels_list[i]))

        aim_width = width // (2 ** self.n_blocks)

        self.rpm = ReversiblePredictiveModule(channels=channels_list[-1], n_layers=self.n_layers, width=aim_width, num_slots=4)

        self.pixel_shuffle = PixelShuffle(n=2)



        self.MSE_criterion = nn.MSELoss().to(self.configs.device)

    def forward(self, inputs_origin, mask_true, train=True):
        inputs = inputs_origin.permute(0, 1, 4, 2, 3).contiguous()
        #mask = mask.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        out_len = 10
        device = self.configs.device
        batch, sequence, channel, height, width = inputs.shape

        h = []  # 存储隐藏层
        c = []  # 存储cell记忆
        pred = []  # 存储预测结果

        # 初始化最开始的隐藏状态
        for i in range(self.n_layers):
            zero_tensor_h = torch.zeros(batch, self.channels_list[-1], height // 2 ** self.n_blocks,
                                        width // 2 ** self.n_blocks).to(device)
            zero_tensor_c = torch.zeros(batch, self.channels_list[-1], height // 2 ** self.n_blocks,
                                        width // 2 ** self.n_blocks).to(device)
            h.append(zero_tensor_h)
            c.append(zero_tensor_c)

        m = torch.zeros(batch, self.channels_list[-1], height // 2 ** self.n_blocks,
                        width // 2 ** self.n_blocks).to(device)

        # 开始循环，模型在预测部分的输入是前一帧的预测输出
        for s in range(self.configs.total_length - 1):
            if s < self.configs.input_length:
                x = inputs[:, s]
            else:
                #x = x_pred
                x = mask_true[:, s - self.configs.input_length] * inputs[:, s] + \
                      (1 - mask_true[:, s - self.configs.input_length]) * x_pred

            #x = self.conv_first(x)

            x = self.pixel_shuffle.forward(x)
            x = torch.split(x, x.size(1) // 2, dim=1)

            for i in range(self.n_blocks - 1):
                x = self.auto_encoder[i].forward(x)
                x = [self.pixel_shuffle.forward(t) for t in x]
            x = self.auto_encoder[-1].forward(x)

            #x, h, c = self.rpm(x, c, h)
            x, h, c, m = self.rpm(x, h, c, m)

            for i in range(self.n_blocks - 1):
                x = self.auto_encoder[-1 - i].inverse(x)
                x = [self.pixel_shuffle.inverse(t) for t in x]

            x = self.auto_encoder[0].inverse(x)

            x = torch.cat(x, dim=1)

            x_pred = self.pixel_shuffle.inverse(x)

            #if s >= 5:
            pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(prediction, inputs_origin[:, 1:])

        return prediction, loss
