__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.st_slot_module import ModeRNNCell
import numpy as np
import matplotlib.pyplot as plt

import math



class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        print("convlstm initial")

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        num_slots = 4

        self.count = 1
        self.show = 1
        self.num = 200



        for i in range(num_layers):
            cell_list.append(
                ModeRNNCell(num_hidden[-1], width, num_slots)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                               kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv_first = nn.Conv2d(self.frame_channel, num_hidden[num_layers - 1], 2, 2)
        self.conv_first = nn.Conv2d(self.frame_channel, num_hidden[num_layers - 1],
                                    kernel_size=1, stride=1, padding=0, bias=False)

        self.MSE_criterion = nn.MSELoss().to(self.configs.device)


    def forward(self, all_frames, mask_true, train=True):
        # self.inner_update(frames, mask_true)
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = all_frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_block = []
        c_t_block = []
        h_inner = []
        concat_block = [[], [], [], []]
        context = [[]] * 4
        memory = torch.zeros([batch, self.num_hidden[-1], height, width]).to(self.configs.device)
        m_t = []
        z_t = torch.zeros([batch, self.num_hidden[-1], height, width]).to(self.configs.device)
        slots = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)

            h_t.append(zeros)
            c_t.append(zeros)
            slots.append(zeros)


        for t in range(self.configs.total_length - 1):
            h_block = []
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            net_input = self.conv_first(net)
            #print(net_input.size())

            h_t[0], slots[0] = self.cell_list[0](net_input, slots[0], h_t[0], 1)

            for i in range(1, self.num_layers):
                h_t[i], slots[i] = self.cell_list[i](h_t[i - 1], slots[i], h_t[i], i)



            x_gen = self.conv_last(h_t[-1])
            next_frames.append(x_gen)


        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, all_frames[:, 1:])

        return next_frames, loss
