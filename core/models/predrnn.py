__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
from core.layers.SpatioTemporalLSTMCell import DynamicConv
from core.layers.SpatioTemporalLSTMCell_dynamic import SpatioTemporalLSTMCell_dynamic
from core.layers.STConvGRU import STConvGRUCell
from core.layers.LightConvLSTMCell import ConvLSTMCell
import numpy as np
import matplotlib.pyplot as plt

import math




class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        dynamic_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            #in_channel = num_hidden[-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
            dynamic_list.append(DynamicConv(num_hidden[-1], num_hidden[-1] * 1, WH=1,M=3,G=1,r=2))
        self.cell_list = nn.ModuleList(cell_list)
        self.dynamic_list = nn.ModuleList(dynamic_list)
        self.conv_first = nn.Conv2d(self.frame_channel, num_hidden[num_layers-1], 
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.MSE_criterion = nn.MSELoss().to(self.configs.device)

        self.gru = STConvGRUCell(in_channel, num_hidden[0]*1, width, configs.filter_size,
                                       configs.stride, configs.layer_norm)

        self.save_check_0 = []
        self.save_check_1 = []
        self.save_check_2 = []
        self.save_check_3 = []
        self.save_check = []
        self.count = 1
        self.show = 1
        self.num = 200

    def forward(self, all_frames, mask_true, train=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = all_frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        c_t_block = []
        h_t_block = []
        f_t_block = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0]*1, height, width]).to(self.configs.device)

        for t in range(self.configs.total_length-1):
            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            #print(net.size())
            '''
            net_input = self.conv_first(net)
            for k in range(4):
                memory = self.gru(net_input, memory)
                if k == 0:
                    fea = memory
                else:
                    fea = torch.cat([fea, memory], dim=1)
            '''

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            #z_t = self.dynamic_list[0](h_t[0])
            c_t_block.append(c_t[0])
            h_t_block.append(h_t[0])

            #'''
            for i in range(1, self.num_layers):
                '''
                memory = self.gru(h_t[i-1], memory)
                for k in range(4):
                    memory = self.gru(net_input, memory)
                    if k == 0:
                        fea = memory
                    else:
                        fea = torch.cat([fea, memory], dim=1)
                '''
                #z_t = self.dynamic_list[i](h_t[i-1])
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        #c_list = torch.stack(c_t_block, dim=0)[:, 0, :, :, :]
        #c_key = torch.mean(torch.mean(torch.mean(c_list, dim=0), 1), 1)
        #if self.show == 1 and self.count < (self.num + 1):
        #    print(self.count)
        #    self.count += 1
        #    self.save_check.append(c_key.cpu().numpy())

        #if self.show == 1 and self.count == (self.num + 1):
        #    self.count += 1
        #    self.save_check = np.array(self.save_check)
        #    print(self.save_check.shape)
        #    np.save('/workspace/yaozhiyu/robonet_tsne/berkeley_sota_1.npy', self.save_check)
        #    print('save')


        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, all_frames[:, 1:])

        return next_frames, loss
