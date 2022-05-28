__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.LayerNorm import*

class ResidualAdapter(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ResidualAdapter, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            #nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
        #    nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            #nn.LayerNorm([num_hidden * 3, width, width])
        )

        self.conv_M = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            #    nn.LayerNorm([num_hidden * 3, width, width])
        )

        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 3, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 3, num_hidden, kernel_size=1, stride=1, padding=0)

        self.gate_x = nn.Sequential(
            nn.Conv2d(1 * num_hidden, 2 * num_hidden,
                      kernel_size=1, stride=1, padding=0, bias=True),
            # nn.LayerNorm([num_hidden * 2, width, width])
        )

        self.gate_memory = nn.Sequential(
            nn.Conv2d(1 * num_hidden, 2 * num_hidden,
                      kernel_size=1, stride=1, padding=0, bias=True),
            # nn.LayerNorm([num_hidden * 2, width, width])
        )

        self.layer_norm_m = DynamicLayerNorm([num_hidden * 1, num_hidden * 1, width, width])

    def forward(self, x_t, m_t):
        x_concat = self.conv_x(x_t)
        m_concat = self.conv_m(m_t)

        feature_x = self.gate_x(m_t)
        mean_x, sigma_x = torch.split(feature_x, self.num_hidden, dim=1)



        # memory_global = self.ConvGRU_global(x_t, memory_global)
        # mean_c, sigma = self.convlstm(x_t, h_t, c_t)



        i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        #i_M, f_M, g_M = torch.split(m_concat_global, self.num_hidden, dim=1)



        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime
        m_new = self.layer_norm_m(m_new, mean_x, sigma_x)


        return m_new









