import torch
import torch.nn as nn
from core.layers.LayerNorm import*
from mmcv.cnn import constant_init, kaiming_init

import torch.nn.functional as F
import random
from einops import rearrange



def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)



class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        #self.ratio = ratio
        self.planes = planes
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.planes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            #out = out * channel_mul_term
            return  channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out




class DynamicConvOrigin_1(nn.Module):
    def __init__(self, in_planes, features, dim, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(DynamicConvOrigin_1, self).__init__()
        d = max(int(features / r), L)
        self.M = (M + 1) * 1
        self.features = features
        self.convs = nn.ModuleList([])
        self.convx = nn.Conv2d(in_planes,
                               features,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=G)

        self.convs.append(nn.Sequential(
            nn.Conv2d(in_planes,
                      features,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=G),
            # nn.Conv2d(features,
            #          features,
            #          kernel_size=1,
            #          stride=1,
            #          padding=0,
            #          groups=G),
        )
        )

        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_planes,
                              features,
                              kernel_size=(3 + 0 * 2, 3),
                              stride=stride,
                              padding=(1 + 0, 1),
                              groups=G),
                    # nn.Conv2d(features,
                    #          features,
                    #          kernel_size=(1, 3 + 0 * 2),
                    #          stride=stride,
                    #          padding=(0, 1 + 0),
                    #          groups=G),

                )
            )

            '''
                nn.Sequential(
                    nn.Conv2d(in_planes // 4,
                              features,
                              kernel_size=(3 + i * 2, 1),
                              stride=stride,
                              padding=(1 + i, 0),
                              groups=G),
                    nn.Conv2d(features,
                              features,
                              kernel_size=(1, 3 + i * 2),
                              stride=stride,
                              padding=(0, 1 + i),
                              groups=G),

                )
            '''
        # self.gap = nn.AvgPool2d(int(WH/stride))
        # self.fc = nn.Linear(inplanes, d)
        # self.fc = nn.GRUCell(features, d)
        self.fc = ContextBlock(dim, d)
        # self.MMOE = SKLinear(in_planes, features, self.M)
        self.fcs = nn.ModuleList([])
        self.fcx = nn.Linear(d, features)

        # self.transformer = AttModel(hp, 10000, 10000)
        for i in range(self.M):
            self.fcs.append(nn.Linear(d, features))
            # self.fcs.append(SKLinear(d, features, 3))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, out, x):
        channel_num = out.size(1) // self.M
        # feas = out
        feax = self.convx(x)

        for i, conv in enumerate(self.convs):
            # if i == 1:
            #    continue
            fea = conv(out[:, i]).unsqueeze_(dim=1)
            # fea = conv(x[:, :, i]).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
                # break


            else:
                # break
                feas = torch.cat([feas, fea], dim=1)

            # break

        '''

        if not test_adapt:
            for i, conv in enumerate(self.convs):
                fea = conv(x[:, channel_num * i: (i + 1) * channel_num]).unsqueeze_(dim=1)
                # fea = conv(x[:, :, i]).unsqueeze_(dim=1)
                if i == 0:
                    feas = fea
                else:
                    feas = torch.cat([feas, fea], dim=1)
        else:
            with torch.no_grad():
                for i, conv in enumerate(self.convs):
                    fea = conv(x[:, channel_num * i: (i + 1) * channel_num]).unsqueeze_(dim=1)
                    # fea = conv(x[:, :, i]).unsqueeze_(dim=1)
                    if i == 0:
                        feas = fea
                    else:
                        feas = torch.cat([feas, fea], dim=1)
            feas = feas.detach_()
        '''

        # fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_():q
        # fea_s = x.mean(-1).mean(-1)
        # fea_s = fea_U

        # fea_z = self.fc(x).squeeze(-1).squeeze(-1)

        # attention_vectors = self.MMOE(fea_s)
        # print(fea_z.size())
        # fea_z = self.fc(x)
        # fea_v = (feas * fea_z).sum(dim=1)

        # print(x.size())

        fea_z = self.fc(x).squeeze(-1).squeeze(-1)
        vector_x = self.fcx(fea_z)
        attention_vectorx = torch.sigmoid(vector_x).unsqueeze(-1).unsqueeze(-1)
        fea_vx = feax * attention_vectorx
        # fea_z = y.squeeze(-1).squeeze(-1)

        for i, fc in enumerate(self.fcs):
            # if i == 1:
            #    continue
            # print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
                # break

            else:
                # break
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)

        attention_vectors = torch.sigmoid(attention_vectors)
        # print(attention_vectors.size(), feas.size())
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)

        feas = torch.relu(feas)
        # print(feas.size(), attention_vectors.size(), out.size())
        fea_v = (feas * attention_vectors).sum(dim=1)
        # fea_v = (feas * attention_vectors).sum(dim=1)
        # fea_v = (feas * attention_vectors).squeeze(1)
        # print(fea_v.size())

        return fea_v


class DynamicConvOrigin(nn.Module):
    def __init__(self, in_planes, features, dim, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(DynamicConvOrigin, self).__init__()
        d = max(int(features / r), L)
        self.M = (M + 1)*1
        self.features = features
        self.convs = nn.ModuleList([])
        self.convx = nn.Conv2d(in_planes,
                      features,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=G)

        self.convs.append(nn.Sequential(
            nn.Conv2d(in_planes // self.M,
                      features,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=G),
            # nn.Conv2d(features,
            #          features,
            #          kernel_size=1,
            #          stride=1,
            #          padding=0,
            #          groups=G),
        )
        )

        for i in range(M):
            self.convs.append(
               nn.Sequential(
                    nn.Conv2d(in_planes // self.M,
                              features,
                              kernel_size=(3 + 0 * 2, 3),
                              stride=stride,
                              padding=(1 + 0, 1),
                              groups=G),
                    #nn.Conv2d(features,
                    #          features,
                    #          kernel_size=(1, 3 + 0 * 2),
                    #          stride=stride,
                    #          padding=(0, 1 + 0),
                    #          groups=G),

                ) 
            )
        

        self.fc = ContextBlock(dim, d)
        self.fcs = nn.ModuleList([])
        self.fcx = nn.Linear(d, features)
        for i in range(self.M):
            self.fcs.append(nn.Linear(d, features))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, out, x):
        channel_num = out.size(1) // self.M
        # feas = out
        feax = self.convx(x)

        for i, conv in enumerate(self.convs):
            # if i == 1:
            #    continue
            fea = conv(out[:, i]).unsqueeze_(dim=1)
            # fea = conv(x[:, :, i]).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
                # break


            else:
                # break
                feas = torch.cat([feas, fea], dim=1)

            # break




        fea_z = self.fc(x).squeeze(-1).squeeze(-1)
        vector_x = self.fcx(fea_z)
        attention_vectorx = torch.sigmoid(vector_x).unsqueeze(-1).unsqueeze(-1)
        fea_vx = feax*attention_vectorx
        # fea_z = y.squeeze(-1).squeeze(-1)

        for i, fc in enumerate(self.fcs):
            # if i == 1:
            #    continue
            # print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
                # break

            else:
                # break
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)

        attention_vectors = torch.sigmoid(attention_vectors)
        # print(attention_vectors.size(), feas.size())
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)

        feas = torch.relu(feas)
        # print(feas.size(), attention_vectors.size(), out.size())
        fea_v = (feas * attention_vectors).sum(dim=1) + fea_vx
        return fea_v


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0


        self.conv = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, dilation=1),
            nn.LayerNorm([num_hidden * 4, width, width])
        )



    def forward(self, x, h, c, test_adapt=False):
        # print(x.size(), h.size())
        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis
        # print(self.attention2d(combined).size())

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.num_hidden, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)



        c_next = f * c + i * g
        # mem = torch.cat((c_next, m_new), 1)
        h_next = o * torch.tanh(c_next)
        # print("f", f.size())

        return h_next, c_next















class ModeCell(nn.Module):
    def __init__(self, in_channel, num_hidden, dim, width, filter_size, M):
        super(ModeCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0


        self.conv_og = DynamicConvOrigin(in_channel + num_hidden, num_hidden * 2, in_channel + num_hidden, WH=1, M=M, G=1,
                                      r=2)
        self.conv_if = DynamicConvOrigin(in_channel + num_hidden, num_hidden * 2, in_channel + num_hidden, WH=1, M=M,
                                        G=1, r=2)

        self.layer_norm = nn.LayerNorm([num_hidden * 4, width, width])

        self.layer_norm_og = nn.LayerNorm([num_hidden * 2, width, width])
        self.layer_norm_if = nn.LayerNorm([num_hidden * 2, width, width])
        # )

        self.conv_main = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 4, kernel_size=3, stride=1,
                      padding=1),
            nn.LayerNorm([num_hidden * 4, width, width])
        )

        # self.conv = InvertedResidual(in_channel + num_hidden, num_hidden * 4, 1, 1)

        # self.layer_norm_c = DynamicLayerNorm([num_hidden * 1, num_hidden * 1, width, width])

    def forward(self, out, combined, slots):
        # print(x.size(), h.size())
        # combined = torch.cat([x, h], dim=1)  # concatenate along channel axis
        # print(self.attention2d(combined).size())

        combined_conv_og, feas_og = self.conv_og(out, combined)
        combined_conv_if, feas_if = self.conv_if(out, combined)
        # combined_conv_main = self.conv_main(combined)
        # combined_conv = combined_conv + combined_conv_main
        combined_conv_og = self.layer_norm_og(combined_conv_og)
        combined_conv_if = self.layer_norm_if(combined_conv_if)
        cc_i, cc_f = torch.split(combined_conv_if, self.num_hidden, dim=1)
        cc_o, cc_g = torch.split(combined_conv_og, self.num_hidden, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # m_concat = self.conv_m(m_t)
        # i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        # i_t_prime = torch.sigmoid(i_m)
        # f_t_prime = torch.sigmoid(f_m)
        # g_t_prime = torch.tanh(g_m)

        # m_new = f_t_prime * m_t + i_t_prime * g_t_prime
        m = f * slots + i * g

        # c_next = f * c + i * g
        # mem = torch.cat((c_next, m_new), 1)
        h_next = o * torch.tanh(m)
        # print("f", f.size())

        return h_next, m








class ActionConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, dim, width, filter_size, M):
        super(ActionConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.attention2d = attention2d(in_channel + num_hidden)


        self.conv_a = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([num_hidden * 2, width, width])
        )
        self.conv_og = DynamicConvOrigin(in_channel + num_hidden, num_hidden * 2, in_channel + num_hidden, WH=1, M=M, G=1,
                                      r=2)
        self.conv_if = DynamicConvOrigin(in_channel + num_hidden, num_hidden * 2, in_channel + num_hidden, WH=1, M=M,
                                        G=1, r=2)
        # self.conv = AugmentedConv(in_channels=in_channel + num_hidden, out_channels=4*num_hidden, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, stride=1, shape=16)
        # self.gru = STConvGRUCell(in_channel, in_channel, in_channel, width, 1, stride, layer_norm)
        # self.conv = nn.Sequential(
        #    nn.Conv2d(in_channel + num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, dilation=1),
        self.layer_norm = nn.LayerNorm([num_hidden * 4, width, width])

        self.layer_norm_og = nn.LayerNorm([num_hidden * 2, width, width])
        self.layer_norm_if = nn.LayerNorm([num_hidden * 2, width, width])
        # )

        self.conv_main = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 4, kernel_size=3, stride=1,
                      padding=1),
            nn.LayerNorm([num_hidden * 4, width, width])
        )

        # self.conv = InvertedResidual(in_channel + num_hidden, num_hidden * 4, 1, 1)

        # self.layer_norm_c = DynamicLayerNorm([num_hidden * 1, num_hidden * 1, width, width])

    def forward(self, out, combined, slots):
        # print(x.size(), h.size())
        # combined = torch.cat([x, h], dim=1)  # concatenate along channel axis
        # print(self.attention2d(combined).size())
        #a_concat = self.conv_a(a_t)
        #print(a_t.size(), a_concat.size(), out.size())

        combined_conv_og, feas_og = self.conv_og(out, combined)
        combined_conv_if, feas_if = self.conv_if(out, combined)
        # combined_conv_main = self.conv_main(combined)
        # combined_conv = combined_conv + combined_conv_main
        combined_conv_og = self.layer_norm_og(combined_conv_og)
        combined_conv_if = self.layer_norm_if(combined_conv_if)
        cc_i, cc_f = torch.split(combined_conv_if, self.num_hidden, dim=1)
        cc_o, cc_g = torch.split(combined_conv_og, self.num_hidden, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # m_concat = self.conv_m(m_t)
        # i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        # i_t_prime = torch.sigmoid(i_m)
        # f_t_prime = torch.sigmoid(f_m)
        # g_t_prime = torch.tanh(g_m)

        # m_new = f_t_prime * m_t + i_t_prime * g_t_prime
        m = f * slots + i * g

        # c_next = f * c + i * g
        # mem = torch.cat((c_next, m_new), 1)
        h_next = o * torch.tanh(m)
        # print("f", f.size())

        return h_next, m











