'''
__author__ = 'yunbo'

import torch
import torch.nn as nn

class SKConv(nn.Module):
    def __init__(self, in_planes, features, WH, M, G, r, stride=1, L=32):
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
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M + 1
        self.features = features
        self.convs = nn.ModuleList([])
        self.M = (M + 1) * 1
        self.features = features
        self.convs = nn.ModuleList([])
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_planes,
                      features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
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
                # nn.BatchNorm2d(features),
                # nn.ReLU(inplace=False))
            )

        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(nn.Linear(d, features).cuda())
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, id=0):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s).cuda()
        for i, fc in enumerate(self.fcs):
            #print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            #print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = torch.sigmoid(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = SKConv(in_channel, num_hidden*7,  WH = 1, M = 2, G = 1, r = 2)

        self.conv_h = SKConv(num_hidden, num_hidden*4,  WH = 1, M = 2, G = 1, r = 2)

        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = SKConv(num_hidden * 2, num_hidden, WH=1,M=2,G=1,r=2)
        #    nn.Sequential(
        #    nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
        #    nn.LayerNorm([num_hidden, width, width])
        #)
        self.conv_last = SKConv(num_hidden * 2, num_hidden,  WH = 1, M = 2, G = 1, r = 2)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

'''
__author__ = 'yunbo'

import torch
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init
from core.layers.relation_graph import*

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

class DynamicConv(nn.Module):
    def __init__(self, in_planes, features, WH, M, G, r, stride=1, L=32):
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
        super(DynamicConv, self).__init__()
        d = max(int(features / r), L)
        self.M = (M + 1)*1
        self.features = features
        self.convs = nn.ModuleList([])
        '''
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_planes // 4,
                      features // 4,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=G),
            # nn.Conv2d(features,
            #          features,
            #          kernel_size=1,
            #          stride=1,
            #          padding=0,
            #          groups=G),
        )
        )
        '''

        for i in range(self.M):
            '''
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_planes // 4,
                              features // 4,
                              kernel_size=(3 + i * 2, 1),
                              stride=stride,
                              padding=(1 + i, 0),
                              groups=G),
                    nn.Conv2d(features // 4,
                              features // 4,
                              kernel_size=(1, 3 + i * 2),
                              stride=stride,
                              padding=(0, 1 + i),
                              groups=G),

                )
                # nn.BatchNorm2d(features),
                # nn.ReLU(inplace=False))
            )
            '''
            self.convs.append(
               nn.Sequential(
                    nn.Conv2d(in_planes // 2, features, 3 + 2*1, 1, 1 + 1),

                ) 
            )
            

        # self.gap = nn.AvgPool2d(int(WH/stride))
        #self.fc = nn.Linear(in_planes, d)
        #self.fc = nn.GRUCell(features, d)
        self.output_conv = nn.Conv2d(in_planes,
                              features,
                              kernel_size= 5,
                              stride=1,
                              padding=2,
                              groups=G)
        self.fc = ContextBlock(in_planes, in_planes)
        self.att = nn.Linear(in_planes, features)
        self.fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(nn.Linear(d, features))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        channel_num = x.size(1) // 2
        #input_x = self.fc
        input_x= self.fc(x)
        fea_z = input_x.squeeze(3).squeeze(2)
        output_x = self.output_conv(input_x)
        att = torch.sigmoid(self.att(fea_z)).unsqueeze(-1).unsqueeze(-1)
        result = att * output_x
        return result

        '''
        for i, conv in enumerate(self.convs):
            fea = conv(input_x[:, channel_num * i : channel_num * (i + 1)]).unsqueeze_(dim=1)
            #fea = conv(x[:, channel_num * i : channel_num * (i + 1)])
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        #fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        #fea_s = x.mean(-1).mean(-1)
        #fea_s = fea_U

        #fea_z = self.fc(x)
        #fea_v = (feas * fea_z)
        #return fea_v

        
        #fea_z = self.fc(x).squeeze(3).squeeze(2)
        #print(fea_z.size())
        for i, fc in enumerate(self.fcs):
            #print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            #print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = torch.sigmoid(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        
        return fea_v
        '''
        



class ContextBlock3D(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(ContextBlock3D, self).__init__()
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
            self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=3)
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
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 3, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.planes, kernel_size=1))
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
        batch, channel, length, height, width = x.size()
        #print(x.size())
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, length, channel,  height * width)
            # [N, 1, T, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, T, H * W]
            context_mask = context_mask.view(batch, 1, length, height * width)
            # [N, 1, T, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, T, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, T, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            #context = context.view(batch, length, channel, 1, 1)
            context = context.permute(0, 3, 2, 1, 4)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        #print("ssssssssss", context.size())
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            #out = out * channel_mul_term
            #print("context3d", channel_mul_term.size()) 
            return  channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        #print("context3d", channel_mul_term.size())

        return out


class DynamicConv3D(nn.Module):
    def __init__(self, in_planes, features, WH, M, G, r, stride=1, L=32):
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
        super(DynamicConv3D, self).__init__()
        d = max(int(features / r), L)
        self.M = (M + 1)*1
        self.features = features
        self.convs = nn.ModuleList([])
        
        
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_planes,
                      features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
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
                    nn.Conv2d(in_planes, in_planes, 3 + 2*i, 1, 1 + i, groups=in_planes, bias=False),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(in_planes, features, 1, 1, 0, bias=False),

                ) 
            )
            
            

            ''' 
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_planes,
                              features,
                              kernel_size=(3 + i*2, 1),
                              stride=stride,
                              padding=(1+i*1, 0),
                              groups=G),
                    nn.Conv2d(features,
                              features,
                              kernel_size=(1, 3 + i * 2),
                              stride=stride,
                              padding=(0, 1 + i*1),
                              groups=G),

                )
            
            )
            '''
            
            
        '''
        for i in range(self.M):
            self.convs.append(
               DynamicConv(in_planes, features, WH, M, G, r)
            )
        '''
        

        self.pointnet = nn.Sequential(
            nn.Conv3d(features,
                      features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=features),
            nn.LayerNorm([features, 4, 16, 16])
            # nn.Conv2d(features,
            #          features,
            #          kernel_size=1,
            #          stride=1,
            #          padding=0,
            #          groups=G),
        )
        # self.gap = nn.AvgPool2d(int(WH/stride))
        #self.fc = nn.Linear(inplanes, d)
        #self.fc = nn.GRUCell(features, d)
        self.fc = ContextBlock3D(in_planes, features)
        #self.MMOE = SKLinear(in_planes, features, self.M)
        self.fcs = nn.ModuleList([])

        #self.transformer = AttModel(hp, 10000, 10000)
        for i in range(self.M):
            #self.fcs.append(nn.Linear(d, features))
            self.fcs.append(nn.Conv3d(d,
                      features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=G)
            )
            #self.fcs.append(SKLinear(d, features, 3))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #fea_z = self.fc(x).squeeze(-1).squeeze(-1)
        channel_num = x.size(1) // 4
        for i, conv in enumerate(self.convs):
            fea = conv(x[:, :, i]).unsqueeze_(dim=2)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=2)
        #fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        #fea_s = x.mean(-1).mean(-1)
        #fea_s = fea_U

        #fea_z = self.fc(x).squeeze(-1).squeeze(-1)

        #attention_vectors = self.MMOE(fea_s)
        #print(fea_z.size())
        #fea_z = self.fc(x)
        attention_vectors = self.fc(x)
        #print("sss", fea_z.size())
        #fea_z = y.squeeze(-1).squeeze(-1)

        '''

        for i, fc in enumerate(self.fcs):
            #print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            #print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        '''

        attention_vectors = torch.sigmoid(attention_vectors)
        #print(attention_vectors.size(), feas.size())
        #attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        #print(feas.size(), attention_vectors.size())
        #print(feas.size(), attention_vectors.size())
        fea_v = (feas * attention_vectors)
        #output = self.pointnet(fea_v)
        return fea_v


class GCConv(nn.Module):
    def __init__(self, in_planes, features, kernel_size=3, stride=1, padding=1):
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
        super(GCConv, self).__init__()
        #d = max(int(features / r), L)
        #self.M = (M + 1)*1
        #self.features = features
        self.convs = nn.Conv2d(in_planes, features, kernel_size, stride, padding)
        

        # self.gap = nn.AvgPool2d(int(WH/stride))
        #self.fc = nn.Linear(inplanes, d)
        #self.fc = nn.GRUCell(features, d)
        self.fc = ContextBlock(in_planes, features)
        #self.MMOE = SKLinear(in_planes, features, self.M)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        channel_num = x.size(1) // 4
        feas = self.convs(x)
        #fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        #fea_s = x.mean(-1).mean(-1)
        #fea_s = fea_U

        #fea_z = self.fc(x).squeeze(-1).squeeze(-1)

        #attention_vectors = self.MMOE(fea_s)
        #print(fea_z.size())
        fea_z = self.fc(x)
        fea_v = fea_z * feas

        
        return fea_v




class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        
        '''
        self.conv_x =  DynamicConv(in_channel, num_hidden * 7, WH=1,M=1,G=1,r=2)

        self.conv_h =  DynamicConv(num_hidden, num_hidden * 4, WH=1,M=1,G=1,r=2)

        self.conv_m =  DynamicConv(num_hidden, num_hidden * 3, WH=1,M=1,G=1,r=2)

        self.conv_o =  DynamicConv(num_hidden*2, num_hidden * 1, WH=1,M=1,G=1,r=2)

        self.conv_last =  DynamicConv(num_hidden*2, num_hidden * 1, WH=1,M=1,G=1,r=2)
        '''


        
        
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            #nn.LayerNorm([num_hidden * 7, width, width])
        )
    
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            #nn.LayerNorm([num_hidden * 4, width, width])
        )
        #self.conv_m = DynamicConv(num_hidden, num_hidden * 3, WH=1,M=3,G=1,r=2)
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            #nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            #nn.LayerNorm([num_hidden, width, width])
        )
        
        
        
        
        '''
        self.i_transform = DynamicConv(num_hidden, num_hidden * 1, WH=1,M=3,G=1,r=2) 
        self.f_transform = DynamicConv(num_hidden, num_hidden * 1, WH=1,M=3,G=1,r=2)
        self.g_transform = DynamicConv(num_hidden, num_hidden * 1, WH=1,M=3,G=1,r=2)

        self.i_m_transform = DynamicConv(num_hidden, num_hidden * 1, WH=1,M=3,G=1,r=2) 
        self.f_m_transform = DynamicConv(num_hidden, num_hidden * 1, WH=1,M=3,G=1,r=2)
        self.g_m_transform = DynamicConv(num_hidden, num_hidden * 1, WH=1,M=3,G=1,r=2)  
        self.o_transform = DynamicConv(num_hidden, num_hidden * 1, WH=1,M=3,G=1,r=2)
        '''
        
        


        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
        #nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
          

        '''  
        self.conv_h = TalkConv2d(
            num_hidden, num_hidden*4, 32, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,
            message_type="ws", directed=False, agg="sum",
            sparsity=0.5, p=0.0, talk_mode="dense", seed=None
        
        )
        self.conv_x = TalkConv2d(
            in_channel, num_hidden*7, 32, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,
            message_type="ws", directed=False, agg="sum",
            sparsity=0.5, p=0.0, talk_mode="dense", seed=None
        
        )

        self.conv_m = TalkConv2d(
            num_hidden, num_hidden*3, 32, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,
            message_type="ws", directed=False, agg="sum",
            sparsity=0.5, p=0.0, talk_mode="dense", seed=None
        
        )

        self.conv_o = TalkConv2d(
            num_hidden*2, num_hidden*1, 32, kernel_size=filter_size,
            stride=stride, padding=self.padding, bias=False,
            message_type="ws", directed=False, agg="sum",
            sparsity=0.5, p=0.0, talk_mode="dense", seed=None
        
        )
        '''
        



    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        #i_t = self.i_transform(i_t)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        #f_t = self.f_transform(f_t)
        g_t = torch.tanh(g_x + g_h)
        #g_t = self.g_transform(g_t)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        #i_t_prime = self.i_m_transform()
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new









