import torch
import torch.nn as nn
from core.layers.GHU_layer import GradientHighwayUnit


#__all__ = ['DynamicLayerNorm']


class DynamicLayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-10):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(DynamicLayerNorm, self).__init__()
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        #print("n", normal_shape[0])
        self.gamma_weight = nn.Conv2d(normal_shape[1], normal_shape[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.beta_weight = nn.Conv2d(normal_shape[1], normal_shape[0], kernel_size=1, stride=1, padding=0, bias=True)

        '''
        if gamma:
            self.gamma = nn.ones(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.zero(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()
        '''

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x, gamma, beta):
        self.gamma = self.gamma_weight(gamma)
        self.beta = self.beta_weight(beta)
        #x = x.view((x.size(0), -1))
        mean = x.mean(dim=[1,2,3], keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        #print(y.size(), self.gamma.size(), self.beta.size(), x.size(), a.size())
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta

        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )



class AdaLayerNorm(nn.Module):

    def __init__(self, in_channel, num_hidden, filter_size, stride, layer_norm=False):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(AdaLayerNorm, self).__init__()
        #self.normal_shape = torch.Size(normal_shape)
        self.epsilon = 1e-10
        #print("n", normal_shape[0])

        self.ghu = GradientHighwayUnit(in_channel, num_hidden, filter_size, stride, layer_norm=False)
        #self.gamma_weight = nn.Conv2d(normal_shape[1], normal_shape[0], kernel_size=1, stride=1, padding=0, bias=True)
        #self.beta_weight = nn.Conv2d(normal_shape[1], normal_shape[0], kernel_size=1, stride=1, padding=0, bias=True)

        '''
        if gamma:
            self.gamma = nn.ones(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.zero(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()
        '''


    def forward(self, x, z_t):
        #self.gamma = self.gamma_weight(gamma)
        #self.beta = self.beta_weight(beta)
        #x = x.view((x.size(0), -1))
        mean = x.mean(dim=[1,2,3], keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        
        z = self.ghu(y, z_t)
        out = z + x


        return out, z



vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

#vgg.load_state_dict(torch.load("/workspace/yaozhiyu/109_hub/predrnn-pytorch//vgg_normalised.pth"))
#vgg = nn.Sequential(*list(vgg.children())[5:31])




class AdaNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-10):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(AdaNorm, self).__init__()
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        self.vgg = vgg
        #print("n", normal_shape[0])
        #self.gamma_weight = nn.Conv2d(normal_shape[1], normal_shape[0], kernel_size=1, stride=1, padding=0, bias=True)
        #self.beta_weight = nn.Conv2d(normal_shape[1], normal_shape[0], kernel_size=1, stride=1, padding=0, bias=True)

        '''
        if gamma:
            self.gamma = nn.ones(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.zero(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()
        '''

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x, style_content):
        #print(style.size())
        style = self.vgg(style_content)
        #print(style.size())
        mean_style = style.mean(dim=[1, 2, 3], keepdim=True)
        var_style = ((style_content - mean_style) ** 2).mean(dim=-1, keepdim=True)
        std_style = (var_style + self.epsilon).sqrt()
        self.gamma = std_style
        self.beta = mean_style
        #x = x.view((x.size(0), -1))
        mean = x.mean(dim=[1,2,3], keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        #print(y.size(), self.gamma.size(), self.beta.size(), x.size(), a.size())
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta

        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )



    


