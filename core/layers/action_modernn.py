import torch
from torch import nn, einsum
from einops import rearrange
from core.layers.ConvLSTMCell import ActionConvLSTMCell


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         )


    def forward(self, x):
        #x = self.depthwise(x)
        #x = self.bn(x)
        #x = self.pointwise(x)
        x = self.conv(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ConvLSTMAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        #inner_dim = (dim + dim_head) *  heads * 1
        inner_dim = (dim + dim_head)
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(1*dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(2*dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(2*dim, inner_dim, kernel_size, v_stride, pad)

        self.convlstm = ActionConvLSTMCell(dim_head, dim, dim, img_size, 5, heads-1)
        #self.MoE = DynamicConvOrigin(dim*heads, dim, dim, WH=1,M=heads - 1,G=1,r=2)
        self.dim_head = dim_head
        self.conv = nn.Conv2d(dim_head, dim_head,
                               kernel_size=1, stride=1, padding=0, bias=False)


        self.slot_size = dim + dim_head
        self.num_slots = heads
        self.epsilon = 1e-8
        self.num_iterations = 5
        self.layernorm = nn.LayerNorm([dim ,img_size, img_size])
        self.conv_a = nn.Sequential(
            nn.Conv2d(dim, inner_dim, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([inner_dim, img_size, img_size])
        )


        self.slots_mu = nn.Parameter(torch.Tensor(1, 1, img_size, img_size))
        self.slots_log_sigma = nn.Parameter(torch.Tensor(1, 1, img_size, img_size))

        nn.init.xavier_uniform_(self.slots_mu, gain=nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.slots_log_sigma, gain=nn.init.calculate_gain("linear")),


        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, img_size, img_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_parameter(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, img_size, img_size)), gain=nn.init.calculate_gain("linear")),
        )
        


    def forward(self, x, a, prev=None, h=None, index=1):
        a_t = self.conv_a(a)
        head = self.heads
        b = x.size(0)
        n = x.size(1)
        #b, n, _, h = *x.shape, self.heads
        #x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        combined = torch.cat([x, h], dim=1)
        combined = combined * a_t

        k = self.to_k(combined)
        # print("k", k.size())
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=head)



        v = self.to_v(combined)
        # print("v", v.size())
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=head)

        if prev is None:
            # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
            slots_init = torch.randn((b, 1*self.dim_head, x.size(-2), x.size(-1)))
            slots_init = slots_init.type_as(x)
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init
            #slots
            #print("init", x.size(), slots.size())
        else:
            slots = prev

        #combined = torch.cat([x, slots], dim=1)

        slots_prev = slots

        q = self.to_q(slots)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=head)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = rearrange(out, 'b f (l w) -> b f l w', l=self.img_size, w=self.img_size)
        out = rearrange(out, 'b (h d) l w -> b h d l w', l=self.img_size, w=self.img_size, h=head)

        h, slots, feas = self.convlstm(out, combined, slots_prev)



        return h, slots






