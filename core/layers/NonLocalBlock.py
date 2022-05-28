import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self, height, width, q_dim, k_dim, v_dim, o_dim, h_dim = 64):
        super(AttnBlock, self).__init__()
        #self.batch = batch_size
        self.height = height
        self.width = width
        self.dim = h_dim
        #self.ch_attn = ch_attn
        self.conv_q = nn.Conv2d(q_dim, self.dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_k = nn.Conv2d(k_dim, self.dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_v = nn.Conv2d(v_dim, self.dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = nn.Conv2d(self.dim, o_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, q_t, k_t, v_t):
        q_t = self.conv_q(q_t)
        k_t = self.conv_k(k_t)
        v_t = self.conv_v(v_t)

        # e.g. (b,dim,h,w) => (b,dim,h*w)
        q_t = q_t.view(-1, self.dim, self.height * self.width)
        k_t = k_t.view(-1, self.dim, self.height * self.width)
        v_t = v_t.view(-1, self.dim, self.height * self.width)

        # e.g. (b,dim,hw) * (b,hw,dim) => (b,dim,dim)
        h_t = torch.matmul(q_t, k_t.permute(0, 2, 1).contiguous())
        h_t = F.softmax(h_t, dim=-1)
        # e.g. (b,dim,dim) * (b,dim,h*w) => (b,dim,hw) => (b,dim,h,w)
        h_t = torch.matmul(h_t, v_t).view(-1, self.dim, self.height, self.width)

        h_t = self.conv_out(h_t)
        return h_t
