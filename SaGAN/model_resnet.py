import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

import functools
from torch.autograd import Variable

#线性层初始化
#Xavier初始化的基本思想是保持输入和输出的方差一致，这样就避免了所有输出值都趋向于0。
# 这是通用的方法，适用于任何**函数。
def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()

#卷积层初始化
def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

'''谱归一化，实际上Spectral Normaliztion需要做的就是将每层的参数矩阵除以自身的最大奇异值，
所以本质上就是一个逐层SVD的过程，但如果真的去做SVD的话就太耗时间了，所以采用幂迭代的方法求解。
'''
'''
python中@staticmethod方法，类似于C++中的static，方便将外部函数集成到类体中，
主要是可以在不实例化类的情况下直接访问该方法，如果你去掉staticmethod,
在方法中加self也可以通过实例化访问方法也是可以集成。
'''
class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)

#谱过程的标准化
def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)#例如这里就是用了@staticmethod功能中的直接调用

    return module

#谱过程初始化
def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)

#激活函数Leaky ReLU
def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)

#自注意力机制，“8”表示下采样倍数
class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=2 ** 0.5):
        super().__init__()

        self.query = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out

#条件规范化
class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Embedding(n_class, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        return out

#卷积模块，每使用一次ConvBlock特征图就会上采样一次，也就是特征图大小放大一倍
'''
torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
size：据不同的输入制定输出大小；
scale_factor：指定输出为输入的多少倍数；
mode：可使用的上采样算法，有nearest，linear，bilinear，bicubic 和 trilinear。默认使用nearest；
align_corners ：如果为 True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。
'''
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, n_class=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False):
        super().__init__()

        gain = 2 ** 0.5

        self.conv1 = spectral_init(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=False if bn else True),
                                   gain=gain)
        self.conv2 = spectral_init(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=False if bn else True),
                                   gain=gain)

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_skip = spectral_init(nn.Conv2d(in_channel, out_channel,
                                                     1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.norm1 = ConditionalNorm(in_channel, n_class)
            self.norm2 = ConditionalNorm(out_channel, n_class)

    def forward(self, input, class_id=None):
        out = input
        if self.bn:
            out = self.norm1(out, class_id)
        out = self.activation(out)
        if self.upsample:
            out = F.upsample(out, scale_factor=2)
        out = self.conv1(out)
        if self.bn:
            out = self.norm2(out, class_id)
        out = self.activation(out)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_skip(skip)

            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input
        return out + skip

#生成器
class Generator(nn.Module):
    def __init__(self, code_dim=500, n_class=10):#采样100维度的噪声点
        super().__init__()

        self.lin_code = spectral_init(nn.Linear(code_dim, 4 * 4 * 512))#输入的特征图大小4*4
        self.conv = nn.ModuleList([ConvBlock(512, 512, n_class=n_class),#特征图大小8*8
                                   ConvBlock(512, 512, n_class=n_class),#特征图大小16*16
                                   ConvBlock(512, 256, n_class=n_class),#特征图大小32*32
                                   SelfAttention(256),
                                   ConvBlock(256, 128, n_class=n_class),#特征图大小64*64
                                   ConvBlock(128, 64, n_class=n_class)])#特征图大小128*128

        self.bn = nn.BatchNorm2d(64)#这里的“64”需要与上一行的“32”一样
        self.colorize = spectral_init(nn.Conv2d(64, 3, (3, 3), padding=1))#同上“64”

    def forward(self, input, class_id):
        out = self.lin_code(input)
        out = out.view(-1, 512, 4, 4)

        for conv in self.conv:
            if isinstance(conv, ConvBlock):
                out = conv(out, class_id)

            else:
                out = conv(out)
        out = self.bn(out)

        out = F.relu(out)

        out = self.colorize(out)
        return F.tanh(out)

#判别器
class Discriminator(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()

        def conv(in_channel, out_channel, downsample=True):
            return ConvBlock(in_channel, out_channel,
                             bn=False,
                             upsample=False, downsample=downsample)

        gain = 2 ** 0.5
        #nn.Sequential中的“32”与生成器中的“64”一致
        self.pre_conv = nn.Sequential(spectral_init(nn.Conv2d(3, 64, 3,padding=1),gain=gain),
                                      nn.ReLU(),
                                      spectral_init(nn.Conv2d(64, 64, 3,padding=1),gain=gain),
                                      nn.AvgPool2d(2))
        self.pre_skip = spectral_init(nn.Conv2d(3, 64, 1))#同上的“64”
        #这里的nn.Sequential中的顺序与生成器中的nn.ModuleList顺序相反
        self.conv = nn.Sequential(conv(64, 128),
                                  conv(128, 256, downsample=False),
                                  SelfAttention(256),
                                  conv(256, 512),
                                  conv(512, 512),
                                  conv(512, 512))

        self.linear = spectral_init(nn.Linear(512, 1))

        self.embed = nn.Embedding(n_class, 512)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

    def forward(self, input, class_id):
        out = self.pre_conv(input)
        out = out + self.pre_skip(F.avg_pool2d(input, 2))
        out = self.conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        embed = self.embed(class_id)
        prod = (out * embed).sum(1)
        return out_linear + prod
