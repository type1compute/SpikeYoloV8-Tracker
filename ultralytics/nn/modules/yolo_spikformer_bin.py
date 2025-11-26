#这是二进制推理的版本，ILIF神经元输出全二值结果。重写卷积层，在执行完纯加法后送入bn层。
# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from ultralytics.nn.modules.conv import Concat

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import warnings
# from visualizer import get_local


from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors
import math


# 此类用于二进制推理。
class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  #修改时只要修改这里就好

    def forward(self, x):
        """
        Binary inference membrane update.

        Args:
            x: Tensor of shape [T, B, C, H, W], where T is the external timestep count.

        Behavior:
            • Do NOT collapse the external T dimension.
            • Apply MultiSpike4 per timestep to get an integer spike count in [0, 4].
            • Use expand_tensor_cumulative to expand each timestep into D binary substeps.
              For D=4 this yields an internal time axis of length T * 4.
            • Downstream Conv2d_bn layers will then:
                - Flatten time and batch to [T*4*B, C, H, W]
                - Group into 4-substep chunks, sum over those 4 internal steps
                - Restore the external time dimension T when reshaping.

        Result:
            The visible T dimension stays equal to the original time_steps for the whole model
            (one frame per timestep), while each timestep is internally represented by 4 binary
            spike substeps.
        """
        # x: [T, B, C, H, W]
        # Apply integer quantization per timestep without summing over T.
        mem = x  # preserve per-timestep structure

        # MultiSpike4 produces integer spike counts in [0, 4] for each timestep
        spike_int = self.qtrick(mem)  # [T, B, C, H, W] integers 0..4

        # Expand each timestep into D binary substeps; for MultiSpike4, D = 4.
        # Resulting shape: [T * D, B, C, H, W], where D=4 by default.
        spike = expand_tensor_cumulative(spike_int)

        return spike



def expand_tensor_cumulative(tensor, max_value=4):

    T, B, C, H, W = tensor.shape
    # 创建一个 shape 为 [max_value, 1, 1, 1, 1, 1] 的比较向量
    steps = torch.arange(max_value, device=tensor.device).view(max_value, 1, 1, 1, 1, 1)

    # 扩展原始张量维度，便于比较 → [1, T, B, C, H, W]
    tensor_expanded = tensor.unsqueeze(0)

    # 比较：每个位置 v，生成 v 个 1，其余为 0
    binary = (steps < tensor_expanded).float()  # → shape [max_value, T, B, C, H, W]

    # 重新 reshape → [max_value * T, B, C, H, W]
    binary = binary.permute(1, 0, 2, 3, 4, 5).reshape(T * max_value, B, C, H, W)

    return binary


class MultiSpike4(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))  # 这里将两个4改为1，来对比t1时候的snn

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)


class MS_C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        print("MS_C2f_N:",n)
        print("MS_C2f_c:",self.c)
        self.cv1 = MS_StandardConv(c1, 2 * self.c, 1, 1)
        self.cv2 = MS_StandardConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = MS_StandardConv(c1, c_, k[0], 1)
        self.cv2 = MS_StandardConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = Conv2d_bn(c1, c2, k, s, autopad(k, p, d), g=g, bias=False)
        self.lif = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.conv(self.lif(x).flatten(0, 1)).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x

#==============================================================



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)


@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)


class Conv2d_bn(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bias=None,first_layer=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=None)
        self.bn = nn.BatchNorm2d(c2)
        self.first_layer = first_layer

        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    #这里第一版ilif前后不等价的原因是，整数变成脉冲序列后，每个D都加了bias
    #解决办法是bias设为None，或者bias除以4(如果需要将bn归一化到conv中，则需要手工除以4)
    def forward(self, x,first_layer=False):

        # print("input:",x.sum())
        x = self.conv(x)
        # print("conv:", x.sum())
        if first_layer == False:
            new_shape = (4, x.shape[0] // 4) + x.shape[1:]
            x = x.view(new_shape).sum(dim=0)
        x = self.bn(x)
        # print("bn:", x.sum())
        return x


class Conv2d(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bias=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=None)

    def forward(self, x):
        x = self.conv(x)
        new_shape = (4, x.shape[0] // 4) + x.shape[1:]
        x = x.view(new_shape).sum(dim=0)
        return x


class SpikeDFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)  # [0,1,2,...,15]
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))  # 这里不是脉冲驱动的，但是是整数乘法
        self.c1 = c1  # 本质上就是个加权和。输入是每个格子的概率(小数)，权重是每个格子的位置(整数)
        self.lif = mem_update()

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版


class SpikeDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(MS_StandardConv(x, c2, 3), MS_StandardConv(c2, c2, 3), MS_StandardConvWithoutBN(c2, 4 * self.reg_max, 1)) for x
            in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(MS_StandardConv(x, c3, 3), MS_StandardConv(c3, c3, 3), MS_StandardConvWithoutBN(c3, self.nc, 1)) for x in ch)
        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.concat1 = Concat(2)
        self.concat2 = Concat(2)
        self.concat3 = Concat(1)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].mean(0).shape  # BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = self.concat1((self.cv2[i](x[i]), self.cv3[i](x[i])))
            x[i] = x[i].mean(0)  # [2，144，32，684]  #这个地方有时候全是1.之后debug看看
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = self.concat2([xi.view(shape[0], self.no, -1) for xi in x])
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # box: [B,reg_max * 4,anchors]
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = self.concat3((dbox, cls.sigmoid()))
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        #     a[-1].conv.conv.bias.data[:] = 1.0  # box
        #     b[-1].conv.conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            bias=True,
            group=1
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        # hidden_channel = in_channel

        self.conv1 = Conv2d_bn(in_channel, in_channel, 1, 1, 0, bias=True, g=group)
        self.conv2 = Conv2d_bn(in_channel, in_channel, kernel_size, 1, 1, g=in_channel, bias=True)
        self.conv3 = Conv2d_bn(in_channel, out_channel, 1, 1, 0, g=group, bias=True)

        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.conv1(self.lif1(x).flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.conv2(self.lif2(x).flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.conv3(self.lif3(x).flatten(0, 1)).reshape(T, B, -1, H, W)

        return x

#这个类已经弃用了，但是不能直接删，因为load权重需要
class SepRepConv(nn.Module): #放在Sepconv最后一个1*1卷积，采用3*3分组+1*1降维的方式实现，能提0.5个点。之后可以试试改成1*1降维和3*3分组
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
#         conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, 1,0, groups=in_channel, bias=False),  #这里也是分组卷积
            # MultiStepLIFNode(detach_reset=True, v_reset=None, backend='torch'),
            nn.Conv2d(in_channel, out_channel, 1,  1,0, groups=group, bias=False),
        )

        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=True,
                 kernel_size=3,  # 7,3
                 padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = Conv2d_bn(dim, med_channels, k=1, s=1, bias=bias)
        self.dwconv2 = Conv2d_bn(
            med_channels, med_channels, k=kernel_size,  # 7*7
            p=padding, g=med_channels, bias=bias)  # depthwise conv
        self.pwconv3 = Conv2d_bn(med_channels, dim, k=1, s=1, bias=bias)

        self.dwconv4 = Conv2d_bn(dim, dim, 1, 1, 0, g=dim, bias=True)  # 这里也是分组卷积

        # self.bn1 = nn.BatchNorm2d(med_channels)
        # self.bn2 = nn.BatchNorm2d(med_channels)
        # self.bn3 = nn.BatchNorm2d(dim)
        # self.bn4 = nn.BatchNorm2d(dim)

        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()
        self.lif4 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        # print("x.shape:",x.shape)
        # self.lif_before1 = mem_update0()
        # self.lif_before2 = mem_update0()

        # x_before = x

        # x_before = self.lif_before1(x_before)
        x = self.lif1(x)   #  [1,1,64,160,160]这里两个tensor和还是完全相同的 x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高

        # x_before = self.pwconv1(x_before.flatten(0, 1)).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维
        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维


        # x_before = self.lif_before2(x_before)
        x = self.lif2(x)

        # x_before = self.dwconv2(x_before.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.dwconv2(x.flatten(0, 1)).reshape(T, B, -1, H, W)


        # print(x.sum())
        # print(x_before.sum())

        x = self.lif3(x)
        x = self.pwconv3(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.lif4(x)
        x = self.dwconv4(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        return x


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y


class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, full=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)  # 内部扩张2倍
        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio))  # 137以外的模型，在第一个block不做分
        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.add1 = Add()
        self.add2 = Add()

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.add1(self.Conv(x), x)  # sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)

        # x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
        #     #repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        # x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.add2(x_feat, x)

        return x


class MS_AllConvBlock(nn.Module):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)

        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)
        self.add1 = Add()
        self.add2 = Add()

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.add1(self.Conv(x), x)  # sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.add2(x_feat, x)

        return x



class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()

        self.encode_conv = Conv2d_bn(in_channels, embed_dims, k=kernel_size, s=stride,
                                              p=padding)
        # self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = mem_update()
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):  # 如果不是第一层
            # x_pool = self.pool(x)
            x = self.encode_lif(x)
            x = self.encode_conv(x.flatten(0, 1), first_layer=False)  # [1,1,3,640,640]

        else:
            x = self.encode_conv(x.flatten(0, 1),first_layer=True)


        H, W = x.shape[-2],x.shape[-1]
        # print("SHAPE:",x.shape)
        x = x.reshape(T, B, -1, H, W).contiguous()

        return x



class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        if len(x.shape)==4: #3333 在这里调整维度，把T放最前面
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        elif len(x.shape)==5:
            x = x.transpose(0,1)   #1,1,3,256,320 #CHW
        return x


class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x






class MS_StandardConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), g=g)
        self.lif = mem_update()

        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = MS_StandardConv(c1, c_, 1, 1)
        self.cv2 = MS_StandardConv(c_ * 4, c2, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.concat = Concat(2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m1(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m2(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m3(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(self.concat((x, y1, y2, y3)))


class MS_Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):  # 这里输入x是一个list
        for i in range(len(x)):
            if x[i].dim() == 5:
                x[i] = x[i].mean(0)
        return torch.cat(x, self.d)










class MS_C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = MS_StandardConv(c1, c_, 1, 1)
        self.cv2 = MS_StandardConv(c1, c_, 1, 1)
        self.cv3 = MS_StandardConv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 2)) # 11111 2 这里也要把最后一位改成2，确保维度正确

class MS_C3k(MS_C3):
    """MS_C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize MS_C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))



class MS_C3k2(MS_C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, MS_c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize MS_C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            MS_C3k (bool): Whether to use MS_C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            MS_C3k(self.c, self.c, 2, shortcut, g) if MS_c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )



class Ann_ConvBlock:
    pass

class Ann_DownSampling:
    pass
class Ann_StandardConv:
    pass
class Ann_SPPF:
    pass

class Ann_ConvBlock:
    pass
class Conv_1:
    pass
class BasicBlock_1:
    pass
class BasicBlock_2:
    pass
class Concat_res2:
    pass
class MS_FullConvBlock:
    pass
class Sample:
    pass

class MS_ConvBlock_resnet50:
    pass
class MS_Block:
    pass



class BasicBlock_2:
    pass
class MS_ConvBlock_res2net:
    pass
class PatchEmbedInit:
    pass
class PatchEmbeddingStage:
    pass
class TokenSpikingTransformer:
    pass

class SpikeConv:
    pass
class SepAllConv:
    pass


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        x_feat = x
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,W,H)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        T,B,C,W,H = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,W,H).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,W,H).contiguous()
        x = self.fc2_lif(x)
        return x


class SpikingTransformer(nn.Module): #用于深层
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = Spiking_Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x



class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps



class SpikeConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)

class SpikeConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = mem_update()

        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x
