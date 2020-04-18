import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


class conv2DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DGroupNorm(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cg_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(n_filters)))

    def forward(self, inputs):
        outputs = self.cg_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DGroupNormRelu(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cgr_unit = nn.Sequential(
            conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs




# class residualBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, n_filters, stride=1, downsample=None):
#         super(residualBlock, self).__init__()

#         self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3, stride, 1, bias=False)
#         self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
#         self.downsample = downsample
#         self.stride = stride
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x

#         out = self.convbnrelu1(x)
#         out = self.convbn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)
#         return out


class pyramidPooling(nn.Module):
    def __init__(
        self, in_channels, pool_sizes, model_name="pspnet", fusion_mode="cat", is_batchnorm=True
    ):
        super(pyramidPooling, self).__init__()

        bias = not is_batchnorm

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                conv2DBatchNormRelu(
                    in_channels,
                    int(in_channels / len(pool_sizes)),
                    1,
                    1,
                    0,
                    bias=bias,
                    is_batchnorm=is_batchnorm,
                )
            )

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]

        if self.model_name == "pspnet": 
            k_sizes = []
            strides = []
            for pool_size in self.pool_sizes:
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
        else:  
            k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
            strides = [(5, 10), (10, 20), (16, 32), (33, 65)]

        if self.fusion_mode == "cat":  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                pp_sum = pp_sum + out

            return pp_sum


class bottleNeckPSP(nn.Module):
    def __init__(
        self, in_channels, mid_channels, out_channels, stride, dilation=1, is_batchnorm=True
    ):
        super(bottleNeckPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(
            in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                is_batchnorm=is_batchnorm,
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=stride,
                padding=1,
                bias=bias,
                dilation=1,
                is_batchnorm=is_batchnorm,
            )
        self.cb3 = conv2DBatchNorm(
            mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        self.cb4 = conv2DBatchNorm(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1, is_batchnorm=True):
        super(bottleNeckIdentifyPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(
            in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=1,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                is_batchnorm=is_batchnorm,
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=1,
                padding=1,
                bias=bias,
                dilation=1,
                is_batchnorm=is_batchnorm,
            )
        self.cb3 = conv2DBatchNorm(
            mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )

    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x + residual, inplace=True)


class residualBlockPSP(nn.Module):
    def __init__(
        self,
        n_blocks,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation=1,
        include_range="all",
        is_batchnorm=True,
    ):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        # residualBlockPSP = convBlockPSP + identityBlockPSPs
        layers = []
        if include_range in ["all", "conv"]:
            layers.append(
                bottleNeckPSP(
                    in_channels,
                    mid_channels,
                    out_channels,
                    stride,
                    dilation,
                    is_batchnorm=is_batchnorm,
                )
            )
        if include_range in ["all", "identity"]:
            for i in range(n_blocks - 1):
                layers.append(
                    bottleNeckIdentifyPSP(
                        out_channels, mid_channels, stride, dilation, is_batchnorm=is_batchnorm
                    )
                )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


