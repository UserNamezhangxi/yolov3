from collections import OrderedDict

import torch
import torch.nn as nn
from darknet53 import darknet53


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1))
    ]))

def make_last_layer(in_filters, filters_list, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, bias=True)
    )
    return m


class YoloNet(nn.Module):
    def __init__(self, pretrained=False):
        super(YoloNet, self).__init__()
        # ========================#
        # 生成darknet53 主干网络
        # 获取三个共享特征层,分别是
        # 13,13,1024
        # 26,26,512
        # 52,52,256
        # ========================#
        self.backbone = darknet53()
        out_filters = self.backbone.layers_out_filters

        # 初始化yolo的输出通道
        # 输出通道75 = 3*(5+20) = 3*((4+1) + 20分类(对于V0C2007数据集))

        # 将输出13,13,75
        self.last_layer13 = make_last_layer(1024, [512, 1024], 75)

        # 输出13,13,512
        self.last_layer13_conv = conv2d(512,256, 1)
        # 输出 26,26,256
        self.last_layer13_upsample = nn.Upsample(scale_factor=2 , mode='nearest')
        # 输出 26,26,75
        self.last_layer26 = make_last_layer(512 + 256,[256, 512], 75)

        # 输出26,26,256
        self.last_layer26_conv = conv2d(256,128, 1)
        # 输出52,52,128
        self.last_layer26_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 输出52,52,75
        self.last_layer52 = make_last_layer(256 + 128,[128, 256], 75)

    def forward(self, x):
        x52, x26, x13 = self.backbone(x)
        # self.last_layer13[:5] 其实就是对make_last_layer 的模型取前5个,方便后面用x13_branch 进行1x1卷积和上采样
        x13_branch = self.last_layer13[:5](x13)  # [13,13,512]
        x13_branch_out = self.last_layer13[5:](x13_branch) # [13,13,75]

        # 对13*13的conv五次卷积结果进行1x1卷积和上采样
        x13_branch_conv1x1 = self.last_layer13_conv(x13_branch) # [13,13,256]
        x13_branch_upsample = self.last_layer13_upsample(x13_branch_conv1x1) # [26,26,256]

        # 26,26,512 + 26,26,256 -> 26,26,768
        out26concat = torch.concat([x26, x13_branch_upsample], dim=1)
        x26_branch = self.last_layer26[:5](out26concat)  # [26,26,256]
        x26_branch_out = self.last_layer26[5:](x26_branch) # [26,26,75}

        x26_branch_conv1x1 = self.last_layer26_conv(x26_branch) # [26,26,128]
        x26_branch_upsample = self.last_layer26_upsample(x26_branch_conv1x1) # [52,52,128]
        out52concat = torch.concat([x52, x26_branch_upsample], dim=1) # [52,52,384]

        # 52, 52, 384 -> 52, 52, 75
        x52_branch_out = self.last_layer52(out52concat) # [52,52,75]

        return x13_branch_out, x26_branch_out, x52_branch_out


if __name__ == "__main__":
    net = YoloNet()
    inputs = torch.rand((1,3,416, 416))
    x13_branch_out, x26_branch_out, x52_branch_out = net.forward(inputs)
    print(x13_branch_out.shape)
    print(x26_branch_out.shape)
    print(x52_branch_out.shape)