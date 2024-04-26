import torch.nn.init
from torch import nn


def weight_init(net, init_gain=0.02):
    def init_func(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data,0, init_gain)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, std=init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


