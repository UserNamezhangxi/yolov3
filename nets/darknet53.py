import math

from torch import nn
from collections import OrderedDict


class BasicBlock(nn.Module):
    def __init__(self, in_channel, channels):
        super(BasicBlock, self).__init__()
        print("input={},output={}".format(in_channel, channels[0]))
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=channels[0],
                               kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return residual + out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.in_channel = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu1 = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

    def _make_layer(self, channels, block_size):
        layers = []
        layers.append(("ds_conv",
                       nn.Conv2d(in_channels=self.in_channel, out_channels=channels[1], kernel_size=3, stride=2,
                                 padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(channels[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        self.in_channel = channels[1]
        for i in range(0, block_size):
            layers.append(("residual_{}".format(i), BasicBlock(in_channel=self.in_channel, channels=channels)))

        return nn.Sequential(OrderedDict(layers))

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model

if __name__ == "__main__":
    # model = BasicBlock(32, 64)
    # inputs = torch.randn(*(1, 32, 128, 128))  # nchw
    # print(model)
    # out = model.forward(inputs)
    # print(out.shape)
    model = darknet53()
    print(model)
