# -- coding : uft-8 --
# Author : Wang Han 
# Southeast University
import copy
import time
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.act = {'softmax': nn.Softmax(dim=1), 'sigmoid': nn.Sigmoid()}

    def data_parallel(self, device_ids):
        self.model = nn.DataParallel(self.model, device_ids=device_ids)

        return

    def load(self, path):
        weight = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(weight)

        return

    def save(self, path):
        torch.save(self.model.module.state_dict(), path)

        return

    def forward(self, x, y):
        return self.model(x, y)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool3d(1)
    return model


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool3d(1)
    return model


def se_resnet50(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool3d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool3d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool3d(1)
    return model


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResBlock1d, self).__init__()
        self.out_channels = out_channels
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.mish = nn.Mish(self.out_channels)
        self.fres = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        res = x.view(x.size(0), -1)
        # residual = self.fres(res)

        out = self.conv1d(x)
        out = self.bn1(out)
        out = self.mish(out)
        out = out.view(out.size(0), -1)

        # out += residual
        # out = self.mish(out)

        return out


class Norm1d():
    def __init__(self, *args):
        self.arg = args

    def __call__(self, x):
        mean = torch.mean(x, 0, True)
        std = torch.std(x, 0, True)
        out = (x - mean) / std
        return out


def ProbW(x, sigma=1 / 3, Lambda=1 / 2):
    B, L = x.size()
    # out = copy.deepcopy(x)
    if L == 12:
        t1ce_p = sigma * x[:, :2] + sigma * x[:, 2:4] + sigma * x[:, 4:6]
        t2_p = sigma * x[:, 6:8] + sigma * x[:, 8:10] + sigma * x[:, 10:]
        out = Lambda * t1ce_p + (1 - Lambda) * t2_p
    else:
        raise KeyError(f"Expected length is 12 but got {L} !")
    return out


class Prob():
    def __init__(self, sigma=1 / 3, *args):
        self.arg = args
        self.sigma = sigma

    def __call__(self, x):
        _, L = x.size()
        # out = copy.deepcopy(x)
        if L == 12:
            t1ce_p = self.sigma * x[:, :2] + self.sigma * x[:, 2:4] + self.sigma * x[:, 4:6]
            t2_p = self.sigma * x[:, 6:8] + self.sigma * x[:, 8:10] + self.sigma * x[:, 10:]
            out = 0.5 * t1ce_p + 0.5 * t2_p
        else:
            raise KeyError(f"Expected length is 12 but got {L} !")
        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, in_channel=3, num_classes=2, feature_channel=32, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.fea = feature_channel
        self.fea_out = int(feature_channel / 4)
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.conv1 = nn.Conv3d(
            self.in_channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(
            16, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.convy = nn.Conv1d(
            self.fea, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.bn2 = nn.BatchNorm1d(self.fea)
        self.bny = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(
            block, 128, blocks=n_size, stride=2, reduction=reduction)
        self.layer5 = self._make_layer(
            block, 256, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, 8)
        self.fcl = nn.Linear(8, self.num_classes)
        self.fcy = nn.Linear(self.fea_out, self.num_classes)
        self.res = ResBlock1d(self.fea, self.fea_out)
        self.act = nn.Softmax(dim=1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = torch.cat((x, y), dim=-1).unsqueeze(-1)
        x = self.bn2(x)
        x = self.res(x)

        x = self.fcy(x)
        x = self.act(x)

        return x


# class SEResNet(nn.Module):
#     def __init__(self, block, n_size, in_channel=3, num_classes=2, reduction=16):
#         super(SEResNet, self).__init__()
#         self.inplane = 16
#         self.num_classes = num_classes
#         self.in_channel = in_channel
#         self.conv1 = nn.Conv3d(
#             self.in_channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(self.inplane)
#         self.bny = nn.BatchNorm1d(8)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(
#             block, 16, blocks=n_size, stride=1, reduction=reduction)
#         self.layer2 = self._make_layer(
#             block, 32, blocks=n_size, stride=2, reduction=reduction)
#         self.layer3 = self._make_layer(
#             block, 64, blocks=n_size, stride=2, reduction=reduction)
#         self.layer4 = self._make_layer(
#             block, 128, blocks=n_size, stride=2, reduction=reduction)
#         self.avgpool = nn.AdaptiveAvgPool3d(1)
#         self.fc1 = nn.Linear(64, 16)
#         self.fc2 = nn.Linear(16, self.num_classes)
#         self.act = nn.Softmax(dim=1)
#         self.initialize()
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride, reduction):
#         strides = [stride] + [1] * (blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inplane, planes, stride, reduction))
#             self.inplane = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.act(x)
#
#         return x


class SEResNet(nn.Module):
    def __init__(self, block, n_size, in_channel=3, num_classes=2, reduction=16):
        super(SEResNet, self).__init__()
        self.inplane = 16
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.conv1 = nn.Conv3d(
            self.in_channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(
            block, 128, blocks=n_size, stride=2, reduction=reduction)
        self.layer5 = self._make_layer(
            block, 256, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 16)
        self.fc2 = nn.Linear(16, self.num_classes)
        self.act = nn.Softmax(dim=1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x)

        return x


class CifarMixSEResNet(nn.Module):
    def __init__(self, block, n_size, in_channel=3, num_classes=2, feature_channel1=32, feature_channel2=12,
                 reduction=16):
        super(CifarMixSEResNet, self).__init__()
        self.inplane = 16
        self.fea1 = feature_channel1 + 10
        self.fea2 = feature_channel2
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.conv1 = nn.Conv3d(
            self.in_channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.fea1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, 10)
        self.res = ResBlock1d(1, 3)
        self.fc1 = nn.Linear(self.fea2, self.num_classes)
        self.fc2 = nn.Linear(3 * self.fea1, 18)
        self.fc3 = nn.Linear(18 + self.fea2, 10)
        self.fc4 = nn.Linear(10, self.num_classes)
        self.act = nn.Softmax(dim=1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x, y1, y2):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x1 = torch.cat((x, y1), dim=-1).unsqueeze(-2)
        x1 = self.res(x1)
        x1 = self.fc2(x1)

        out = torch.cat((x1, y2), dim=-1)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.act(out)

        return out


# class CifarSEPreActResNet(CifarSEResNet):
#     def __init__(self, block, n_size, in_channel=3, num_classes=2, feature_channel=32, reduction=16):
#         super(CifarSEPreActResNet, self).__init__(
#             block, n_size, in_channel, num_classes, reduction)
#         self.fea_ = feature_channel + 128
#         self.fea_out_ = int(feature_channel / 4)
#         self.bn1 = nn.BatchNorm3d(self.inplane)
#         self.fc0 = nn.Linear(256, 128)
#         self.fc1 = nn.Linear(self.fea_, 8)
#         self.fc2 = nn.Linear(8, self.num_classes)
#         self.initialize()
#
#     def forward(self, x, y):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         # print(x.shape)
#
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc0(x)
#
#         x = torch.cat((x, y), dim=-1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.act(x)
#
#         return x


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, in_channel=3, num_classes=2, feature_channel=12, linear_channel=24, reduction=16):
        super(CifarSEPreActResNet, self).__init__(
            block, n_size, in_channel, num_classes, reduction)
        self.linear_c = linear_channel
        self.fea_c = feature_channel + self.linear_c
        self.fea_out_ = int(feature_channel / 4)
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.fc0 = nn.Linear(256, self.linear_c)
        self.fc1 = nn.Linear(self.fea_c, 8)
        self.fc2 = nn.Linear(8, self.num_classes)
        self.initialize()

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)

        x = torch.cat((x, y), dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x)

        return x


def se_resnet(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = SEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_res9net(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = SEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_resnet51(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 51, **kwargs)
    return model


def se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_mixresnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarMixSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_mixresnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarMixSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_mixresnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarMixSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_mixresnet51(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarMixSEResNet(CifarSEBasicBlock, 51, **kwargs)
    return model


if __name__ == '__main__':
    # start = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x = torch.randn((2, 3, 128, 128, 128)).to(device)
    # fea = torch.rand(size=[2, 12]).to(device)
    # model = se_res9net(in_channel=3, num_classes=2).to(device)
    #
    # total = sum([param.nelement() for param in model.parameters()])
    # print(total)
    #
    # y = model(x)
    # print(y)
    # print(time.time() - start)
    strides = [2] + [1] * (9 - 1)
    print(strides)
