# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_parameter(model, parameter):
    result = []
    if hasattr(model, parameter):
        result.append(getattr(model, parameter))
    for l in model.children():
        result += get_parameter(l, parameter)
    return result


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _ResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNet18(nn.Module):
    def __init__(self, n_channels=3, block=_ResBlock, num_blocks=[2, 2, 2, 2], n_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, torch.zeros(1).to(out.device)


class InitialPathway(nn.Module):
    def __init__(self, n_channels=3, block=_ResBlock, num_blocks=[2, 2]):
        super(InitialPathway, self).__init__()

        self.in_planes = 64
        self.conv1 = conv3x3(n_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class SResNet18(nn.Module):
    """
    Split-ResNet 18 that processes inputs from two views, and adds them at an intermediate layer
    """

    def __init__(self, n_channels=3, block=_ResBlock, num_blocks=[2, 2, 2, 2], n_classes=10):
        super(SResNet18, self).__init__()

        self.pathway_a = InitialPathway(n_channels, block, num_blocks[:2])
        self.pathway_b = InitialPathway(n_channels, block, num_blocks[:2])

        self.in_planes = self.pathway_a.in_planes

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x_a, x_b):
        out_a = self.pathway_a(x_a)
        out_b = self.pathway_b(x_b)
        out = out_a + out_b
        out = self.layer3_out = self.layer3(out)
        out = self.layer4_out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, Variable(torch.zeros(1).cuda())


class SResNet18MultiHead(nn.Module):
    """
    Split-ResNet 18 with Multiple Heads that outputs a target based on input_a, input_b, or both of them
    This is achieved in the train() of main_independent.py by ensuring the target corresponds to either input_a or
    input_b during the deficit.
    Processes inputs from two views, and adds them at an intermediate layer
    """

    def __init__(self, n_channels=3, block=_ResBlock, num_blocks=[2, 2, 2, 2], n_classes=10):
        super(SResNet18MultiHead, self).__init__()

        self.pathway_a = InitialPathway(n_channels, block, num_blocks[:2])
        self.pathway_b = InitialPathway(n_channels, block, num_blocks[:2])
        self.in_planes = self.pathway_a.in_planes
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_both = nn.Linear(512 * block.expansion, n_classes)
        self.linear_a = nn.Linear(512 * block.expansion, n_classes)
        self.linear_b = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x_a, x_b):
        out_a = self.pathway_a(x_a)
        out_b = self.pathway_b(x_b)
        out = out_a + out_b
        out = self.layer3_out = self.layer3(out)
        out = self.layer4_out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out_both = self.linear_both(out)
        out_a_final = self.linear_a(out)
        out_b_final = self.linear_b(out)

        return out_both, out_a_final, out_b_final
