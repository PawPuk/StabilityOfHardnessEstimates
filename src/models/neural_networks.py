import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet basic block (2 convs + shortcut)."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetLowRes(nn.Module):
    """
    ResNet for low-resolution images (CIFAR/TinyImageNet).
    Depth must be 6n+2 (e.g., 20, 32, 44, 56, 110).
    """
    def __init__(self, depth, num_classes=10, num_filters=16):
        super(ResNetLowRes, self).__init__()
        assert (depth - 2) % 6 == 0, "Depth must be 6n+2"
        n = (depth - 2) // 6  # number of blocks per stage

        # Initial convolution (no downsampling)
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Three stages, each with n BasicBlocks
        self.stage1 = self._make_stage(num_filters, num_filters, n, stride=1)
        self.stage2 = self._make_stage(num_filters, num_filters * 2, n, stride=2)
        self.stage3 = self._make_stage(num_filters * 2, num_filters * 4, n, stride=2)

        # Final pooling and FC
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_filters * 4, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_planes, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(in_planes, out_planes, s))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
