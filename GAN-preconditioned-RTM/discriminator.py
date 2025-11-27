import torch
import torch.nn as nn
class PatchGAN(nn.Module):
    def __init__(self):
        super(PatchGAN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.Conv2d(512, 1, kernel_size=4),
            # nn.AdaptiveAvgPool2d(1),  # 添加全局平均池化层
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x = self.model(x)
        return x.view(x.size(0), -1)  # 调整输出形状以匹配目标标签的尺寸