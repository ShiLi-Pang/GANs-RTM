import torch
import sys
sys.path.append('../input')
sys.path.insert(0, 'Utilities')
try:
    from scipy.sparse.linalg.isolve.utils import make_system
except:
    from scipy.sparse.linalg._isolve.utils import make_system
from torch.nn import functional as F
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, 5, 1, 2),
            nn.Tanh(),
            nn.Conv2d(C_out, C_out, 5, 1, 2),
        )
        self.tanh = nn.Tanh()
        # 添加一个1x1卷积用于匹配维度，仅当输入和输出通道数不相等时使用
        self.match_channels = nn.Conv2d(C_in, C_out, 1, 1, 0) if C_in != C_out else None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        # 如果输入和输出通道数不相等，使用1x1卷积调整残差的通道数
        if self.match_channels is not None:
            residual = self.match_channels(x)
        out += residual  # 添加残差
        return self.tanh(out)  # 应用激活函数


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 5, 2, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest-exact")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 4次下采样
        self.C1 = Conv(2, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)
        # 3次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)
        self.Th = nn.Tanh()
        self.pred = torch.nn.Conv2d(64, 2, 5, 1, 2)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))
        # 上采样部分
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))
        return self.Th(self.pred(O4))
