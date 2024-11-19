# import
import torch.nn as nn
import torch.nn.functional as F

# parameter
state_size = (3, 3) # env.state_size
action_size = 9 # env.action_size
CONV_UNITS = 64
RESIDUAL_NUM = 16

# ResNet
class Net(nn.Module):
    def __init__(self, state_size, action_size, conv_units):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn = nn.BatchNorm2d(conv_units)
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1)

        self.fc_size = conv_units * (state_size[-1]+2) * (state_size[-2]+2) * state_size[0]
        self.fc = nn.Linear(self.fc_size, action_size)

    def _residual_block(self, x):
        sc = x
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x += sc
        x = F.relu(x)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pooling(x)

        # residual block
        for i in range(RESIDUAL_NUM):
            x = self._residual_block(x)

        # pooling
        x = self.pool(x)
        # 전연결 신경망
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        # policy: action probability (vector) - softmax
        policy = F.softmax(x, self.action_size)

        # value: win probability (scalar) - tanh
        value = nn.Tanh(x)

        return policy, value