import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural Architecture 1 : SVDD
'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 256, 256]           2,400
       BatchNorm2d-2         [-1, 32, 256, 256]              64
         MaxPool2d-3         [-1, 32, 128, 128]               0
            Conv2d-4         [-1, 16, 128, 128]          12,800
       BatchNorm2d-5         [-1, 16, 128, 128]              32
         MaxPool2d-6           [-1, 16, 64, 64]               0
            Conv2d-7            [-1, 8, 64, 64]           3,200
       BatchNorm2d-8            [-1, 8, 64, 64]              16
         MaxPool2d-9            [-1, 8, 32, 32]               0
           Conv2d-10            [-1, 4, 32, 32]             800
      BatchNorm2d-11            [-1, 4, 32, 32]               8
        MaxPool2d-12            [-1, 4, 16, 16]               0
           Linear-13                   [-1, 32]          32,800
================================================================
Total params: 52,120
Trainable params: 52,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 41.13
Params size (MB): 0.20
Estimated Total Size (MB): 42.08
----------------------------------------------------------------
'''



class SVDD(nn.Module):

    def __init__(self):
        super().__init__()
        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(7, 7), padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5), padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(5, 5), padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(4)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4 * 16 * 16, self.rep_dim)
        # self.fc1 = nn.Linear(8 * 32 * 32, self.rep_dim*4)

    def forward(self, x):

        x = self.maxpool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.maxpool3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.maxpool4(F.leaky_relu(self.bn4(self.conv4(x))))
        x = x.reshape(-1, 4 * 16 * 16)  # Flatten
        # x = x.reshape(-1, 8 * 32 * 32)  # Flatten
        x = self.fc1(x)

        return x
