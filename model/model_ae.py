import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# Neural Architecture 2 : Auto Encoder
'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 256, 256]           4,704
       BatchNorm2d-2         [-1, 32, 256, 256]              64
         MaxPool2d-3         [-1, 32, 128, 128]               0
            Conv2d-4         [-1, 16, 128, 128]          25,088
       BatchNorm2d-5         [-1, 16, 128, 128]              32
         MaxPool2d-6           [-1, 16, 64, 64]               0
            Conv2d-7            [-1, 8, 64, 64]           3,200
       BatchNorm2d-8            [-1, 8, 64, 64]              16
         MaxPool2d-9            [-1, 8, 32, 32]               0
           Conv2d-10            [-1, 4, 32, 32]             800
      BatchNorm2d-11            [-1, 4, 32, 32]               8
        MaxPool2d-12            [-1, 4, 16, 16]               0
           Linear-13                   [-1, 32]          32,800
          Dropout-14                   [-1, 32]               0
           Linear-15                 [-1, 1024]          33,792
          Dropout-16                 [-1, 1024]               0
  ConvTranspose2d-17            [-1, 8, 32, 32]             800
      BatchNorm2d-18            [-1, 8, 32, 32]              16
  ConvTranspose2d-19           [-1, 16, 64, 64]           3,200
      BatchNorm2d-20           [-1, 16, 64, 64]              32
  ConvTranspose2d-21         [-1, 32, 128, 128]          25,088
      BatchNorm2d-22         [-1, 32, 128, 128]              64
  ConvTranspose2d-23          [-1, 3, 256, 256]           4,704
================================================================
Total params: 134,408
Trainable params: 134,408
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 51.77
Params size (MB): 0.51
Estimated Total Size (MB): 53.04
----------------------------------------------------------------
'''

# class Autoencoder(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.rep_dim = 32
#         self.pool = nn.MaxPool2d(2, 2)

#         # Encoder
#         self.encoder_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), padding=3, bias=False)
#         self.encoder_bn1 = nn.BatchNorm2d(32)
#         self.encoder_maxpool1 = nn.MaxPool2d(2, 2)
        
#         self.encoder_conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(7, 7), padding=3, bias=False)
#         self.encoder_bn2 = nn.BatchNorm2d(16)
#         self.encoder_maxpool2 = nn.MaxPool2d(2, 2)
        
#         self.encoder_conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5), padding=2, bias=False)
#         self.encoder_bn3 = nn.BatchNorm2d(8)
#         self.encoder_maxpool3 = nn.MaxPool2d(2, 2)
        
#         self.encoder_conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(5, 5), padding=2, bias=False)
#         self.encoder_bn4 = nn.BatchNorm2d(4)
#         self.encoder_maxpool4 = nn.MaxPool2d(2, 2)

#         # self.encoder_fc1 = nn.Linear(4 * 16 * 16, self.rep_dim)
#         self.encoder_fc1 = nn.Linear(8 * 32 * 32, self.rep_dim*4)

#         # Decoder
#         # self.decoder_fc1 = nn.Linear(self.rep_dim, 4 * 16 * 16)
#         self.decoder_fc1 = nn.Linear(self.rep_dim*4, 8 * 32 * 32)
        
#         self.decoder_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=(5, 5), padding=2, bias=False)
#         self.decoder_bn1 = nn.BatchNorm2d(8)
        
#         self.decoder_conv2 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2, bias=False)
#         self.decoder_bn2 = nn.BatchNorm2d(16)
        
#         self.decoder_conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3, bias=False)
#         self.decoder_bn3 = nn.BatchNorm2d(32)
        
#         self.decoder_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(7, 7), padding=3, bias=False)
#         self.dropout = nn.Dropout(0.5)
#     def forward(self, x):
#         # Encoder
#         x = self.encoder_maxpool1(F.leaky_relu(self.encoder_bn1(self.encoder_conv1(x))))
#         x = self.encoder_maxpool2(F.leaky_relu(self.encoder_bn2(self.encoder_conv2(x))))
#         x = self.encoder_maxpool3(F.leaky_relu(self.encoder_bn3(self.encoder_conv3(x))))
#         # x = self.encoder_maxpool4(F.leaky_relu(self.encoder_bn4(self.encoder_conv4(x))))
#         # x = x.reshape(-1, 4 * 16 * 16)  # Flatten
#         x = x.reshape(-1, 8 * 32 * 32)  # Flatten
#         x = self.encoder_fc1(x)
#         # Decoder
#         x = F.leaky_relu(self.decoder_fc1(x))
#         # x = x.reshape(-1, 4, 16, 16)  # Reshape
#         x = x.reshape(-1, 8, 32, 32)  # Reshape
#         x = F.interpolate(F.leaky_relu(x), scale_factor=2)
#         # x = F.interpolate(F.leaky_relu(self.decoder_bn1(self.decoder_conv1(x))), scale_factor=2)
#         x = F.interpolate(F.leaky_relu(self.decoder_bn2(self.decoder_conv2(x))), scale_factor=2)
#         x = F.interpolate(F.leaky_relu(self.decoder_bn3(self.decoder_conv3(x))), scale_factor=2)
#         x = torch.sigmoid(self.decoder_conv4(x))

#         return x

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), padding=3, bias=False)
        self.encoder_bn1 = nn.BatchNorm2d(32)
        self.encoder_maxpool1 = nn.MaxPool2d(2, 2)
        
        self.encoder_conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(7, 7), padding=3, bias=False)
        self.encoder_bn2 = nn.BatchNorm2d(16)
        self.encoder_maxpool2 = nn.MaxPool2d(2, 2)
        
        self.encoder_conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5), padding=2, bias=False)
        self.encoder_bn3 = nn.BatchNorm2d(8)
        self.encoder_maxpool3 = nn.MaxPool2d(2, 2)
        
        self.encoder_conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(5, 5), padding=2, bias=False)
        self.encoder_bn4 = nn.BatchNorm2d(4)
        self.encoder_maxpool4 = nn.MaxPool2d(2, 2)

        # self.encoder_fc1 = nn.Linear(4 * 16 * 16, self.rep_dim)
        self.encoder_fc1 = nn.Linear(8 * 32 * 32, self.rep_dim*4) # dimension : (128, 1)

        # Decoder
        # self.decoder_fc1 = nn.Linear(self.rep_dim, 4 * 16 * 16)
        self.decoder_fc1 = nn.Linear(self.rep_dim*4, 8 * 32 * 32)
        
        self.decoder_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=(5, 5), padding=2, bias=False)
        self.decoder_bn1 = nn.BatchNorm2d(8)
        
        self.decoder_conv2 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2, bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(16)
        
        self.decoder_conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3, bias=False)
        self.decoder_bn3 = nn.BatchNorm2d(32)
        
        self.decoder_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(7, 7), padding=3, bias=False)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # Encoder
        x = self.encoder_maxpool1(F.relu((self.encoder_conv1(x))))
        x = self.encoder_maxpool2(F.relu((self.encoder_conv2(x))))
        x = self.encoder_maxpool3(F.relu((self.encoder_conv3(x))))
        # x = self.encoder_maxpool4(F.leaky_relu(self.encoder_bn4(self.encoder_conv4(x))))
        # x = x.reshape(-1, 4 * 16 * 16)  # Flatten
        x = x.reshape(-1, 8 * 32 * 32)  # Flatten
        x = self.encoder_fc1(x)
        latent = x
        # Decoder
        x = F.relu(self.decoder_fc1(x))
        # x = x.reshape(-1, 4, 16, 16)  # Reshape
        x = x.reshape(-1, 8, 32, 32)  # Reshape
        x = F.interpolate(F.relu(x), scale_factor=2)
        # x = F.interpolate(F.leaky_relu(self.decoder_bn1(self.decoder_conv1(x))), scale_factor=2)
        x = F.interpolate(F.relu((self.decoder_conv2(x))), scale_factor=2)
        x = F.interpolate(F.relu((self.decoder_conv3(x))), scale_factor=2)
        x = torch.sigmoid(self.decoder_conv4(x))

        return x, latent


# 모델 인스턴스 생성
# model = SVDD_ae()
