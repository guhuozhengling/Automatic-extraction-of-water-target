import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(conv_block,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
    def forward(self,x):
        x=self.conv(x)
        return x


class upsamping(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsamping,self).__init__()
        self.up=nn.Sequential(
            # nn.interpolate(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
    def forward(self,x):
        x=self.up(x)
        return x


class UNet(nn.Module):

    def __init__(self,in_channels=3,n_class=2):
        super(UNet,self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)  # 2×2
        self.conv1=conv_block(in_channels,64)
        self.conv2=conv_block(64,128)
        self.conv3=conv_block(128,256)
        self.conv4=conv_block(256,512)
        self.conv5=conv_block(512,1024)
        self.upsamping5=upsamping(1024,512)
        self.upconv5=conv_block(1024,512)
        self.upsamping4=upsamping(512,256)
        self.upconv4=conv_block(512,256)
        self.upsamping3=upsamping(256,128)
        self.upconv3=conv_block(256,128)
        self.upsamping2=upsamping(128,64)
        self.upconv2=conv_block(128,64)
        self.upconv1=nn.Conv2d(64,n_class,kernel_size=1,stride=1,padding=0)   #和上面conv比没有 bias=True

    def forward(self,x):
        # contracting path

        x1=self.conv1(x)  # [4, 64, 160, 160]

        x2=self.maxpool(x1)
        x2=self.conv2(x2)  # [4, 128, 80, 80]

        x3=self.maxpool(x2)
        x3=self.conv3(x3)  # [4, 256, 40, 40]

        x4=self.maxpool(x3)
        x4=self.conv4(x4)  # [4, 512, 20, 20]

        x5=self.maxpool(x4)
        x5=self.conv5(x5)  # [4, 1024, 10, 10]

        # expanding path

        d5=self.upsamping5(x5)
        d5=torch.cat((x4,d5),dim=1)
        d5=self.upconv5(d5)  # [4, 512, 20, 20]

        d4=self.upsamping4(d5)
        d4=torch.cat((x3,d4),dim=1)
        d4=self.upconv4(d4)  # [4, 256, 40, 40]

        d3=self.upsamping3(d4)
        d3=torch.cat((x2,d3),dim=1)
        d3=self.upconv3(d3)  # [4, 128, 80, 80]

        d2=self.upsamping2(d3)
        d2=torch.cat((x1,d2),dim=1)
        d2=self.upconv2(d2)  # [4, 64, 160, 160]

        d1=self.upconv1(d2)   # [4, 2, 160, 160]
        return d1