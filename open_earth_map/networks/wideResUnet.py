import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from functools import partial


nonlinearity = partial(F.relu, inplace=True)


class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=16, padding=16
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class WideResNetEncoder(nn.Module):
    def __init__(self, backbone):
        super(WideResNetEncoder, self).__init__()
        self.initial = nn.Sequential(
            backbone.conv1,  # 64 -> 64
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # /4 -> 128x128
        )
        self.layer1 = backbone.layer1  # 输出通道: 256, 尺寸: /4
        self.layer2 = backbone.layer2  # 输出通道: 512, 尺寸: /8
        self.layer3 = backbone.layer3  # 输出通道: 1024, 尺寸: /16
        self.layer4 = backbone.layer4  # 输出通道: 2048, 尺寸: /32

    def forward(self, x):
        x0 = self.initial(x)
        # print(f"x0 shape: {x0.shape}")  # [batch, 64, 128, 128]
        x1 = self.layer1(x0)
        # print(f"x1 shape: {x1.shape}")  # [batch, 256, 128, 128]
        x2 = self.layer2(x1)
        # print(f"x2 shape: {x2.shape}")  # [batch, 512, 64, 64]
        x3 = self.layer3(x2)
        # print(f"x3 shape: {x3.shape}")  # [batch, 1024, 32, 32]
        x4 = self.layer4(x3)
        # print(f"x4 shape: {x4.shape}")  # [batch, 2048, 16, 16]
        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, up_mode='upconv'):
        super(DecoderBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            raise ValueError("up_mode must be either 'upconv' or 'upsample'")

        self.conv = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            skip = F.interpolate(
                skip, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        up_mode='upconv',
        backbone_name='wide_resnet50_2',
        pretrained=True,
    ):
        super(ResUNet, self).__init__()
        self.encoder_out_channels = []

        # 加载预训练的WideResNet
        if backbone_name == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(
                weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            )
            self.encoder_out_channels = [64, 256, 512, 1024, 2048]  # x0, x1, x2, x3, x4
        elif backbone_name == 'wide_resnet101_2':
            backbone = models.wide_resnet101_2(
                weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
            )
            self.encoder_out_channels = [
                128,
                512,
                1024,
                2048,
                4096,
            ]  # 类似wide_resnet50_2
        else:
            raise ValueError(
                "Unsupported backbone. Choose from 'wide_resnet50_2', 'wide_resnet101_2'."
            )
        encoder_out_channels = self.encoder_out_channels

        self.encoder = WideResNetEncoder(backbone)

        # 定义解码器块，注意通道数的对应
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=encoder_out_channels[-1],  # 2048
                    skip_channels=encoder_out_channels[-2],  # 1024
                    out_channels=encoder_out_channels[-2],  # 1024
                    up_mode=up_mode,
                ),  # 2048 -> 1024
                DecoderBlock(
                    in_channels=encoder_out_channels[-2],  # 1024
                    skip_channels=encoder_out_channels[-3],  # 512
                    out_channels=encoder_out_channels[-3],  # 512
                    up_mode=up_mode,
                ),  # 1024 -> 512
                DecoderBlock(
                    in_channels=encoder_out_channels[-3],  # 512
                    skip_channels=encoder_out_channels[-4],  # 256
                    out_channels=encoder_out_channels[-4],  # 256
                    up_mode=up_mode,
                ),  # 512 -> 256
                DecoderBlock(
                    in_channels=encoder_out_channels[-4],  # 256
                    skip_channels=encoder_out_channels[-5],  # 64
                    out_channels=encoder_out_channels[-5],  # 64
                    up_mode=up_mode,
                ),  # 256 -> 64
            ]
        )

        # 增加一个额外的上采样步骤，将特征图从256x256上采样到512x512
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(
                encoder_out_channels[-5],
                encoder_out_channels[-5],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(encoder_out_channels[-5], n_classes, kernel_size=1)

    def forward(self, x):
        enc_features = self.encoder(x)
        x = enc_features[-1]  # x4: 2048通道, 尺寸: 16x16

        for i, decoder_block in enumerate(self.decoder):
            skip = enc_features[-2 - i]  # 从x3, x2, x1, x0
            x = decoder_block(x, skip)

        # 额外的上采样步骤
        x = self.final_upsample(x)  # 从64通道, 256x256 -> 64通道, 512x512
        x = self.final_conv(x)  # 从64通道 -> n_classes通道, 512x512

        return x


class ResUNet_withDilated(ResUNet):
    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        up_mode='upconv',
        backbone_name='wide_resnet50_2',
        pretrained=True,
    ):
        super().__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            up_mode=up_mode,
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        self.dblock = Dblock(self.encoder_out_channels[-1])

    def forward(self, x):
        enc_features = self.encoder(x)
        x = enc_features[-1]  # x4: 2048通道, 尺寸: 16x16

        x = self.dblock(x)

        for i, decoder_block in enumerate(self.decoder):
            skip = enc_features[-2 - i]  # 从x3, x2, x1, x0
            x = decoder_block(x, skip)

        # 额外的上采样步骤
        x = self.final_upsample(x)  # 从64通道, 256x256 -> 64通道, 512x512
        x = self.final_conv(x)  # 从64通道 -> n_classes通道, 512x512

        return x


class ResUNet_withMoreDilated(ResUNet):
    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        up_mode='upconv',
        backbone_name='wide_resnet50_2',
        pretrained=True,
    ):
        super().__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            up_mode=up_mode,
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        self.dblock = Dblock(self.encoder_out_channels[-1])

    def forward(self, x):
        enc_features = self.encoder(x)
        x = enc_features[-1]  # x4: 2048通道, 尺寸: 16x16

        x = self.dblock(x)

        for i, decoder_block in enumerate(self.decoder):
            skip = enc_features[-2 - i]  # 从x3, x2, x1, x0
            x = decoder_block(x, skip)

        # 额外的上采样步骤
        x = self.final_upsample(x)  # 从64通道, 256x256 -> 64通道, 512x512
        x = self.final_conv(x)  # 从64通道 -> n_classes通道, 512x512

        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512)
    model = ResUNet_withMoreDilated(
        in_channels=3,
        n_classes=9,
        up_mode='upconv',
        backbone_name='wide_resnet50_2',
        pretrained=True,
    )
    output = model(x)
    print(output.shape)  # 预期输出: torch.Size([1, 1, 512, 512])
