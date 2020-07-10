import torch
import torch.nn as nn


class DenseLayer(nn.Sequential):
    def __init__(self, in_size, growth_rate):
        super().__init__()
        dilated = []
        for d in [1, 3, 5]:
          layers = [nn.Conv2d(in_size, growth_rate, kernel_size=3, stride=1, dilation=d, padding=d, bias=True), nn.Dropout2d(0.2)]
          dilated.append(nn.Sequential(*layers))
        self.dilated_layers = nn.ModuleList(dilated)
        out_size = 3 * growth_rate
        self.out_layer = nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.dilated_layers[0](x)
        x2 = self.dilated_layers[1](x)
        x3 = self.dilated_layers[2](x)
        x = torch.cat([x1, x2, x3], 1)
        x = self.out_layer(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_size, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [DenseLayer(in_size + 3*i*growth_rate, growth_rate) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1) # 1 = channel axis
        return x


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
 
        self.dense = DenseBlock(in_channels, 16, num_layers=5)
        self.down1 = UNetDown(243, 128, normalize=False)
        self.down2 = UNetDown(128, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
 
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
 
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, out_channels, 4, padding=1),
            nn.Tanh(),
        )
 
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        dense = self.dense(x)
        d1 = self.down1(dense)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
 
        return self.final(u7)


if __name__ == '__main__':
    import numpy as np
    input = torch.Tensor(np.ones((1, 3, 1024, 1024)))
    model = GeneratorUNet()
    print(model(input).size())
    print(sum([len(p) for p in model.parameters()]))
