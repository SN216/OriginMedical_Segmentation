import torch
import torch.nn as nn

# Re-using the same U-Net structure
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1): # Default to 1 output
        super(SimpleUNet, self).__init__()
        # ... (Copy the exact same U-Net code from landmark/model.py) ...
        # ... Only the __init__ defaults change ...
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.d1 = double_conv(in_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.d2 = double_conv(64, 128)
        self.d3 = double_conv(128, 256)
        self.d4 = double_conv(256, 512)
        self.b = double_conv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.u1 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u2 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u3 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u4 = double_conv(128, 64)
        
        # No Sigmoid here if using BCEWithLogitsLoss
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.d1(x)
        p1 = self.pool(c1)
        c2 = self.d2(p1)
        p2 = self.pool(c2)
        c3 = self.d3(p2)
        p3 = self.pool(c3)
        c4 = self.d4(p3)
        p4 = self.pool(c4)
        b = self.b(p4)
        x = self.up1(b); x = torch.cat([x, c4], dim=1); x = self.u1(x)
        x = self.up2(x); x = torch.cat([x, c3], dim=1); x = self.u2(x)
        x = self.up3(x); x = torch.cat([x, c2], dim=1); x = self.u3(x)
        x = self.up4(x); x = torch.cat([x, c1], dim=1); x = self.u4(x)
        return self.final(x)