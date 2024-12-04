import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.seg_output = nn.Conv2d(64, 1, kernel_size=1)
        self.fp_output = nn.Conv2d(64, 1, kernel_size=1)  # Output for feature perturbation

    def forward(self, inputs, return_features=False, fp=False):
        """ Encoder """
        s1, p1 = self.e1(inputs)  # Output size: [batch, 64, H, W]
        s2, p2 = self.e2(p1)      # Output size: [batch, 128, H/2, W/2]
        s3, p3 = self.e3(p2)      # Output size: [batch, 256, H/4, W/4]
        s4, p4 = self.e4(p3)      # Output size: [batch, 512, H/8, W/8]

        """ Bottleneck """
        b = self.b(p4)            # Output size: [batch, 1024, H/16, W/16]

        """ Decoder """
        d1 = self.d1(b, s4)       # Output size: [batch, 512, H/8, W/8]
        d2 = self.d2(d1, s3)      # Output size: [batch, 256, H/4, W/4]
        d3 = self.d3(d2, s2)      # Output size: [batch, 128, H/2, W/2]
        d4 = self.d4(d3, s1)      # Output size: [batch, 64, H, W]

        segmentation_output = self.seg_output(d4)  # Output size: [batch, 1, H, W]
        perturbation_output = self.fp_output(d4)  # Output size: [batch, 1, H, W] for feature perturbation

        if return_features:
            # Concatenate flattened features from encoder stages and bottleneck
            features = torch.cat([s1.flatten(1), s2.flatten(1), s3.flatten(1), s4.flatten(1), b.flatten(1)], dim=1)
            if fp:
                return segmentation_output, perturbation_output, features
            return segmentation_output, features
        
        # If fp is set to True, return both segmentation and perturbation
        if fp:
            return segmentation_output, perturbation_output
        return segmentation_output

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    model = UNet()
    y, fp = model(x, fp=True)  # Forward pass with feature perturbation
    print("Segmentation output shape:", y.shape)
    print("Perturbation output shape:", fp.shape)
