import torch
import torch.nn as nn
import torch.nn.functional as F

class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=1):
        super(Aux_Module, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)
class UNetDecoder(nn.Module):
    def __init__(self, in_planes, num_classes=1, aux_planes=256, use_aux=False):
        super(UNetDecoder, self).__init__()

        # Define convolution layers for the decoder with skip connections
        self.upconv4 = nn.ConvTranspose2d(in_planes, 512, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Fix: Change 64 to 256 channels here
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Ensure this layer outputs 256 channels
            nn.ReLU(inplace=True)
        )

        # Final classification layer
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

        # Representation conv layer to get the representation
        self.representation_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Auxiliary module for auxiliary loss
        self.use_aux = use_aux
        if self.use_aux:
            self.aux_module = Aux_Module(aux_planes, num_classes)

    def forward(self, x, encoder_features):
        # Decoder with skip connections from encoder features
        x = self.upconv4(x)
        x = F.interpolate(x, size=encoder_features[3].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, encoder_features[3]], dim=1)  # Concatenate along the channel dimension
        x = self.conv4(x)

        x = self.upconv3(x)
        x = F.interpolate(x, size=encoder_features[2].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, encoder_features[2]], dim=1)
        x = self.conv3(x)

        x = self.upconv2(x)
        x = F.interpolate(x, size=encoder_features[1].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, encoder_features[1]], dim=1)
        x = self.conv2(x)

        x = self.upconv1(x)
        x = F.interpolate(x, size=encoder_features[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, encoder_features[0]], dim=1)

        # **Fixing the channel size issue here**:
        # After concatenation with encoder_features[0], the number of channels should be 128
        # We need to ensure that the output of `conv1` is 256 channels.
        x = self.conv1(x)  # Now this outputs 256 channels

        # Now, x should have 256 channels, and we can safely pass it through representation_conv
        rep_output = self.representation_conv(x)

        # Final output layer
        final_output = self.final_conv(x)

        outputs = {"main": final_output, "rep": rep_output}

        # Auxiliary output
        if self.use_aux:
            aux_output = self.aux_module(encoder_features[1])  # Use intermediate encoder feature for aux
            outputs["aux"] = aux_output

        return outputs


