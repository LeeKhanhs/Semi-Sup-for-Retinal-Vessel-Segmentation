import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.decoder_u2pl import Aux_Module, UNetDecoder

class UNetEncoder(nn.Module):
    def __init__(self):
        super(UNetEncoder, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        return x1, x2, x3, x4  # Returning skip connections for UNetDecoder


class UNet_U2PL(nn.Module):
    def __init__(self):
        super(UNet_U2PL, self).__init__()
        
        self._num_classes = 1  # Binary classification (foreground and background)
        
        # Encoder with UNetEncoder
        self.encoder = UNetEncoder()
        
        # Decoder
        self.decoder = UNetDecoder(in_planes=512, num_classes=self._num_classes, use_aux=False)

        # Auxiliary loss module if required
        self._use_auxloss = False  # Auxiliary loss is disabled by default
        if self._use_auxloss:
            self.loss_weight = 0.4  # Example weight for auxiliary loss
            self.auxor = Aux_Module(in_planes=256, num_classes=self._num_classes)

    def forward(self, x):
        # Run through encoder and get skip connections
        x1, x2, x3, x4 = self.encoder(x)
        
        # Ensure all encoder feature maps are resized to match the decoder's output size
        x2 = F.interpolate(x2, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        # Pass skip connections to decoder
        outs = self.decoder(x4, encoder_features=[x1, x2, x3, x4])  # Pass encoder features into decoder
        
        # Add 'pred' key in the output dictionary (which is expected by the training code)
        outs["pred"] = outs["main"]  # 'main' is the final classification output from the decoder
        
        # If auxiliary loss is enabled
        if self._use_auxloss:
            pred_aux = self.auxor(x2)  # Use `x2` as the intermediate feature map for auxiliary loss
            outs["aux"] = pred_aux
        
        # Include the 'rep' (representation) key
        outs["rep"] = outs["rep"]  # 'rep' is the representation output from the decoder
        
        return outs

# Example to test
if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))  # Example input tensor (batch size 2, 3 channels, 512x512 image)
    model = UNet_U2PL()  # Initialize the model
    outs = model(x)  # Forward pass
    print(outs["pred"].shape)  # Ensure it outputs the correct prediction under 'pred' key
    print(outs["rep"].shape)  # Check the shape of the 'rep' output
