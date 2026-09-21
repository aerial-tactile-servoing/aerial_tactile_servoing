
from torch.nn.functional import relu
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

def weighted_mse_loss(pred, target, bg_weight=0.1, fg_weight=1.0):
    # Create mask: foreground is where target > 0
    mask = (target > 0).float()
    weights = mask * fg_weight + (1 - mask) * bg_weight
    loss = weights * (pred - target) ** 2
    return loss.mean()

def focal_mse(pred, target, gamma=4):
    error = (pred - target).abs()
    weights = (1 - error) ** gamma
    return (weights * error**2).mean()

def generate_color_coordinates(height, width):
    # Row indices for Red: increase from bottom to top
    red = np.linspace(1.0, 0.0, height).reshape(height, 1).repeat(width, axis=1)
    
    # Column indices for Green: increase from right to left
    green = np.linspace(1.0, 0.0, width).reshape(1, width).repeat(height, axis=0)
    
    # Column indices for Blue: increase from left to right
    blue = np.linspace(0.0, 1.0, width).reshape(1, width).repeat(height, axis=0)
    
    coords = np.stack([red, green, blue], axis=0)
    return torch.tensor(coords, dtype=torch.float32)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, use_checkpoint=False):
        super(UNet, self).__init__()
        self.use_checkpoint = use_checkpoint

        # Encoder (contracting path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        # Bottleneck (reduced to 1024 channels)
        self.bottleneck = self.conv_block(1024, 1024)

        # Decoder (expansive path)
        self.upconv5 = self.upconv_block(1024, 512)
        self.dec5 = self.conv_block(1536, 512)

        self.upconv4 = self.upconv_block(512, 256)
        self.dec4 = self.conv_block(768, 256)

        self.upconv3 = self.upconv_block(256, 128)
        self.dec3 = self.conv_block(384, 128)

        self.upconv2 = self.upconv_block(128, 64)
        self.dec2 = self.conv_block(192, 64)
        
        self.upconv1 = self.upconv_block(64, 64)
        self.dec1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward_impl(self, x):
        # Encoder
        enc1 = self.enc1(x)                    # 64 x 640 x 480
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # 128 x 320 x 240
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # 256 x 160 x 120
        enc4 = self.enc4(F.max_pool2d(enc3, 2))  # 512 x 80 x 60
        enc5 = self.enc5(F.max_pool2d(enc4, 2))  # 1024 x 40 x 30

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc5, 2))  # 1024 x 20 x 15

        # Decoder
        up5 = self.upconv5(bottleneck)  # 512 x 40 x 30
        dec5 = self.dec5(torch.cat([up5, enc5], dim=1))  # 1024 → 512

        up4 = self.upconv4(dec5)  # 256 x 80 x 60
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))  # 512 → 256

        up3 = self.upconv3(dec4)  # 128 x 160 x 120
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))  # 256 → 128

        up2 = self.upconv2(dec3)  # 64 x 320 x 240
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))  # 128 → 64

        # Final output
        up1 = self.upconv1(dec2)
        out = self.final_conv(up1)
        return out

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)

class TactileDataset(Dataset):
    def __init__(self, root_dir, entries, transform=None, use_polar_coords=True):
        self.root_dir = root_dir
        self.entries = entries
        self.transform = transform
        self.use_polar_coords = use_polar_coords

        # Assume shape from first sample
        example_img = cv2.imread(os.path.join(root_dir, entries[0], "gsmini.png"))
        h, w = example_img.shape[:2]
        self.coords = generate_color_coordinates(h, w)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rel_path = self.entries[idx]
        sample_path = os.path.join(self.root_dir, rel_path)

        rgb = cv2.imread(os.path.join(sample_path, "gsmini.png"))
        gray = cv2.imread(os.path.join(sample_path, "tuned_gray.png"), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            rgb = self.transform(rgb)
        else:
            rgb = T.ToTensor()(rgb)

        gray = torch.tensor(gray, dtype=torch.float32).unsqueeze(0) / 255.0
       
        # Add polar coordinates as channels
        if self.use_polar_coords:
            rgb = torch.cat([rgb, self.coords], dim=0)  

        return rgb, gray