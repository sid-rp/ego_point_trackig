import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PixelDisplacement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 8)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(8, 16)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(16, 32)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(32 + 16, 16)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(16 + 8, 8)

        # Final output: 2 channels (dx, dy)
        self.final = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        # Encode
        x1 = self.enc1(x)                    # (B, 8, H, W)
        x2 = self.enc2(self.pool1(x1))       # (B, 16, H/2, W/2)
        x3 = self.bottleneck(self.pool2(x2)) # (B, 32, H/4, W/4)

        # Decode
        x = self.up2(x3)                     # (B, 32, H/2, W/2)
        x = self.dec2(torch.cat([x, x2], dim=1)) # (B, 16, H/2, W/2)

        x = self.up1(x)                      # (B, 16, H, W)
        x = self.dec1(torch.cat([x, x1], dim=1)) # (B, 8, H, W)

        return self.final(x)                 # (B, 2, H, W)


class DeltaNet(nn.Module):
    def __init__(self, dim=768, num_heads=1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # Convolutional layer to learn local context between feat1 and feat2
        self.local_conv = nn.Sequential(
            nn.Conv1d(2 * dim, dim, kernel_size=3, padding=1, groups=8),  # Learn interaction between feat1 and feat2
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)  # Final output layer after interaction
        )

        # Conv1d for fusion: takes 2*dim (from feat1 and feat2) -> dim
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(2 * dim, dim, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, feat1, feat2):

        combined_feat = torch.cat([feat1, feat2], dim=-1)  # [B, 256, 2*768] -> combine the features

        # Learn local interactions between the two feature maps
        local_feat = self.local_conv(combined_feat.transpose(1, 2)).transpose(1, 2)  # [B, 256, 768]

        # Cross-attention to compute the global context between feat1 and feat2
        attn_output, _ = self.cross_attn(feat2, feat1, feat1)  # [B, 256, 768]

        # Concatenate local features and attention output
        fused = torch.cat([local_feat, attn_output], dim=-1)  # [B, 256, 768] + [B, 256, 768] = [B, 256, 2*768]

        # Fuse with 1x1 Conv1d to produce final delta feature map
        delta = self.fuse_conv(fused.transpose(1, 2)).transpose(1, 2)  # [B, 256, 768]

        return delta


class DinoDeltaModel(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', displacement_net = False):

        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)

        self.delta_net = DeltaNet(dim=768)

        self.rgb_proj = nn.Conv2d(768, 3, kernel_size=1)

        self.pixel_displacement  = PixelDisplacement(3)
        self.displacement_net = displacement_net

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False


    def extract_features(self, x):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)  # [B, 256, 768]
        return feats["x_norm_patchtokens"]

    def tokens_to_spatial(self, feat, H=16, W=16, out_h=224, out_w=224):
        B, N, C = feat.shape
        feat_reshaped = feat.permute(0, 2, 1).reshape(B, C, H, W)
        feat_upsampled = F.interpolate(feat_reshaped, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return feat_upsampled

    def forward(self, img1, img2):
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)
        
        # Cross-attention & delta computation
        delta = self.delta_net(feat1, feat2)  # [B, 256, 768]

        # Reshape to [B, C, H, W]
        B, N, C = delta.shape
        delta_reshaped = delta.permute(0, 2, 1).reshape(B, C, 16, 16)  # [B, 768, 16, 16]

        delta_upsampled = F.interpolate(delta_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
        feat1_upsampled = self.rgb_proj(self.tokens_to_spatial(feat1))
        feat2_upsampled = self.rgb_proj(self.tokens_to_spatial(feat2))
        delta_upsampled = self.rgb_proj(delta_upsampled)

        if self.displacement_net:
            pixel_displacement  = self.pixel_displacement(delta_upsampled)
            return feat1_upsampled, feat2_upsampled, delta_upsampled, pixel_displacement
        else:
            return feat1_upsampled, feat2_upsampled, delta_upsampled