import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .croco_downstream import *
from .pos_embed import interpolate_pos_embed


class DeltaNet(nn.Module):
    def __init__(self, dim=768, num_heads=1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.local_conv = nn.Sequential(
            nn.Conv1d(2 * dim, dim, kernel_size=3, padding=1, groups=8),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv1d(2 * dim, dim, kernel_size=1),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(dim)


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)

    def forward(self, feat1, feat2):
        combined_feat = torch.cat([feat1, feat2], dim=-1)
        local_feat = self.local_conv(combined_feat.transpose(1, 2)).transpose(1, 2)
        attn_output, _ = self.cross_attn(feat2, feat1, feat1)
        fused = torch.cat([local_feat, attn_output], dim=-1)
        delta = self.fuse_conv(fused.transpose(1, 2)).transpose(1, 2)
        delta = self.norm(delta)
        return delta


WEIGHTS_PATH = "/scratch/projects/fouheylab/dma9300/OSNOM/weights/CroCo_V2_ViTBase_SmallDecoder.pth"
class CrocoF(nn.Module):
    def __init__(self, model_weights = WEIGHTS_PATH, delta = False):
        super().__init__()
        self.delta  = delta
        assert os.path.isfile(model_weights)
        ckpt = torch.load(model_weights, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        self.backbone = CroCoDownstreamMonocularEncoder(**croco_args)
        interpolate_pos_embed(self.backbone, ckpt['model'])
        _  = self.backbone.load_state_dict(ckpt['model'], strict=False)

        #Freeze all layers of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        B = x.shape[0]
        with torch.no_grad():
            layers = self.backbone(x)  # (B, 196, 768)
            layers = [layers[i] for i in [-1]]

        patch_tokens = torch.cat(layers, dim=-1)  # (B, 196, 768)
        return patch_tokens
       

    def forward(self, img1):
        token = self.extract_features(img1)

        return token



class DinoF(nn.Module):
    def __init__(self, model_name='dinov2_vitb14'):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)

        #Freeze all layers of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def extract_features(self, x):
        B = x.shape[0]
        with torch.no_grad():
            layers = self.backbone.get_intermediate_layers(x, n=1)  # (B, 256, 768)

        patch_tokens = torch.cat(layers, dim=-1)  # (B, 196, 768)
        return patch_tokens
       

    def forward(self, img1):
        token = self.extract_features(img1)

        return token


WEIGHTS_PATH = "/scratch/projects/fouheylab/dma9300/OSNOM/weights/CroCo_V2_ViTBase_SmallDecoder.pth"
class CrocoDeltaNet(nn.Module):
    def __init__(self, model_weights = WEIGHTS_PATH, delta = False):
        super().__init__()
        self.delta  = delta
        assert os.path.isfile(model_weights)
        ckpt = torch.load(model_weights, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        self.backbone = CroCoDownstreamMonocularEncoder(**croco_args)
        interpolate_pos_embed(self.backbone, ckpt['model'])
        _  = self.backbone.load_state_dict(ckpt['model'], strict=False)

        # Conv-based projection with LayerNorm-compatible GroupNorm
        self.feature_proj_conv = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=1),
            nn.GroupNorm(1, 768),  # Simulates LayerNorm for conv output
            nn.GELU()
        )
        if self.delta:
            self.delta_net = DeltaNet(dim=768)
            self.res_scale = nn.Parameter(torch.tensor(0.1))  # Learnable residual scaling


        self.norm = nn.LayerNorm(768)

        #Freeze all layers of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def extract_features(self, x):
        B = x.shape[0]
        with torch.no_grad():
            layers = self.backbone(x)  # (B, 196, 768)
            layers = [layers[i] for i in [-1]]

        patch_tokens = torch.cat(layers, dim=-1)  # (B, 196, 768)
        feat = patch_tokens.permute(0, 2, 1).reshape(B, 768, 14, 14)
        feat_proj = self.feature_proj_conv(feat)  # (B, 768, 14, 13)
        feat_px = F.interpolate(feat_proj, size=(224, 224), mode='bilinear', align_corners=True)
        patch_feat = feat_proj.flatten(2).transpose(1, 2)  # (B, 2196, 768)
        patch_feat = self.norm(patch_feat)
        return patch_feat, feat_px

    def forward(self, img1, img2):
        feat1_patch, feat1_px = self.extract_features(img1)
        feat2_patch, feat2_px = self.extract_features(img2)
        if self.delta:
            delta_patch = self.delta_net(feat1_patch, feat2_patch)
            B, N, C = delta_patch.shape
            delta_pixel = delta_patch.transpose(1, 2).reshape(B, 768, 14, 14)
            delta_px = F.interpolate(delta_pixel, size=(224, 224), mode='bilinear', align_corners=True)
            feat2_px = feat2_px + self.res_scale * delta_px

        return feat1_px, feat2_px


class DinoDeltaModel(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', delta = False):
        super().__init__()
        self.delta  = delta
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)

        # Conv-based projection with LayerNorm-compatible GroupNorm
        self.feature_proj_conv = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=1),
            nn.GroupNorm(1, 768),  # Simulates LayerNorm for conv output
            nn.GELU()
        )

        if self.delta:
            self.delta_net = DeltaNet(dim=768)
            self.res_scale = nn.Parameter(torch.tensor(0.1))  # Learnable residual scaling

        self.norm = nn.LayerNorm(768)

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x):
        B = x.shape[0]
        with torch.no_grad():
            layers = self.backbone.get_intermediate_layers(x, n=1)  # (B, 256, 768)

        patch_tokens = torch.cat(layers, dim=-1)  # (B, 256, 3072)
        feat = patch_tokens.permute(0, 2, 1).reshape(B, 768, 16, 16)
        feat_proj = self.feature_proj_conv(feat)  # (B, 768, 16, 16)
        feat_px = F.interpolate(feat_proj, size=(224, 224), mode='bilinear', align_corners=True)
        patch_feat = feat_proj.flatten(2).transpose(1, 2)  # (B, 256, 768)
        patch_feat = self.norm(patch_feat)
        return patch_feat, feat_px

    def forward(self, img1, img2):
        feat1_patch, feat1_px = self.extract_features(img1)
        feat2_patch, feat2_px = self.extract_features(img2)
        if self.delta:
            delta_patch = self.delta_net(feat1_patch, feat2_patch)
            B, N, C = delta_patch.shape
            delta_pixel = delta_patch.transpose(1, 2).reshape(B, 768, 16, 16)
            delta_px = F.interpolate(delta_pixel, size=(224, 224), mode='bilinear', align_corners=False)
            feat2_px = feat2_px + self.res_scale * delta_px
        return feat1_px, feat2_px
