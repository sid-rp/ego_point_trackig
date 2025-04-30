import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .croco_downstream import *
from .pos_embed import interpolate_pos_embed

WEIGHTS_PATH = "/scratch/projects/fouheylab/dma9300/OSNOM/weights/CroCo_V2_ViTBase_SmallDecoder.pth"

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, num_heads=1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, q_feat, kv_feat):
        # q_feat, kv_feat: (B, N, C)
        q = self.norm_q(q_feat)
        kv = self.norm_kv(kv_feat)
        out, _ = self.cross_attn(q, kv, kv)
        return q_feat * out

class FeatureMixer(nn.Module):
    def __init__(self, dim=768, num_heads=1, out_tokens=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.token_proj = nn.Linear(452, out_tokens)  # Project token dim

    def forward(self, dino_feat, croco_feat):
        x = torch.cat([dino_feat, croco_feat], dim=1)  # (B, 452, 768)
        x = x * self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = self.mlp(self.norm2(x))  # (B, 452, 768)
        x = x.transpose(1, 2)  # (B, 768, 452)
        x = self.token_proj(x)  # (B, 768, 256)
        return x.transpose(1, 2)  # (B, 256, 768)


class MLPHeatmapHead(nn.Module):
    def __init__(self, input_dim=768, output_size=(224, 224), num_keypoints=2):
        super().__init__()
        self.output_size = output_size
        self.num_keypoints = num_keypoints
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_keypoints * output_size[0] * output_size[1]),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

    def forward(self, x):
        """
        x: Tensor of shape [B, N, C]
        Returns: Tensor of shape [B, num_keypoints, H, W]
        """
        B, N, C = x.shape
        x = torch.max(x, dim=1).values  # Max pooling over token embeddings: [B, C]
        out = self.mlp(x)  # [B, num_keypoints * H * W]
        out = out.view(B, self.num_keypoints, self.output_size[0], self.output_size[1])  # [B, num_keypoints, H, W]
        return out



class KeyPointNet(nn.Module):
    def __init__(self, model_weights=WEIGHTS_PATH):
        super().__init__()
        
        # Load DINOv2 backbone
        self.dino_net = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14", pretrained=True)
        
        assert os.path.isfile(model_weights), f"Model weights not found: {model_weights}"
        ckpt = torch.load(model_weights, map_location="cpu")
        croco_args = croco_args_from_ckpt(ckpt)

        # Load CroCo backbone
        self.croco_net = CroCoDownstreamMonocularEncoder(**croco_args)
        interpolate_pos_embed(self.croco_net, ckpt['model'])
        _ = self.croco_net.load_state_dict(ckpt['model'], strict=False)

        # Freeze both backbones
        for param in self.dino_net.parameters():
            param.requires_grad = False
        for param in self.croco_net.parameters():
            param.requires_grad = False

        # Fusion and projection layers
        self.feature_mixer = FeatureMixer()
        self.interaction_layer = CrossAttentionFusion()
        self.proj_to_map = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        )
        # self.key_point_head  = MLPHeatmapHead()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                continue  # Skip frozen layers

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def extract_croco_features(self, x):
        with torch.no_grad():
            features = self.croco_net(x)
            return torch.cat([features[i] for i in [-1]], dim=-1)  # (B, 196, 768)

    def extract_dino_features(self, x):
        with torch.no_grad():
            features = self.dino_net.get_intermediate_layers(x, n=1)
            return torch.cat(features, dim=-1)  # (B, 256, 768)

    def forward(self, x1, x2):

        # Extract features
        dino_feat1 = self.extract_dino_features(x1)
        croco_feat1 = self.extract_croco_features(x1)
        dino_feat2 = self.extract_dino_features(x2)
        croco_feat2 = self.extract_croco_features(x2)

        # Mix features
        feat1_mix = self.feature_mixer(dino_feat1, croco_feat1)
        feat2_mix = self.feature_mixer(dino_feat2, croco_feat2)

        # Cross-attend between the two sets
        feat1_tokens = self.interaction_layer(feat1_mix, feat2_mix)
        feat2_tokens = self.interaction_layer(feat2_mix, feat1_mix)
        # kpts2_pred  = self.key_point_head(feat2_tokens)

        # Reshape to 2D and upsample
        B = feat1_tokens.size(0)
        feat1_2d = feat1_tokens.transpose(1, 2).reshape(B, 768, 16, 16)
        feat2_2d = feat2_tokens.transpose(1, 2).reshape(B, 768, 16, 16)

        out1 = self.proj_to_map(feat1_2d)
        out2 = self.proj_to_map(feat2_2d)
        return out1, out2


    
        
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

class MLPFeatureProjector(nn.Module):
    def __init__(self, in_dim=2*768, mid_dim=512, out_dim=768, H=14, W=14):
        super().__init__()
        self.H, self.W = H, W
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, in_dim, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # → (B, H*W, C)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = x.permute(0, 2, 1).view(B, -1, H, W)  # → (B, out_dim, H, W)
        return x

WEIGHTS_PATH = "/scratch/projects/fouheylab/dma9300/OSNOM/weights/CroCo_V2_ViTBase_SmallDecoder.pth"
class CrocoMultiLayerFeatures(nn.Module):
    def __init__(self, model_weights=WEIGHTS_PATH):
        super().__init__()
        assert os.path.isfile(model_weights), f"Model weights not found: {model_weights}"
        ckpt = torch.load(model_weights, map_location="cpu")
        croco_args = croco_args_from_ckpt(ckpt)

        # Initialize backbone
        self.backbone = CroCoDownstreamMonocularEncoder(**croco_args)
        interpolate_pos_embed(self.backbone, ckpt['model'])
        _ = self.backbone.load_state_dict(ckpt['model'], strict=False)

        # Feature projection head (input dim = 4*768 if concatenating 4 layers)
        self.mlp_proj = MLPFeatureProjector()

        # Upsample using nn.Upsample with bilinear interpolation and lightweight convolution
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)  # 14x14 -> 224x224
        self.refinement_conv = nn.Conv2d(768, 768, kernel_size=3, padding=1)  # Lightweight convolution to refine

        # self.norm = nn.LayerNorm(2 * 768)

        # Freeze all backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        """Extract dense and patch-level features from image input using layers 6, 8, 10, 12."""
        B = x.shape[0]
        with torch.no_grad():
            layers = self.backbone(x)  # List of layer outputs
            selected_layers = [layers[i] for i in [6, 8]]  # 0-indexed

        patch_tokens = torch.cat(selected_layers, dim=-1)  # (B, 196, 4*768)
        # patch_tokens = self.norm(patch_tokens)
        feat = patch_tokens.permute(0, 2, 1).reshape(B, 2 * 768, 14, 14)
        feat_proj = self.mlp_proj(feat)
        
        # Upsample the projected feature map
        feat_up = self.upsample(feat_proj)
        feat_px = self.refinement_conv(feat_up)  # Lightweight refinement step
        
        return feat_px

    def forward(self, img1, img2):
        """Forward pass: extract pixel features from both images."""
        feat1_px = self.extract_features(img1)
        feat2_px = self.extract_features(img2)
        return feat1_px, feat2_px



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

        # Freeze all layers of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # for param in self.backbone.enc_blocks[-1].parameters():
        #     param.requires_grad = True
        

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
