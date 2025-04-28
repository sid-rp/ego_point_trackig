import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointAlignmentLossL2(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_feature_at_kp(self, feat_map, kp_coords):
        """
        feat_map: [B, C, H, W]
        kp_coords: [B, N, 2] in (x, y) pixel coordinates
        Returns: [B, N, C]
        """
        B, C, H, W = feat_map.shape
        x = kp_coords[..., 0] / (W - 1) * 2 - 1
        y = kp_coords[..., 1] / (H - 1) * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # [B, C, N, 1]
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C]

    def forward(self, feat1, feat2, kp1, kp2, kp1_mask, kp2_mask):
        """
        feat1, feat2, delta_map: [B, 768, H, W]
        kp1, kp2: [B, N, 2]
        kp1_mask, kp2_mask: [B, N]
        Returns: scalar loss
        """
        B, C, H, W = feat1.shape
        f1_at_kp = self.extract_feature_at_kp(feat1, kp1)         # [B, N, C]
        f2_at_kp = self.extract_feature_at_kp(feat2, kp2)         # [B, N, C]
        # delta_at_kp = self.extract_feature_at_kp(delta_map, kp2)  # [B, N, C]

        f2_plus_delta = f2_at_kp # [B, N, C]

        valid_mask = kp1_mask & kp2_mask  # [B, N]
        valid_mask = valid_mask.unsqueeze(-1)  # [B, N, 1] for broadcasting
        f1_norm = F.normalize(f1_at_kp, dim=-1)
        f2_norm = F.normalize(f2_plus_delta, dim=-1)

        # Compute L2 distance
        l2_dist = ((f1_norm - f2_norm) ** 2).sum(dim=-1)  # [B, N]

        # Apply mask to zero out invalid keypoints
        l2_dist = l2_dist * valid_mask.squeeze(-1)

        # Compute mean loss over valid points
        total_valid = valid_mask.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)

        loss = l2_dist.sum() / total_valid
        return loss


class KeypointAlignmentLossCosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_feature_at_kp(self, feat_map, kp_coords):
        """
        feat_map: [B, C, H, W]
        kp_coords: [B, N, 2] in (x, y) pixel coordinates
        Returns: [B, N, C]
        """
        B, C, H, W = feat_map.shape
        x = kp_coords[..., 0] / (W - 1) * 2 - 1
        y = kp_coords[..., 1] / (H - 1) * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # [B, C, N, 1]
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C]

    def forward(self, feat1, feat2, kp1, kp2, kp1_mask, kp2_mask):
        """
        feat1, feat2: [B, 768, H, W]
        kp1, kp2: [B, N, 2]
        kp1_mask, kp2_mask: [B, N]
        Returns: scalar loss
        """
        B, C, H, W = feat1.shape
        f1_at_kp = self.extract_feature_at_kp(feat1, kp1)         # [B, N, C]
        f2_at_kp = self.extract_feature_at_kp(feat2, kp2)         # [B, N, C]

        f2_plus_delta = f2_at_kp # [B, N, C]

        valid_mask = kp1_mask & kp2_mask  # [B, N]
        valid_mask = valid_mask.unsqueeze(-1)  # [B, N, 1] for broadcasting

        # Normalize before cosine sim
        f1_norm = F.normalize(f1_at_kp, dim=-1)
        f2_norm = F.normalize(f2_plus_delta, dim=-1)

        cosine_sim = (f1_norm * f2_norm).sum(dim=-1)  # [B, N]
        cosine_sim = cosine_sim * valid_mask.squeeze(-1)  # zero out invalid

        total_valid = valid_mask.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)
        
        loss = 1.0 - cosine_sim.sum() / total_valid
        return loss


class CombinedKeypointLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_feature_at_kp(self, feat_map, kp_coords):
        """
        feat_map: [B, C, H, W]
        kp_coords: [B, N, 2] in (x, y) pixel coordinates
        Returns: [B, N, C]
        """
        B, C, H, W = feat_map.shape
        x = kp_coords[..., 0] / (W - 1) * 2 - 1
        y = kp_coords[..., 1] / (H - 1) * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # [B, C, N, 1]
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C]

    def forward(self, feat1, feat2, kp1, kp2, kp1_mask, kp2_mask):
        """
        feat1, feat2: [B, 768, H, W]
        kp1, kp2: [B, N, 2]
        kp1_mask, kp2_mask: [B, N]
        Returns: scalar loss
        """
        f1_at_kp = self.extract_feature_at_kp(feat1, kp1)  # [B, N, C]
        f2_at_kp = self.extract_feature_at_kp(feat2, kp2)  # [B, N, C]

        valid_mask = (kp1_mask & kp2_mask).unsqueeze(-1)  # [B, N, 1]

        # Normalize for cosine and L2
        f1_norm = F.normalize(f1_at_kp, dim=-1)
        f2_norm = F.normalize(f2_at_kp, dim=-1)

        # Cosine similarity loss
        cosine_sim = (f1_norm * f2_norm).sum(dim=-1)  # [B, N]
        cosine_loss = (1.0 - cosine_sim) * valid_mask.squeeze(-1)  # [B, N]

        # L2 loss
        l2_loss = ((f1_norm - f2_norm) ** 2).sum(dim=-1) * valid_mask.squeeze(-1)  # [B, N]

        total_valid = valid_mask.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)

        # Combine with equal weights
        combined_loss = (cosine_loss.sum() + l2_loss.sum()) / (2 * total_valid)
        return combined_loss
