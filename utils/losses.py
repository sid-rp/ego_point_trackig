import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_feature_at_kp(self, feat_map, kp_coords):
        """
        feat_map: [B, C, H, W]
        kp_coords: [B, N, 2] in (x, y) pixel coordinates
        Returns: [B, N, C]
        """
        B, C, H, W = feat_map.shape
        x = kp_coords[..., 0] / (W - 1) * 2 - 1  # Normalize x to [-1, 1]
        y = kp_coords[..., 1] / (H - 1) * 2 - 1  # Normalize y to [-1, 1]
        grid = torch.stack((x, y), dim=-1).unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # [B, C, N, 1]
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C]


    def forward(self, feat1, feat2, kp1, kp2, kp1_mask, kp2_mask, delta_map):
        """
        feat1, feat2, delta_map: [B, C, H, W]
        kp1, kp2: [B, N, 2] - keypoint locations
        kp1_mask, kp2_mask: [B, N] - valid keypoints
        Returns: scalar loss
        """
        f1_at_kp = self.extract_feature_at_kp(feat1, kp1)         # [B, N, C]
        f2_at_kp = self.extract_feature_at_kp(feat2, kp2)         # [B, N, C]
        delta_at_kp = self.extract_feature_at_kp(delta_map, kp2)  # [B, N, C]

        f2_plus_delta = f2_at_kp + delta_at_kp

        valid_mask = kp1_mask & kp2_mask               # [B, N]
        
        # Masked tensors
        f1_valid = f1_at_kp[valid_mask]                # [num_valid, C]
        f2_valid = f2_plus_delta[valid_mask]           # [num_valid, C]

        return F.mse_loss(f1_valid, f2_valid)


# def apply_mask_and_retrieve_displacements(delta_x, x2, kp2_mask, H, W):
#     """
#     Extract valid displacements using keypoint positions and mask.

#     Args:
#         delta_x: (B, 2, H, W) - predicted displacements (dx, dy)
#         x2: (B, N, 2) - keypoints in image 2 (y, x), already int64
#         kp2_mask: (B, N) - bool mask for valid keypoints
#         H, W: Image dimensions

#     Returns:
#         valid_displacements: (B, N, 2) - N = total keypoints in each batch
#         valid_x2: (B, N, 2)
#     """
#     B, N, _ = x2.shape
#     device = delta_x.device

#     # Gather y, x coordinates
#     y = x2[:, :, 0]  # (B, N) y-coordinates
#     x = x2[:, :, 1]  # (B, N) x-coordinates

#     # Cast y and x to long (int64) for indexing
#     y = y.long()  # Cast y to long (int64)
#     x = x.long()  # Cast x to long (int64)

#     # Get displacement values at those pixel locations
#     dx = delta_x[:, 0, :, :]  # (B, H, W)
#     dy = delta_x[:, 1, :, :]  # (B, H, W)

#     # Use batch indexing to gather dx, dy
#     batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)  # (B, N)
#     disp_x = dx[batch_idx, y, x]  # (B, N)
#     disp_y = dy[batch_idx, y, x]  # (B, N)

#     # Stack to (B, N, 2)
#     displacements = torch.stack([disp_x, disp_y], dim=-1)  # (B, N, 2)
    
#     # Apply mask
#     mask = kp2_mask.bool()  # (B, N)

#     # Select valid displacements and valid keypoints, preserving the original shape (B, N, 2)
#     valid_displacements = displacements * mask.unsqueeze(-1)  # (B, N, 2)
#     valid_x2 = x2 * mask.unsqueeze(-1)  # (B, N, 2)

#     return valid_displacements, valid_x2


def apply_mask_and_retrieve_displacements(pixel_delta, x1, x2, kp1_mask, kp2_mask, H, W):
    """
    Extract valid displacements using keypoint positions and masks.

    Args:
        pixel_delta: (B, 2, H, W) - predicted displacements (dx, dy) in 
        x1: (B, N, 2) - ground truth keypoints in image 1
        x2: (B, N, 2) - keypoints in image 2 (y, x), already int64
        kp1_mask: (B, N) - bool mask for valid keypoints in x1
        kp2_mask: (B, N) - bool mask for valid keypoints in x2
        H, W: Image dimensions

    Returns:
        valid_displacements: (B, N, 2)
        valid_x2: (B, N, 2)
        valid_x1: (B, N, 2)
    """
    B, N, _ = x2.shape
    device = pixel_delta.device

    # Gather y, x coordinates for indexing delta
    y = torch.clamp(x2[:, :, 0].long(), 0, H - 1)  # (B, N)
    x = torch.clamp(x2[:, :, 1].long(), 0, W - 1)  # (B, N)

    # Get displacement values at keypoint positions
    dx = pixel_delta[:, 0, :, :]  # (B, H, W)
    dy = pixel_delta[:, 1, :, :]  # (B, H, W)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)  # (B, N)
    disp_x = dx[batch_idx, y, x]  # (B, N)
    disp_y = dy[batch_idx, y, x]  # (B, N)

    displacements = torch.stack([disp_x, disp_y], dim=-1)  # (B, N, 2)

    # Combined mask
    mask = (kp1_mask & kp2_mask).bool()  # (B, N)

    # Apply mask to all three outputs
    valid_displacements = displacements * mask.unsqueeze(-1)  # (B, N, 2)
    valid_x2 = x2 * mask.unsqueeze(-1)  # (B, N, 2)
    valid_x1 = x1 * mask.unsqueeze(-1)  # (B, N, 2)

    return valid_displacements, valid_x2, valid_x1




# class SampsonLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super(SampsonLoss, self).__init__()
#         self.eps = eps

#     def forward(self, x2, delta_x, F, kp2_mask, H=224, W=224):

#         # Apply mask and retrieve valid displacements and keypoints
#         valid_delta, valid_x2 = apply_mask_and_retrieve_displacements(delta_x, x2, kp2_mask, H, W)
        
#         # x1 = x2 + delta (only valid keypoints)
#         x1 = valid_x2 + valid_delta

#         B, N, _ = x1.shape  # B: batch size, N: number of valid keypoints

#         # Homogeneous coordinates
#         ones = torch.ones(B, N, 1, device=x1.device)
#         x1_h = torch.cat([x1, ones], dim=2)  # (B, N, 3)
#         x2_h = torch.cat([valid_x2, ones], dim=2)  # (B, N, 3)

#         # Reshape for bmm (batch matrix multiplication)
#         x1_h_T = x1_h.transpose(1, 2)  # (B, 3, N)
#         x2_h_T = x2_h.transpose(1, 2)  # (B, 3, N)

#         # Compute Fx2 and Ftx1 (Batch matrix multiplication)
#         Fx2 = torch.bmm(F, x2_h_T)  # (B, 3, N)
#         Ftx1 = torch.bmm(F.transpose(1, 2), x1_h_T)  # (B, 3, N)

#         # x1^T * F * x2 for each batch
#         Fx2_T = Fx2.transpose(1, 2)  # (B, N, 3)
#         x1tFx2 = torch.sum(x1_h * Fx2_T, dim=2)  # (B, N)

#         # Sampson error
#         numerator = x1tFx2 ** 2

#         # sum of squared terms for Fx2 and Ftx1
#         denom = Fx2[:, 0]**2 + Fx2[:, 1]**2 + Ftx1[:, 0]**2 + Ftx1[:, 1]**2  # (B, N)

#         # Compute the Sampson error and return the mean over the batch
#         sampson_error = numerator / (denom + self.eps)  # (B, N)

        
#         return sampson_error.mean()  # Return mean over the batch

class PixelDisplacementLoss(nn.Module):
    def __init__(self):
        super(PixelDisplacementLoss, self).__init__()

    def forward(self, x1, x2, kp1_mask, kp2_mask, pixel_delta, H=224, W=224):
        """
        Compute L2 loss between predicted keypoints (x2 + delta) and ground truth keypoints x1.

        Args:
            x1: (B, N, 2) - ground truth keypoints in image 1
            x2: (B, N, 2) - keypoints in image 2 (y, x) integer coords
            kp1_mask: (B, N) - valid mask for x1
            kp2_mask: (B, N) - valid mask for x2
            pixel_delta: (B, 2, H, W) - predicted displacement map
            H, W: image dimensions

        Returns:
            Scalar loss
        """
        # Predicted displacements from delta_x
        predicted_pixel_delta, valid_x2, valid_x1 = apply_mask_and_retrieve_displacements(pixel_delta, x1,  x2, kp1_mask, kp2_mask, H, W)

        # Predicted x1 = x2 + delta
        pred_x1 = valid_x2 + predicted_pixel_delta  # (B, N, 2)

        # Apply combined mask
        valid_mask = (kp1_mask & kp2_mask).bool()  # (B, N)

       
        return F.mse_loss(pred_x1, valid_x1)