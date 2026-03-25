import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask=None):

        if valid_mask is not None:
            valid_mask = valid_mask.detach()
            diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
            loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                              self.lambd * torch.pow(diff_log.mean(), 2))

        else:

            diff_log = torch.log(target) - torch.log(pred)
            loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                              self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class PixelVarianceLoss(nn.Module):
    def __init__(self):
        super(PixelVarianceLoss, self).__init__()

    def forward(self, pred_depth, gt_depth, wolf_mask=None, world_size=7):
        """
        Args:
            pred_depth (torch.Tensor): Tensor of shape (Batch, H, W).
            gt_depth (torch.Tensor): Tensor of shape (B, C, H, W) => pseudo GT depth.
            mask (torch.Tensor, optional): Binary mask of shape (H, W) for Wolf (0 for static, 255 for dynamic).
        
        Returns:
            torch.Tensor: Variance loss calculated over masked static pixels.
        """
        #depth = F.interpolate(raw_depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

        #gt_depth = torch.mean(gt_depth, dim=1) # Shape (B, H, W)
        #gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=pred_depth.shape[-2:], mode='bilinear', align_corners=True).squeeze(0) # Shape (B, H, W)
        gt_depth = torch.mean(gt_depth, dim=0) # Shape (H, W)

        #Compute variance across the batch dimension (dim=0)
        pixel_variance_loss = torch.var(pred_depth, dim=0, unbiased=False) # Shape (H, W)
        #pixel_variance_loss = torch.mean(pixel_variance_loss)
        predicted_mean_depth = torch.mean(pred_depth, dim=0)  # Shape (H, W)

        #mean_consistency_loss = torch.mean((predicted_mean_depth - gt_depth) ** 2)
        mean_consistency_loss = (predicted_mean_depth - gt_depth) ** 2 # Shape (H, W)

        if wolf_mask is not None:
            wolf_mask = wolf_mask.detach()
            # Invert the mask
            no_wolf_mask = (wolf_mask==0).float()

            pixel_variance_loss = pixel_variance_loss * no_wolf_mask
            mean_consistency_loss = mean_consistency_loss * no_wolf_mask

        combined_loss =  pixel_variance_loss + mean_consistency_loss
        # else:
        #     loss = torch.mean(pixel_variance_loss)

        return combined_loss



# class PixelVarianceLoss(nn.Module):
#     def __init__(self):
#         super(PixelVarianceLoss, self).__init__()

#     def forward(self, pred_depth, gt_depth, mask=None, world_size=7):
#         # Ensure pred_depth has requires_grad=True
#         if not pred_depth.requires_grad:
#             pred_depth.requires_grad_()

#         # Calculate the local mean and variance
#         local_mean = torch.mean(pred_depth, dim=0, keepdim=True)  # Mean across the batch on each GPU
#         local_sq_diff = (pred_depth - local_mean) ** 2
#         local_variance = torch.mean(local_sq_diff, dim=0, keepdim=True)  # Variance across batch

#         # Detach local_mean and local_variance from the graph before using all_reduce
#         global_mean = local_mean.clone()
#         dist.all_reduce(global_mean, op=torch.distributed.ReduceOp.SUM)
#         global_mean /= world_size  # Global mean

#         global_variance = local_variance.clone()
#         dist.all_reduce(global_variance, op=torch.distributed.ReduceOp.SUM)
#         global_variance /= world_size  # Global variance

#         pixel_variance_loss = global_variance

#         # Predicted mean depth for mean consistency loss
#         predicted_mean_depth = global_mean

#         # Mean of ground truth depth
#         global_gt_depth = gt_depth.clone()  # Detach to avoid autograd warning
#         dist.all_reduce(global_gt_depth, op=torch.distributed.ReduceOp.SUM)
#         global_gt_depth /= world_size  # Global mean of ground truth

#         # Apply mask if provided
#         if mask is not None:
#             predicted_mean_depth = predicted_mean_depth * mask
#             pixel_variance_loss = pixel_variance_loss * mask

#         # Mean consistency loss calculation
#         mean_consistency_loss = (predicted_mean_depth - global_gt_depth) ** 2
#         combined_loss = pixel_variance_loss + mean_consistency_loss

#         return combined_loss.squeeze()




# class PixelVarianceLoss(nn.Module):
#     def __init__(self):
#         super(PixelVarianceLoss, self).__init__()

#     def forward(self, pred_depth, gt_depth, mask=None):
#         """
#         Args:
#             pred_depth (torch.Tensor): Tensor of shape (Batch, H, W).
#             gt_depth (torch.Tensor): Tensor of shape (B, C, H, W) => pseudo GT depth.
#             mask (torch.Tensor, optional): Binary mask of shape (H, W) for static pixels (1 for static, 0 for dynamic).
        
#         Returns:
#             torch.Tensor: Variance loss calculated over masked static pixels.
#         """
#         #depth = F.interpolate(raw_depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

#         #gt_depth = torch.mean(gt_depth, dim=1) # Shape (B, H, W)
#         #gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=pred_depth.shape[-2:], mode='bilinear', align_corners=True).squeeze(0) # Shape (B, H, W)
#         gt_depth = torch.mean(gt_depth, dim=0) # Shape (H, W)
        
#         #Compute variance across the batch dimension (dim=0)
#         pixel_variance_loss = torch.var(pred_depth, dim=0, unbiased=False) # Shape (H, W)
#         #pixel_variance_loss = torch.mean(pixel_variance_loss)
#         predicted_mean_depth = torch.mean(pred_depth, dim=0)  # Shape (H, W)

#         if mask is not None:
#             predicted_mean_depth = predicted_mean_depth * mask
#             pixel_variance_loss = pixel_variance_loss * mask  
#             #loss = torch.sum(pixel_variance_loss) / torch.sum(mask)

#         #mean_consistency_loss = torch.mean((predicted_mean_depth - gt_depth) ** 2)
#         mean_consistency_loss = (predicted_mean_depth - gt_depth) ** 2 # Shape (H, W)

#         combined_loss =  pixel_variance_loss + mean_consistency_loss
#         # else:
#         #     loss = torch.mean(pixel_variance_loss)

#         return combined_loss
