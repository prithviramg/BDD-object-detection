import torch
import torch.nn.functional as F


def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (B, H*W, C)
    feat = _gather_feat(feat, ind)
    return feat


def focal_loss(preds, targets):
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)

    pred = torch.clamp(preds, 1e-6, 1 - 1e-6)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    return (pos_loss + neg_loss).sum() / (pos_inds.sum() + 1e-6)


def reg_l2_loss(preds, targets, mask):
    mask = mask.unsqueeze(2).expand_as(preds).float()
    loss = F.mse_loss(preds * mask, targets * mask, reduction="sum")
    return loss / (mask.sum() + 1e-6)


class MultiScaleCenterNetLoss(torch.nn.Module):
    def __init__(self, wh_weight=0.1, off_weight=1.0, scale_weights=None):
        super().__init__()
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.scale_weights = scale_weights  # Optional list of weights per scale

    def forward(self, outputs_per_scale, targets_per_scale):
        total_loss = 0.0
        num_scales = len(outputs_per_scale)

        for i, (outputs, targets) in enumerate(
            zip(outputs_per_scale, targets_per_scale)
        ):
            heatmap_pred, wh_pred, offset_pred = outputs
            heatmap_gt = targets["heatmap"]
            wh_gt = targets["wh"]
            reg_mask = targets["reg_mask"]
            ind = targets["ind"]
            offset_gt = targets["reg"]

            wh_pred = _transpose_and_gather_feat(wh_pred, ind)
            offset_pred = _transpose_and_gather_feat(offset_pred, ind)

            heatmap_loss = focal_loss(heatmap_pred, heatmap_gt)
            wh_loss = reg_l2_loss(wh_pred, wh_gt, reg_mask)
            offset_loss = reg_l2_loss(offset_pred, offset_gt, reg_mask)

        return heatmap_loss, wh_loss, offset_loss
        #     scale_loss = heatmap_loss + self.wh_weight * wh_loss + self.off_weight * offset_loss

        #     weight = self.scale_weights[i] if self.scale_weights else 1.0
        #     total_loss += weight * scale_loss

        # return total_loss
