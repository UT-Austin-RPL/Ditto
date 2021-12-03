import torch
import torch.nn.functional as F
from torch import nn


class DistanceBasedAttentionLoss(nn.Module):
    def __init__(self, temp_pred, temp_target):
        super().__init__()
        self.temp_pred = temp_pred
        self.temp_target = temp_target
        self.cri = nn.KLDivLoss(reduction="none")

    def forward(self, pc1, pc2, idx1, idx2, score, return_gt=False):
        """
        pc1: B*N*3
        pc2: B*N*3
        idx1: B*N'*3
        idx2: B*N'*3
        score: B*N'*N' / B*h*N'*N'
        """

        fps_points1 = torch.gather(pc1, 1, idx1.unsqueeze(-1).repeat(1, 1, 3))
        fps_points2 = torch.gather(pc2, 1, idx2.unsqueeze(-1).repeat(1, 1, 3))
        # B*N'*N'*3
        diff = fps_points1.unsqueeze(2) - fps_points2.unsqueeze(1)
        dist = (diff ** 2).sum(-1).sqrt()
        gt_attn_score = F.softmax(-dist / self.temp_target, dim=-1)
        if len(score.shape) == 4:  # multi-head attention
            num_head = score.size(1)
            gt_attn_score = gt_attn_score.unsqueeze(1).repeat(1, num_head, 1, 1)
        # B*N'*N' / B*h*N'*N'
        loss = self.cri(
            torch.log_softmax(score / self.temp_pred, dim=-1), gt_attn_score
        )
        loss = loss.sum(-1)
        loss = loss.mean(-1)
        if len(loss.shape) > 1:  # multi-head attention
            loss = loss.mean(-1)
        if return_gt:
            return loss, gt_attn_score
        else:
            return loss


def norm(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return (tensor ** 2).sum(dim).sqrt()


def batch_eye(batch_size: int, dim: int) -> torch.Tensor:
    e = torch.eye(dim)
    e = e.unsqueeze(0)
    e = e.repeat(batch_size, 1, 1)
    return e


def rotation_matrix_from_axis(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    theta = theta.unsqueeze(-1).unsqueeze(-1)
    batch_size = axis.size(0)
    R = batch_eye(batch_size, 3).to(axis.device) * torch.cos(theta)
    R += skew(axis) * torch.sin(theta)
    R += (1 - torch.cos(theta)) * torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1))
    return R


def skew(vector: torch.Tensor) -> torch.Tensor:
    # vector: B*3
    result = torch.zeros(vector.size(0), 3, 3).to(vector.device)
    result[:, 0, 1] = -vector[:, 2]
    result[:, 0, 2] = vector[:, 1]
    result[:, 1, 0] = vector[:, 2]
    result[:, 1, 2] = -vector[:, 0]
    result[:, 2, 0] = -vector[:, 1]
    result[:, 2, 1] = vector[:, 0]
    return result


# def cosine_loss_with_ambiguity(pred_axis, gt_axis):
#     # pred: B * 3
#     # target: B * 3
#     cosine_sim_0 = torch.einsum('bm,bm->b', pred_axis, -gt_axis)
#     cosine_sim_1 = torch.einsum('bm,bm->b', pred_axis, gt_axis)
#     cosine_sim_max = torch.maximum(cosine_sim_0, cosine_sim_1)
#     return -cosine_sim_max
