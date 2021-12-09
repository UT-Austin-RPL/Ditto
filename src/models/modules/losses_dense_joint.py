import torch
import torch.nn.functional as F
from torch import nn
from torch._C import TracingState

from src.utils import utils

log = utils.get_logger(__name__)


def norm(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return (tensor ** 2).sum(dim).sqrt()


def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)


def angle_axis_to_rotation_matrix(angle_axis, theta):
    # Stolen from PyTorch geometry library. Modified for our code
    angle_axis_shape = angle_axis.shape
    angle_axis_ = angle_axis.contiguous().view(-1, 3)
    theta_ = theta.contiguous().view(-1, 1)

    k_one = 1.0
    normed_axes = angle_axis_ / angle_axis_.norm(dim=-1, keepdim=True)
    wx, wy, wz = torch.chunk(normed_axes, 3, dim=1)
    cos_theta = torch.cos(theta_)
    sin_theta = torch.sin(theta_)

    r00 = cos_theta + wx * wx * (k_one - cos_theta)
    r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
    r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
    r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
    r11 = cos_theta + wy * wy * (k_one - cos_theta)
    r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
    r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
    r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
    r22 = cos_theta + wz * wz * (k_one - cos_theta)
    rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
    return rotation_matrix.view(list(angle_axis_shape[:-1]) + [3, 3])


def mask_mean(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (data * mask).sum(-1) / (mask.sum(-1) + 1.0e-5)


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


def cosine(pred_axis, gt_axis, ambiguity=False):
    # pred: B * N * 3
    # target: B * 3
    if ambiguity:
        cosine_sim_0 = torch.einsum("bnm,bm->bn", pred_axis, -gt_axis)
        cosine_sim_1 = torch.einsum("bnm,bm->bn", pred_axis, gt_axis)
        cosine_sim_max = torch.maximum(cosine_sim_0, cosine_sim_1)
    else:
        cosine_sim_max = torch.einsum("bnm,bm->bn", pred_axis, gt_axis)
    return cosine_sim_max


class PrismaticLoss(nn.Module):
    def __init__(self, param_dict):
        super().__init__()
        self.param_dict = param_dict
        if self.param_dict["p_cos_ambiguity"] and self.param_dict["p_use_state_loss"]:
            raise ValueError("Don't use ambiguous cosine loss & enforce state loss")

    def forward(self, seg_label, pred_axis, pred_t, gt_axis, gt_t, debug=False):  # B*N
        """
        pred_axis: B*N*3
        pred_t: B*N
        gt_axis: B*3
        gt_t: B
        """
        pred_axis_ = normalize(pred_axis, 2)
        # axis ori loss
        loss = 0.0
        ori_cos = cosine(
            pred_axis_, gt_axis, ambiguity=self.param_dict["p_cos_ambiguity"]
        )
        if self.param_dict["p_ori_arccos"]:
            loss_ori = torch.arccos(torch.clamp(ori_cos, min=-0.9999, max=0.9999))
        else:
            loss_ori = -ori_cos
        if self.param_dict["no_seg_mask"]:
            loss_ori = loss_ori.mean(-1)
        else:
            loss_ori = mask_mean(loss_ori, seg_label)
        loss += self.param_dict["p_ori_weight"] * loss_ori
        # state loss
        if self.param_dict["p_use_state_loss"]:
            if self.param_dict["no_seg_mask"]:
                loss += F.l1_loss(pred_t, gt_t.unsqueeze(-1), reduction="none").mean(-1)
            else:
                loss += mask_mean(
                    F.l1_loss(pred_t, gt_t.unsqueeze(-1), reduction="none"),
                    seg_label,
                )
        # translation loss
        diff = gt_axis.unsqueeze(1) * gt_t.unsqueeze(-1).unsqueeze(
            -1
        ) - pred_axis_ * pred_t.unsqueeze(-1)
        translation = norm(diff, -1)
        if self.param_dict["no_seg_mask"]:
            translation = translation.mean(-1)
        else:
            translation = mask_mean(translation, seg_label)
        # B
        loss += self.param_dict["p_offset_weight"] * translation

        if torch.isnan(loss.sum()):
            log.error("Loss NaN")

        if debug:
            print("loss orientation:", loss_ori.detach().cpu())
            print("loss translation:", translation.detach().cpu())
        return loss, {
            "axis_ori": loss_ori
            if self.param_dict["p_ori_arccos"]
            else torch.arccos(-loss_ori)
        }


class RevoluteLoss(nn.Module):
    def __init__(self, param_dict):
        super().__init__()
        self.param_dict = param_dict
        if self.param_dict["r_cos_ambiguity"] and self.param_dict["r_use_state_loss"]:
            raise ValueError("Don't use ambiguous cosine loss & enforce state loss")

    def forward(
        self,
        points,  # B*N*3
        seg_label,  # B*N
        pred_axis,  # B*N*3
        pred_t,  # B*N
        pred_p2l_vec,  # B*N*3
        pred_p2l_dist,  # B*N
        gt_axis,  # B*3
        gt_t,  # B
        gt_p2l_vec,  # B*N*3
        gt_p2l_dist,  # B*N
        debug=False,
    ):
        pred_axis_ = normalize(pred_axis, 2)
        pred_p2l_vec_ = normalize(pred_p2l_vec, 2)
        # rotation axis ori loss
        loss = 0.0
        ori_cos = cosine(
            pred_axis_, gt_axis, ambiguity=self.param_dict["r_cos_ambiguity"]
        )
        if self.param_dict["r_ori_arccos"]:
            loss_ori = torch.arccos(torch.clamp(ori_cos, min=-0.9999, max=0.9999))
        else:
            loss_ori = -ori_cos
        if self.param_dict["no_seg_mask"]:
            loss_ori = loss_ori.mean(-1)
        else:
            loss_ori = mask_mean(loss_ori, seg_label)
        loss += self.param_dict["r_ori_weight"] * loss_ori

        # point to axis, direction ori loss
        loss_p2l_ori = -torch.einsum("bnm,bnm->bn", pred_p2l_vec_, gt_p2l_vec)
        if self.param_dict["r_p2l_ori_arccos"]:
            loss_p2l_ori = torch.arccos(
                torch.clamp(-loss_p2l_ori, min=-0.9999, max=0.9999)
            )
        if self.param_dict["no_seg_mask"]:
            loss_p2l_ori = loss_p2l_ori.mean(-1)
        else:
            loss_p2l_ori = mask_mean(loss_p2l_ori, seg_label)
        loss += self.param_dict["r_p2l_ori_weight"] * loss_p2l_ori

        # point to axis, distance loss
        loss_p2l_dist = F.l1_loss(pred_p2l_dist, gt_p2l_dist, reduction="none")
        if self.param_dict["no_seg_mask"]:
            loss_p2l_dist = loss_p2l_dist.mean(-1)
        else:
            loss_p2l_dist = mask_mean(loss_p2l_dist, seg_label)
        loss += self.param_dict["r_p2l_dist_weight"] * loss_p2l_dist

        # state loss
        if self.param_dict["r_use_state_loss"]:
            if self.param_dict["no_seg_mask"]:
                loss_state = F.l1_loss(
                    pred_t, gt_t.unsqueeze(-1), reduction="none"
                ).mean(-1)
            else:
                loss_state = mask_mean(
                    F.l1_loss(pred_t, gt_t.unsqueeze(-1), reduction="none"),
                    seg_label,
                )
            loss += self.param_dict["r_state_weight"] + loss_state

        # rotation loss
        # B * N * 3 * 3
        rotation_pred = angle_axis_to_rotation_matrix(pred_axis_, pred_t)
        # B * 1 * 3 * 3
        rotation_gt = angle_axis_to_rotation_matrix(
            gt_axis.unsqueeze(1), gt_t.unsqueeze(1)
        )
        rotation_gt = rotation_gt.repeat(1, rotation_pred.size(1), 1, 1)
        I_ = torch.eye(3).reshape((1, 3, 3))
        I_ = I_.repeat(rotation_pred.size(0) * rotation_pred.size(1), 1, 1).to(
            rotation_pred.device
        )
        loss_rot = torch.norm(
            I_
            - torch.bmm(
                rotation_pred.view(-1, 3, 3),
                rotation_gt.view(-1, 3, 3).transpose(1, 2),
            ),
            dim=(1, 2),
            p=2,
        ).view(rotation_pred.shape[:2])
        if self.param_dict["no_seg_mask"]:
            loss_rot = loss_rot.mean(-1)
        else:
            loss_rot = mask_mean(loss_rot, seg_label)
        loss += self.param_dict["r_rot_weight"] * loss_rot

        # displacement loss
        pred_pivot_point = points + pred_p2l_vec_ * pred_p2l_dist.unsqueeze(-1)
        gt_pivot_point = points + gt_p2l_vec * gt_p2l_dist.unsqueeze(-1)
        # B * N * 3
        rotated_pred = (
            torch.einsum("bnxy,bny->bnx", rotation_pred, points - pred_pivot_point)
            + pred_pivot_point
        )
        rotated_gt = (
            torch.einsum("bnxy,bny->bnx", rotation_gt, points - gt_pivot_point)
            + gt_pivot_point
        )
        displacement = ((rotated_pred - rotated_gt) ** 2).sum(-1).sqrt()
        if self.param_dict["no_seg_mask"]:
            displacement = displacement.mean(-1)
        else:
            displacement = mask_mean(displacement, seg_label)
        loss += self.param_dict["r_displacement_weight"] * displacement

        if torch.isnan(loss.sum()):
            log.error("Loss NaN")

        if debug:
            print("loss orientation:", loss_ori.detach().cpu())
            print("loss p2l vec:", loss_p2l_ori.detach().cpu())
            print("loss p2l dist:", loss_p2l_dist.detach().cpu())
            print("loss rotation:", loss_rot.detach().cpu())
            print("displacement:", displacement.detach().cpu())
        return loss, {
            "axis_ori": loss_ori
            if self.param_dict["r_ori_arccos"]
            else torch.arccos(-loss_ori),
            "p2l_ori": loss_p2l_ori
            if self.param_dict["r_p2l_ori_arccos"]
            else torch.arccos(-loss_p2l_ori),
            "p2l_dist": loss_p2l_dist,
            "displacement": displacement,
            "rotation_pred": rotation_pred,
            "pred_pivot_point": pred_pivot_point,
            "pred_axis": pred_axis_,
            "pred_t": pred_t,
        }


def rotate_points(points, axis, pivot, theta):
    """
    points: B * N * 3
    axis: B * N * 3
    pivot: B * N * 3
    theta: B * N
    """
    rotation_mat = angle_axis_to_rotation_matrix(axis, theta)
    rotated_points = torch.einsum("bnxy,bny->bnx", rotation_mat, points - pivot) + pivot
    return rotated_points
