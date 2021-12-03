"""
From the implementation of https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch
import torch.nn as nn

from src.third_party.ConvONets.encoder.pointnetpp_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
)


def MLP_head(in_dim, dims, use_bn):
    head = nn.Sequential()
    dim_src = in_dim
    for i, dim_dst in enumerate(dims):
        head.add_module(name="linear_%d" % (i), module=nn.Conv1d(dim_src, dim_dst, 1))
        if i == len(dims) - 1:
            break
        if use_bn:
            head.add_module(name="bn_%d" % (i), module=nn.BatchNorm1d(dim_dst))
        head.add_module(name="relu_%d" % (i), module=nn.ReLU())
        dim_src = dim_dst
    return head


# concat displacement-based feature with original feature
# and global displacement-based feature
class PointNetPlusPlusCorrFusion(nn.Module):
    def __init__(
        self,
        dim=None,
        c_dim=128,
        padding=0.1,
        mlp_kwargs=None,
        corr_aggregation="mean",
        **kwargs
    ):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[512, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, c_dim])

        if len(mlp_kwargs.dims) > 0:
            corr_dim = mlp_kwargs.dims[-1]
        else:  # not coordinate embedding
            corr_dim = 3
        self.fp2_corr = PointNetFeaturePropagation(
            in_channel=2 * corr_dim + 384, mlp=[512, 256]
        )
        self.fp1_corr = PointNetFeaturePropagation(
            in_channel=256, mlp=[256, 128, c_dim]
        )

        if corr_aggregation == "mean":
            self.corr_pool = nn.AdaptiveAvgPool1d(1)
        elif corr_aggregation == "max":
            self.corr_pool = nn.AdaptiveMaxPool1d(1)

        self.coord_emb = MLP_head(3, mlp_kwargs.dims, mlp_kwargs.bn)

    def encode_deep_feature(self, xyz, return_xyz=False):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        if return_xyz:
            return l2_points, l2_xyz, fps_idx
        else:
            return l2_points

    def forward(self, xyz, xyz2, return_score=False):
        """
        xyz: B*N*3
        xyz2: B*N*3
        -------
        return:
        B*N'*3
        B*N'*C
        B*N'
        B*N'
        B*N'*N'
        """
        b_size = xyz.size(0)
        xyz = xyz.permute(0, 2, 1)
        l2_points_xyz2, l2_xyz2, fps_idx2 = self.encode_deep_feature(
            xyz2, return_xyz=True
        )

        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # B * 3 * N
        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        # B * 128 * 128
        score = torch.matmul(l2_points.transpose(-2, -1), l2_points_xyz2)
        score = torch.softmax(score, dim=-1)
        # B * 3 * 128 * 128
        displacement = l2_xyz.unsqueeze(-1) - l2_xyz2.unsqueeze(-2)
        num_abs_point = displacement.size(2)
        displacement = displacement.view(b_size, 3, -1)
        displacement_feat = self.coord_emb(displacement).view(
            b_size, -1, num_abs_point, num_abs_point
        )
        # B * C * 128 * 128; weighted sum
        displacement_feat = displacement_feat * score.unsqueeze(1)
        # B * C * 128
        displacement_feat = displacement_feat.sum(-1)
        # B * C * 1
        displacement_feat_global = self.corr_pool(displacement_feat)
        displacement_feat_global = displacement_feat_global.repeat(1, 1, num_abs_point)
        # B * (C+256) * 128
        displacement_feat = torch.cat(
            (displacement_feat_global, displacement_feat, l2_points), dim=1
        )
        # mean_displacement =
        # displacement.view(b_size, 3, num_abs_point, num_abs_point) * score.unsqueeze(1)
        # mean_displacement = mean_displacement.sum(-1)

        l1_points_new = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points_new)

        l1_points_corr = self.fp2_corr(l1_xyz, l2_xyz, l1_points, displacement_feat)
        l0_points_corr = self.fp1_corr(l0_xyz, l1_xyz, None, l1_points_corr)

        if return_score:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points_corr.permute(0, 2, 1),
                fps_idx,
                fps_idx2,
                score,
            )
        else:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points_corr.permute(0, 2, 1),
            )
