import torch
import torch.nn as nn
import torch.nn.functional as F

from src.third_party.ConvONets.common import (
    map2local,
    normalize_3d_coordinate,
    normalize_coordinate,
)
from src.third_party.ConvONets.layers import ResnetBlockFC


class FCDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
    dim (int): input dimension
    c_dim (int): dimension of latent conditioned code c
    out_dim (int): dimension of latent conditioned code c
    leaky (bool): whether to use leaky ReLUs
    sample_mode (str): sampling feature strategy, bilinear|nearest
    padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim

        self.fc = nn.Linear(dim + c_dim, out_dim)
        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)

        net = self.fc(torch.cat((c, p), dim=2)).squeeze(-1)

        return net


class LocalDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        concat_feat_4=False,
        no_xyz=False,
    ):
        super().__init__()
        self.concat_feat = concat_feat or concat_feat_4
        if concat_feat:
            c_dim *= 3
        elif concat_feat_4:
            c_dim *= 4
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size) for i in range(n_blocks)]
            )

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c.append(self.sample_grid_feature(p, c_plane["grid"]))
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)

        p = p.float()

        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


# different feature for different head
class LocalDecoderV1(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        feature_keys=None,
        concat_feat=True,
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.feature_keys = feature_keys
        self.concat_feat = concat_feat

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                for k in self.feature_keys:
                    if "grid" in k:
                        c.append(self.sample_grid_feature(p, c_plane[k]))
                    elif "xy" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xy"))
                    elif "yz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="yz"))
                    elif "xz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                for k in self.feature_keys:
                    if "grid" in k:
                        c += self.sample_grid_feature(p, c_plane[k])
                    elif "xy" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xy")
                    elif "yz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="yz")
                    elif "xz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xz")
                c = c.transpose(1, 2)

        p = p.float()

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
