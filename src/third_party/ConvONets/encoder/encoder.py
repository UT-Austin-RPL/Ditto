import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean

from src.third_party.ConvONets.common import (
    coordinate2index,
    normalize_3d_coordinate,
    normalize_coordinate,
)
from src.third_party.ConvONets.encoder.pointnetpp_attn import PointNetPlusPlusAttnFusion
from src.third_party.ConvONets.encoder.pointnetpp_corr import PointNetPlusPlusCorrFusion
from src.third_party.ConvONets.encoder.unet import UNet
from src.third_party.ConvONets.encoder.unet3d import UNet3D
from src.third_party.ConvONets.layers import ResnetBlockFC


class LocalPoolPointnetPPFusion(nn.Module):
    """PointNet++Attn-based encoder network with ResNet blocks for each point.
        The network takes two inputs and fuse them with Attention layer
        Number of input points are fixed.
        Separate features for geometry and articulation

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(
        self,
        c_dim=128,
        dim=3,
        hidden_dim=128,
        scatter_type="max",
        mlp_kwargs=None,
        attn_kwargs=None,
        unet=False,
        unet_kwargs=None,
        unet3d=False,
        unet3d_kwargs=None,
        unet_corr=False,
        unet_kwargs_corr=None,
        unet3d_corr=False,
        unet3d_kwargs_corr=None,
        corr_aggregation=None,
        plane_resolution=None,
        grid_resolution=None,
        plane_type="xz",
        padding=0.0,
        n_blocks=5,
        feat_pos="attn",
        return_score=False,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.return_score = return_score
        if feat_pos == "attn":
            self.feat_pos = PointNetPlusPlusAttnFusion(
                c_dim=hidden_dim * 2, attn_kwargs=attn_kwargs
            )
        elif feat_pos == "corr":
            self.feat_pos = PointNetPlusPlusCorrFusion(
                c_dim=hidden_dim * 2,
                mlp_kwargs=mlp_kwargs,
                corr_aggregation=corr_aggregation,
            )
        else:
            raise NotImplementedError(f"Encoder {feat_pos} not implemented!")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.blocks_corr = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c_corr = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if unet_corr:
            self.unet_corr = UNet(c_dim, in_channels=c_dim, **unet_kwargs_corr)
        else:
            self.unet_corr = None

        if unet3d_corr:
            self.unet3d_corr = UNet3D(**unet3d_kwargs_corr)
        else:
            self.unet3d_corr = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def generate_plane_features(self, p, c, plane="xz", unet=None):
        # acquire indices of features in plane
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(
            p.size(0), self.c_dim, self.reso_plane, self.reso_plane
        )  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if unet is not None:
            fea_plane = unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c, unet3d=None):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(
            p.size(0),
            self.c_dim,
            self.reso_grid,
            self.reso_grid,
            self.reso_grid,
        )  # sparce matrix (B x 512 x reso x reso)

        if unet3d is not None:
            fea_grid = unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_grid ** 3,
                )
            else:
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_plane ** 2,
                )
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p, p2):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if "xz" in " ".join(self.plane_type):
            coord["xz"] = normalize_coordinate(
                p.clone(), plane="xz", padding=self.padding
            )
            index["xz"] = coordinate2index(coord["xz"], self.reso_plane)
        if "xy" in " ".join(self.plane_type):
            coord["xy"] = normalize_coordinate(
                p.clone(), plane="xy", padding=self.padding
            )
            index["xy"] = coordinate2index(coord["xy"], self.reso_plane)
        if "yz" in " ".join(self.plane_type):
            coord["yz"] = normalize_coordinate(
                p.clone(), plane="yz", padding=self.padding
            )
            index["yz"] = coordinate2index(coord["yz"], self.reso_plane)
        if "grid" in " ".join(self.plane_type):
            coord["grid"] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index["grid"] = coordinate2index(
                coord["grid"], self.reso_grid, coord_type="3d"
            )
        _, net, net_corr = self.feat_pos(p, p2, return_score=self.return_score)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)

        net_corr = self.blocks_corr[0](net_corr)
        for block_corr in self.blocks_corr[1:]:
            pooled = self.pool_local(coord, index, net_corr)
            net_corr = torch.cat([net_corr, pooled], dim=2)
            net_corr = block_corr(net_corr)
        c_corr = self.fc_c_corr(net_corr)

        fea = {}
        for f in self.plane_type:
            k1, k2 = f.split("_")
            if k2 in ["xy", "yz", "xz"]:
                if k1 == "geo":
                    fea[f] = self.generate_plane_features(
                        p, c, plane=k2, unet=self.unet
                    )
                elif k1 == "corr":
                    fea[f] = self.generate_plane_features(
                        p, c_corr, plane=k2, unet=self.unet_corr
                    )
            elif k2 == "grid":
                if k1 == "geo":
                    fea[f] = self.generate_grid_features(p, c, unet3d=self.unet3d)
                elif k1 == "corr":
                    fea[f] = self.generate_grid_features(
                        p, c_corr, unet3d=self.unet3d_corr
                    )
        return fea
