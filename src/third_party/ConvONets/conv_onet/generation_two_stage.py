import numpy as np
import torch
import torch.optim as optim
import trimesh
from torch import autograd
from tqdm import tqdm, trange

from src.third_party.ConvONets.common import (
    add_key,
    coord2index,
    make_3d_grid,
    normalize_coord,
)

try:
    from src.third_party.ConvONets.utils import libmcubes
    from src.third_party.ConvONets.utils.libmise import MISE
    from src.third_party.ConvONets.utils.libsimplify import simplify_mesh
except:
    print("import utils failed")
import math
import time

counter = 0


class Generator3D(object):
    """Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    """

    def __init__(
        self,
        model,
        points_batch_size=100000,
        threshold=0.5,
        seg_threshold=0.5,
        refinement_step=0,
        device=None,
        resolution0=16,
        upsampling_steps=3,
        with_normals=False,
        padding=0.1,
        sample=False,
        input_type=None,
        vol_info=None,
        vol_bound=None,
        simplify_nfaces=None,
        sdf=False,
    ):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.seg_threshold = seg_threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.sdf = sdf

        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def generate_mesh(self, data, return_stats=True):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
        self.model.eval()
        device = self.device
        stats_dict = {}

        kwargs = {}

        t0 = time.time()

        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(data["pc_start"], data["pc_end"])
        stats_dict["time (encode inputs)"] = time.time() - t0

        mesh_dict, mobile_points_all = self.generate_from_latent(
            c, stats_dict=stats_dict, **kwargs
        )

        if return_stats:
            return mesh_dict, mobile_points_all, c, stats_dict
        else:
            return mesh_dict, mobile_points_all, c

    def generate_from_latent(self, c=None, stats_dict={}, **kwargs):
        """Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        """
        threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        mesh_dict = {}
        mobile_points_list = []
        for index in (0, 1):
            # Shortcut
            if self.upsampling_steps == 0:
                nx = self.resolution0
                pointsf = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (nx,) * 3)

                values, mobile_points = self.eval_points(pointsf, c, index, **kwargs)
                values = values.cpu().numpy()
                mobile_points_list.append(mobile_points)
                value_grid = values.reshape(nx, nx, nx)
            else:
                mesh_extractor = MISE(
                    self.resolution0, self.upsampling_steps, threshold
                )

                points = mesh_extractor.query()
                while points.shape[0] != 0:
                    # Query points
                    pointsf = points / mesh_extractor.resolution
                    # Normalize to bounding box
                    pointsf = box_size * (pointsf - 0.5)
                    pointsf = torch.FloatTensor(pointsf).to(self.device)
                    # Evaluate model and update
                    values, mobile_points = self.eval_points(
                        pointsf, c, index, **kwargs
                    )
                    values = values.cpu().numpy()
                    mobile_points_list.append(mobile_points)
                    values = values.astype(np.float64)
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()

                value_grid = mesh_extractor.to_dense()

                # Extract mesh
                stats_dict["time (eval points)"] = time.time() - t0

                mesh_dict[index] = self.extract_mesh(
                    value_grid, c, stats_dict=stats_dict
                )
        mobile_points_all = torch.cat(mobile_points_list, dim=1)
        return mesh_dict, mobile_points_all

    def eval_points(self, p, c, index, **kwargs):
        """Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): encoded feature volumes
            index (int): the index of part to query
        """
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        mobile_points = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode_occ(pi, c, **kwargs)
                seg_hat = self.model.decode_seg(pi, c, **kwargs)
                if self.sdf:
                    occ_hat = -occ_hat

            seg_prob = torch.sigmoid(seg_hat)
            if index == 0:
                occ_hat[seg_prob > self.seg_threshold] -= 1e5
            elif index == 1:
                occ_hat[seg_prob <= self.seg_threshold] -= 1e5

            mobile_points.append(
                pi[
                    :,
                    torch.logical_and(
                        seg_prob > self.seg_threshold,
                        torch.sigmoid(occ_hat) > self.threshold,
                    )[0],
                ]
            )
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        mobile_points = torch.cat(mobile_points, dim=1)
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat, mobile_points

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        """Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        """
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(occ_hat, 1, "constant", constant_values=-1e6)
        # import pdb; pdb.set_trace()
        vertices, triangles = libmcubes.marching_cubes(occ_hat_padded, threshold)
        stats_dict["time (marching cubes)"] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1

        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound["query_vol"][:, 0].min(axis=0)
            bb_max = self.vol_bound["query_vol"][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (
                self.vol_bound["axis_n_crop"].max()
                * self.resolution0
                * 2 ** self.upsampling_steps
            )
            vertices = vertices * mc_unit + bb_min
        else:
            # Normalize to bounding box
            vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
            vertices = box_size * (vertices - 0.5)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict["time (normals)"] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(
            vertices, triangles, vertex_normals=normals, process=False
        )

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.0)
            stats_dict["time (simplify)"] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict["time (refine)"] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        """Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        """
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode_occ(vi, c)
            if self.sdf:
                occ_hat = -occ_hat
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        """Refines the predicted mesh.

        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        """

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert n_x == n_y == n_z
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            occ_hat = self.model.decode_occ(face_point.unsqueeze(0), c)
            if self.sdf:
                occ_hat = -occ_hat
            face_value = torch.sigmoid(occ_hat)
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True
            )[0]

            normal_target = normal_target / (
                normal_target.norm(dim=1, keepdim=True) + 1e-10
            )
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
