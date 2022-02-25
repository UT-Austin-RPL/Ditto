import glob
import os
import random

import numpy as np
import torch
from numpy.lib.arraysetops import isin
from omegaconf import ListConfig
from torch.utils.data import Dataset

from src.utils.misc import occ_to_binary_label, sample_occ_points, sample_point_cloud
from src.utils.transform import Rotation


# different occ points and seg points
# represent articulation as dense joints
# include transformed surface points
class GeoArtDatasetV0(Dataset):
    def __init__(self, opt):
        if isinstance(opt["data_path"], ListConfig):
            # multi class
            self.path_list = []
            for data_path in opt["data_path"]:
                self.path_list.extend(
                    glob.glob(
                        os.path.join(opt["data_dir"], data_path, "scenes", "*.npz")
                    )
                )
        else:
            self.path_list = glob.glob(
                os.path.join(opt["data_dir"], opt["data_path"], "scenes", "*.npz")
            )

        if opt.get("num_data"):
            random.shuffle(self.path_list)
            self.path_list = self.path_list[: opt["num_data"]]
        self.num_point = opt["num_point"]
        self.num_point_occ = opt["num_point_occ"]
        self.num_point_seg = opt["num_point_seg"]
        self.norm = opt.get("norm", False)
        self.rand_rot = opt.get("rand_rot", False)
        self.rand_scale = min(opt.get("rand_scale", 0), np.pi * 2)
        self.rand_scale = max(self.rand_scale, 0)
        self.weighted_occ_sample = opt.get("weighted_occ_sample", False)
        if self.norm:
            self.norm_padding = opt.get("norm_padding", 0.1)

    def __getitem__(self, index):
        data = np.load(self.path_list[index])
        pc_start, pc_start_idx = sample_point_cloud(data["pc_start"], self.num_point)
        pc_end, pc_end_idx = sample_point_cloud(data["pc_end"], self.num_point)
        pc_start_end = data["pc_start_end"][pc_start_idx]
        pc_end_start = data["pc_end_start"][pc_end_idx]
        pc_seg_label_start = data["pc_seg_start"][pc_start_idx]
        pc_seg_label_end = data["pc_seg_end"][pc_end_idx]
        state_start = data["state_start"]
        state_end = data["state_end"]
        screw_axis = data["screw_axis"]
        screw_moment = data["screw_moment"]
        joint_type = data["joint_type"]
        joint_index = data["joint_index"]
        # shape2motion's 0 joint start from base object
        if "Shape2Motion" in self.path_list[index]:
            occ_label, seg_label = occ_to_binary_label(
                data["start_occ_list"], joint_index + 1
            )
        else:
            if "syn" in self.path_list[index]:
                joint_index = 1
            occ_label, seg_label = occ_to_binary_label(
                data["start_occ_list"], joint_index
            )

        # process occ and seg points

        occ_label = occ_label.astype(np.bool)
        p_occ_start_postive = data["start_p_occ"][occ_label]
        seg_label = seg_label[occ_label]
        p_seg_start, seg_idx_start = sample_point_cloud(
            p_occ_start_postive, self.num_point_seg
        )
        seg_label = seg_label[seg_idx_start]

        if self.weighted_occ_sample:
            start_occ_density = data["start_occ_density"]
            occ_idx_start = sample_occ_points(
                occ_label - 0.5, start_occ_density, self.num_point_occ
            )
            occ_label = occ_label[occ_idx_start]
            p_occ_start = data["start_p_occ"][occ_idx_start]
        else:
            p_occ_start, occ_idx_start = sample_point_cloud(
                data["start_p_occ"], self.num_point_occ
            )
            occ_label = occ_label[occ_idx_start]

        screw_point = np.cross(screw_axis, screw_moment)
        # random rotation
        if self.rand_rot:
            ax, ay, az = (torch.rand(3).numpy() - 0.5) * self.rand_scale
            rand_rot_mat = Rotation.from_euler("xyz", (ax, ay, az)).as_matrix()
            pc_start = rand_rot_mat.dot(pc_start.T).T
            pc_end = rand_rot_mat.dot(pc_end.T).T
            pc_start_end = rand_rot_mat.dot(pc_start_end.T).T
            pc_end_start = rand_rot_mat.dot(pc_end_start.T).T
            p_occ_start = rand_rot_mat.dot(p_occ_start.T).T
            p_seg_start = rand_rot_mat.dot(p_seg_start.T).T
            screw_axis = rand_rot_mat.dot(screw_axis)
            screw_point = rand_rot_mat.dot(screw_point)

        # normalize
        bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
        bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max()
        scale = scale * (1 + self.norm_padding)

        pc_start = (pc_start - center) / scale
        pc_end = (pc_end - center) / scale
        p_occ_start = (p_occ_start - center) / scale
        p_seg_start = (p_seg_start - center) / scale
        pc_start_end = (pc_start_end - center) / scale
        pc_end_start = (pc_end_start - center) / scale

        # ((p - c) / s) X l
        # = (p X l) / s - (c X l) / s
        # = m / s - (c X l) / s
        screw_point = (screw_point - center) / scale

        screw_moment = np.cross(screw_point, screw_axis)

        # screw_moment = screw_moment / scale - np.cross(center, screw_axis) / scale
        if joint_type == 1:
            # prismatic joint, state change with scale
            state_start /= scale
            state_end /= scale

        # only change revolute joint
        # only change z-axis joints
        if screw_axis[2] <= -0.9 and joint_type == 0:
            screw_axis = -screw_axis
            screw_moment = -screw_moment
            state_start, state_end = state_end, state_start

        screw_point = np.cross(screw_axis, screw_moment)
        p2l_vec, p2l_dist = batch_perpendicular_line(
            p_seg_start, screw_axis, screw_point
        )

        return_dict = {
            "pc_start": pc_start,  # N, 3
            "pc_end": pc_end,
            "pc_start_end": pc_start_end,
            "pc_end_start": pc_end_start,
            "pc_seg_label_start": pc_seg_label_start,
            "pc_seg_label_end": pc_seg_label_end,
            "state_start": state_start,
            "state_end": state_end,
            "screw_axis": screw_axis,
            "screw_moment": screw_moment,
            "p2l_vec": p2l_vec,
            "p2l_dist": p2l_dist,
            "joint_type": joint_type,
            "joint_index": np.array(joint_index),
            "p_occ": p_occ_start,
            "occ_label": occ_label,
            "p_seg": p_seg_start,
            "seg_label": seg_label,
            "scale": scale,
            "center": center,
            "data_path": self.path_list[index],
        }
        for k, v in return_dict.items():
            if isinstance(v, np.ndarray):
                return_dict[k] = torch.from_numpy(v).float()
        return return_dict

    def __len__(self):
        return len(self.path_list)


def batch_perpendicular_line(
    x: np.ndarray, l: np.ndarray, pivot: np.ndarray
) -> np.ndarray:
    """
    x: B * 3
    l: 3
    pivot: 3
    p_l: B * 3
    """
    offset = x - pivot
    p_l = offset.dot(l)[:, np.newaxis] * l[np.newaxis] - offset
    dist = np.sqrt(np.sum(p_l ** 2, axis=-1))
    p_l = p_l / (dist[:, np.newaxis] + 1.0e-5)
    return p_l, dist
