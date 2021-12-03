import os

import numpy as np
import trimesh

from src.utils.visual import as_mesh


def sample_point_cloud(pc, num_point):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(
        np.arange(num_point_all),
        size=(num_point,),
        replace=num_point > num_point_all,
    )
    return pc[idxs], idxs


def binary_occ(occ_list, idx):
    if not isinstance(occ_list, list):
        occ_list = [x for x in occ_list]
    occ_fore = occ_list.pop(idx)
    occ_back = np.zeros_like(occ_fore)
    for o in occ_list:
        occ_back += o
    return occ_fore, occ_back


def occ_to_label(occ_list, idx):
    if not isinstance(occ_list, list):
        occ_list = [x for x in occ_list]
    occ_fore = occ_list.pop(idx)
    occ_back = np.zeros_like(occ_fore)
    for o in occ_list:
        occ_back += o
    occ_label = np.zeros_like(occ_fore, dtype=np.int64)
    occ_label[occ_back] = 1
    occ_label[occ_fore] = 2
    return occ_label


def occ_to_binary_label(occ_list, idx):
    if not isinstance(occ_list, list):
        occ_list = [x for x in occ_list]
    occ_fore = occ_list.pop(idx)
    occ_back = np.zeros_like(occ_fore)
    for o in occ_list:
        occ_back += o
    occ_label = np.zeros_like(occ_fore, dtype=np.int64)
    seg_label = np.zeros_like(occ_fore, dtype=np.int64)
    # occupied space is positive
    occ_label[occ_back] = 1
    occ_label[occ_fore] = 1
    # foreground/mobile part is positive
    seg_label[occ_fore] = 1
    return occ_label, seg_label


def sample_occ_points(values, densities, num_points, MODE="weighted"):
    if MODE == "random":
        sampled_pids = np.random.randint(values.shape[0], size=num_points)
        return sampled_pids
    elif MODE == "weighted":
        half_sampling_num = int(num_points / 2)
        pos_inds = np.argwhere(values > 0)
        neg_inds = np.argwhere(values < 0)
        if len(pos_inds) <= num_points / 5 or len(neg_inds) <= num_points / 5:
            # if there is too few points in one side...
            # just random sampling here
            sampled_pids = np.random.randint(values.shape[0], size=num_points)
            return sampled_pids
        else:
            pos_probs = densities[pos_inds] / np.sum(densities[pos_inds])
            pos_probs = np.squeeze(pos_probs, axis=1)
            pos_probs = np.squeeze(pos_probs, axis=1)
            neg_probs = densities[neg_inds] / np.sum(densities[neg_inds])
            neg_probs = np.squeeze(neg_probs, axis=1)
            neg_probs = np.squeeze(neg_probs, axis=1)

            if pos_inds.shape[0] > half_sampling_num:
                sampled_pos_inds = np.random.choice(
                    pos_inds.shape[0],
                    size=half_sampling_num,
                    replace=False,
                    p=pos_probs,
                )
            else:
                sampled_pos_inds = np.random.choice(
                    pos_inds.shape[0],
                    size=half_sampling_num - pos_inds.shape[0],
                    replace=False,
                    p=pos_probs,
                )
                another = np.array([i for i in range(pos_inds.shape[0])])
                sampled_pos_inds = np.concatenate((sampled_pos_inds, another), axis=0)
            sampled_pos_inds = pos_inds[sampled_pos_inds]
            if neg_inds.shape[0] > half_sampling_num:
                sampled_neg_inds = np.random.choice(
                    neg_inds.shape[0],
                    size=half_sampling_num,
                    replace=False,
                    p=neg_probs,
                )
            else:
                sampled_neg_inds = np.random.choice(
                    neg_inds.shape[0],
                    size=half_sampling_num - neg_inds.shape[0],
                    p=neg_probs,
                )
                another = np.array([i for i in range(neg_inds.shape[0])])
                sampled_neg_inds = np.concatenate((sampled_neg_inds, another), axis=0)
            sampled_neg_inds = neg_inds[sampled_neg_inds]

            sampled_pids = np.concatenate((sampled_pos_inds, sampled_neg_inds), axis=0)
            sampled_pids = np.squeeze(sampled_pids, axis=1)
    else:
        # if it's not uniform sampling or weighted sampling
        print("Sampling mode error!!")
        exit()
    return sampled_pids


def get_gt_mesh_from_data(sample, mesh_pose_dict=None):
    # running directory is experiment directory
    # not real code directory
    root_dir = os.path.abspath(
        os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)
    )
    if mesh_pose_dict is None:
        start_mesh_pos_dict = np.load(sample["data_path"][0], allow_pickle=True)[
            "start_mesh_pose_dict"
        ].item()
    else:
        start_mesh_pos_dict = mesh_pose_dict

    global_trans = np.eye(4)
    global_trans[:3, 3] -= sample["center"][0].cpu().numpy()
    global_trans[:3] /= sample["scale"].item()

    gt_scene = trimesh.Scene()
    gt_mesh_dict = {0: [], 1: None}
    for k, v in start_mesh_pos_dict.items():
        link_scene = trimesh.Scene()
        for mesh_path, scale, pose in v:
            if mesh_path.startswith("#"):  # primitive
                mesh = trimesh.creation.box(extents=scale, transform=pose)
                mesh.apply_transform(global_trans)
            else:
                mesh = trimesh.load(os.path.join(root_dir, mesh_path))
                mesh.apply_scale(scale)
                mesh.apply_transform(pose)
                mesh.apply_transform(global_trans)
            gt_scene.add_geometry(mesh)
            link_scene.add_geometry(mesh)
        if float(k.split("_")[1]) == sample["joint_index"].item():
            gt_mesh_dict[1] = as_mesh(link_scene)
        else:
            gt_mesh_dict[0].append(as_mesh(link_scene))
    gt_mesh_dict[0] = as_mesh(trimesh.Scene(gt_mesh_dict[0]))
    return gt_mesh_dict
