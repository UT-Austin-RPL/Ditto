#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree


def compute_trimesh_chamfer(
    gt_mesh, pred_mesh, offset, scale, num_mesh_samples=30000, verbose=False
):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_mesh: trimesh.base.Trimesh of ground truth mesh

    pred_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """
    if gt_mesh.vertices.shape[0] == 0 or pred_mesh.vertices.shape[0] == 0:
        return np.nan

    pred_points = trimesh.sample.sample_surface(pred_mesh, num_mesh_samples)[0]
    gt_points = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]

    gt_points = (gt_points - offset) / scale

    # one direction
    pred_points_kd_tree = KDTree(pred_points)
    one_distances, one_vertex_ids = pred_points_kd_tree.query(gt_points)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_points)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    if verbose:
        print(
            gt_to_pred_chamfer + pred_to_gt_chamfer,
            gt_to_pred_chamfer,
            pred_to_gt_chamfer,
        )
    return gt_to_pred_chamfer + pred_to_gt_chamfer
