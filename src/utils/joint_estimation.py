import numpy as np
import torch

from src.models.modules.losses import rotation_matrix_from_axis


def norm(a, axis=-1):
    return np.sqrt(np.sum(a ** 2, axis=axis))


def eval_joint_r(pred, gt):
    pred_axis, pred_pivot_point, pred_angle = pred
    gt_axis, gt_pivot_point, gt_angle = gt
    # add ambiguity
    axis_ori_1 = np.arccos(np.dot(pred_axis, gt_axis))
    axis_ori_2 = np.arccos(np.dot(-pred_axis, gt_axis))
    if axis_ori_1 < axis_ori_2:
        axis_ori = axis_ori_1
        config = np.abs(gt_angle - pred_angle) * 180 / np.pi
    else:
        axis_ori = axis_ori_2
        config = np.abs(gt_angle + pred_angle) * 180 / np.pi

    pred_moment = np.cross(pred_pivot_point, pred_axis)
    gt_moment = np.cross(gt_pivot_point, gt_axis)

    if np.abs(axis_ori) < 0.001:
        dist = np.cross(gt_axis, (pred_moment - gt_moment))
        dist = norm(dist)
    else:
        dist = np.abs(gt_axis.dot(pred_moment) + pred_axis.dot(gt_moment)) / norm(
            np.cross(gt_axis, pred_axis)
        )
    axis_ori = axis_ori * 180 / np.pi

    return axis_ori, dist, config


def eval_joint_p(pred, gt):
    pred_axis, pred_d = pred
    gt_axis, gt_d = gt
    # add ambiguity
    axis_ori_1 = np.arccos(np.dot(pred_axis, gt_axis))
    axis_ori_2 = np.arccos(np.dot(-pred_axis, gt_axis))
    if axis_ori_1 < axis_ori_2:
        axis_ori = axis_ori_1
        config_err = np.abs(gt_d - pred_d)
    else:
        axis_ori = axis_ori_2
        config_err = np.abs(gt_d + pred_d)
    return axis_ori, config_err


def aggregate_dense_prediction_r(
    joint_axis, pivot_point, angle, method="mean", est_args={}
):
    if method == "mean":
        pivot_point_pred = np.median(pivot_point, axis=0)
        joint_axis_pred = np.median(joint_axis, axis=0)
        angle_pred = np.mean(angle)
    elif method == "RANSAC":
        fit_data = [
            (joint_axis[i], pivot_point[i], angle[i])
            for i in range(joint_axis.shape[0])
        ]
        model = RANSAC(fit_data, **est_args)
        joint_axis_pred = model.axis
        pivot_point_pred = model.pivot_point
        angle_pred = model.angle[0]
    return joint_axis_pred, pivot_point_pred, angle_pred


def RANSAC(data, n, k, t, d, fit_func):
    """
    data – A set of observations. (List_like)
    n – Minimum number of data points required to estimate model parameters.
    k – Maximum number of iterations allowed in the algorithm.
    t – Threshold value to determine data points that are fit well by model.
    d – Number of close data points required to assert that a model fits well to data.
    fit_func - fitting function return a model
    """
    iterations = 0
    best_fit = None
    best_err = 1e5
    num_data = len(data)

    while iterations < k:
        maybe_inliers_idx = np.random.choice(
            np.arange(num_data, dtype=int), n, replace=False
        )
        maybe_inliers = [data[x] for x in maybe_inliers_idx]
        maybe_model = fit_func(maybe_inliers)
        also_inliers = []
        for i, x in enumerate(data):
            if i in maybe_inliers_idx:
                continue

            err = maybe_model(x)
            if err < t:
                also_inliers.append(x)
        if len(also_inliers) > d:
            better_model = fit_func(also_inliers + maybe_inliers)
            this_err = 0.0
            for x in also_inliers + maybe_inliers:
                this_err += maybe_model(x)
            this_err /= len(also_inliers + maybe_inliers)
            if this_err < best_err:
                best_fit = better_model
                best_err = this_err
        iterations += 1
    return best_fit


class RevoluteJointModel:
    def __init__(self, axis, pivot_point, angle):
        self.axis = axis
        self.pivot_point = pivot_point
        self.angle = angle
        self.rot = rotation_matrix_from_axis(
            torch.from_numpy(axis).unsqueeze(0), torch.from_numpy(angle)
        )
        self.rot = self.rot.numpy()[0]

    def __call__(self, data):
        axis, pivot_point, angle = data
        if len(angle.shape) == 0:
            angle = np.array([angle])
        rot = rotation_matrix_from_axis(
            torch.from_numpy(axis).unsqueeze(0), torch.from_numpy(angle)
        )
        rot = rot.numpy()[0]
        rot_diff = np.eye(3) - rot.dot(self.rot.T)
        rot_diff = rot_diff.sum()
        _, dist, _ = eval_joint_r(
            (axis, pivot_point, angle),
            (self.axis, self.pivot_point, self.angle),
        )
        return rot_diff + dist * 1.0


def fit_revolut_joint(data):
    axis_list = []
    moment_list = []
    angle_list = []
    for axis, point, angle in data:
        moment = np.cross(point, axis)
        axis_list.append(axis)
        moment_list.append(moment)
        angle_list.append(angle)
    axes = np.stack(axis_list, axis=0)
    moments = np.stack(moment_list, axis=0)
    angles = np.stack(angle_list, axis=0)
    axes = np.mean(axes, axis=0)
    moments = np.mean(moments, axis=0)
    angles = np.mean(angles, axis=0)
    if len(angles.shape) == 0:
        angles = np.array([angles])
    return RevoluteJointModel(axes, np.cross(axes, moments), angles)
