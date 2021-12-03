import os

# os.environ["PYOPENGL_PLATFORM"] = "egl"
import subprocess

import matplotlib.pylab as plt
import numpy as np
import pyrender
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R


#########
# Render
#########
def get_camera_pose(radius, center=np.zeros(3), ax=0, ay=0, az=0):
    rotation = R.from_euler("xyz", (ax, ay, az)).as_matrix()
    vec = np.array([0, 0, radius])
    translation = rotation.dot(vec) + center
    camera_pose = np.zeros((4, 4))
    camera_pose[3, 3] = 1
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = translation
    return camera_pose


def render_mesh(mesh, camera, light, camera_pose, light_pose, renderer):
    r_scene = pyrender.Scene()
    o_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    r_scene.add(o_mesh)
    r_scene.add(camera, name="camera", pose=camera_pose)
    r_scene.add(light, name="light", pose=light_pose)
    color_img, _ = renderer.render(r_scene)
    return Image.fromarray(color_img)


#########
# Plot
#########


def plot_3d_point_cloud(
    x,
    y,
    z,
    show=True,
    show_axis=True,
    in_u_sphere=False,
    marker=".",
    s=8,
    alpha=0.8,
    figsize=(5, 5),
    elev=10,
    azim=240,
    axis=None,
    title=None,
    lim=None,
    *args,
    **kwargs
):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if lim:
        ax.set_xlim3d(*lim[0])
        ax.set_ylim3d(*lim[1])
        ax.set_zlim3d(*lim[2])
    elif in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        lim = (
            min(np.min(x), np.min(y), np.min(z)),
            max(np.max(x), np.max(y), np.max(z)),
        )
        ax.set_xlim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_ylim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_zlim(1.3 * lim[0], 1.3 * lim[1])
        plt.tight_layout()

    if not show_axis:
        plt.axis("off")

    if show:
        plt.show()

    return fig


def plot_3d_point_cloud_dict(name_dict, lim, size=2, **kwargs):
    num_plots = len(name_dict)
    fig = plt.figure(figsize=(size * num_plots, size))
    ax = {}
    for i, (k, v) in enumerate(name_dict.items()):
        ax[k] = fig.add_subplot(1, num_plots, i + 1, projection="3d")
        plot_3d_point_cloud(v[0], v[1], v[2], axis=ax[k], show=False, lim=lim, **kwargs)
        ax[k].set_title(k)
    plt.tight_layout()
    return fig


def visualize_pc_screw(pc_start, pc_end, screw_axis, screw_moment):
    bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
    bound_min = np.minimum(pc_start.min(0), pc_end.min(0))

    screw_point = np.cross(screw_axis, screw_moment)
    t_min = (bound_min - screw_point) / screw_axis
    t_max = (bound_max - screw_point) / screw_axis
    axis_index = np.argmin(np.abs(t_max - t_min))
    start_point = screw_point + screw_axis * t_min[axis_index]
    end_point = screw_point + screw_axis * t_max[axis_index]
    points = np.stack((start_point, end_point), axis=1)

    lim = [(bound_min.min(), bound_max.max())] * 3

    fig = plt.figure()
    ax_start = fig.add_subplot(121, projection="3d")
    ax_start.plot(*points, color="red")
    plot_3d_point_cloud(*pc_start.T, lim=lim, axis=ax_start)

    ax_end = fig.add_subplot(122, projection="3d")
    ax_end.plot(*points, color="red")
    plot_3d_point_cloud(*pc_end.T, lim=lim, axis=ax_end)

    return fig


#########
# Mesh
#########


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    The returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh
