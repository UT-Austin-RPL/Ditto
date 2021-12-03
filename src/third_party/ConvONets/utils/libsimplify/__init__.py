import trimesh

from .simplify_mesh import mesh_simplify


def simplify_mesh(mesh, f_target=10000, agressiveness=7.0):
    vertices = mesh.vertices
    faces = mesh.faces

    vertices, faces = mesh_simplify(vertices, faces, f_target, agressiveness)

    mesh_simplified = trimesh.Trimesh(vertices, faces, process=False)

    return mesh_simplified
