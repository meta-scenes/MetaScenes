import os

import bpy
from mathutils import Vector, Matrix
import sys

sys.path.append('/home/huangyue/Mycodes/MetaScenes/scripts/physical_optimization')
from utils import set_origin_obj



def load_one_obj(obj_path, inst_id, matrix= None):
    """
    Loads a 3D object into Blender and applies transformations.

    Parameters:
    obj_path (str): Path to the 3D object file.
    inst_id (str): Instance identifier for the object.
    matrix (list or None): Transformation matrix for the object.

    Returns:
    bool: True if the object has a special texture, False otherwise.
    """
    is_texture = False
    obj = None

    if obj_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
    elif obj_path.endswith('.ply'):
        bpy.ops.wm.ply_import(filepath=obj_path)
    elif obj_path.endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=obj_path)
        bpy.data.objects.remove(bpy.data.objects.get("world"), do_unlink=True)
    else:
        raise ValueError("Unsupported file format")

    obj = bpy.context.selected_objects[-1]
    obj.name = inst_id

    if obj_path.endswith('.glb'):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.rotation_quaternion[0] = -0.707
        obj.rotation_quaternion[1] = 0.707

    if obj.data.users > 1:
        obj.data = obj.data.copy()

    set_origin_obj(obj)

    # Remove unrealistic objects based on size constraints
    if obj.dimensions[2] > 5 or obj.dimensions[0] > 20 or obj.dimensions[1] > 7:
        bpy.data.objects.remove(obj, do_unlink=True)
        return is_texture

    if matrix:
        obj.matrix_world = Matrix(matrix)

    return is_texture

def quary_matrix_by_inst(joint_matrix_all, one_scan, inst_id):

    if inst_id in joint_matrix_all[one_scan]:
        return joint_matrix_all[one_scan][inst_id][inst_id]
    else:
        for src_inst_id in joint_matrix_all[one_scan]:
            if inst_id in joint_matrix_all[one_scan][src_inst_id]:
                return joint_matrix_all[one_scan][src_inst_id][inst_id]

    return  None


def load_object(one_scan, joint_matrix_all, invalid_objs, dataset_path):
    """
    Loads 3D objects from a specified scene and applies transformation matrices.

    Parameters:
    - one_scan (str): Scene identifier.
    - joint_matrix_all (dict): Dictionary of precomputed transformation matrices for objects.
    - invalid_objs (dict): Dictionary of invalid objects that should be skipped for this scene.
    - dataset_path (str): Path to the dataset containing 3D object files.

    Returns:
    - tuple: (List of loaded object instance IDs, List of objects with special textures)
    """

    objs_in_one_scan = os.listdir(os.path.join(dataset_path, one_scan))
    objs_in_one_scan.sort()

    objs_list = []
    special_texture = []

    for inst_id in objs_in_one_scan:
        if inst_id in invalid_objs[one_scan]:
            continue

        obj_path = os.path.join(dataset_path, one_scan, inst_id, f'{inst_id}.obj')
        matrix = quary_matrix_by_inst(joint_matrix_all, one_scan, inst_id)
        is_texture = load_one_obj(obj_path, inst_id, matrix)

        objs_list.append(inst_id)
        if is_texture:
            special_texture.append(inst_id)

    return objs_list, special_texture
