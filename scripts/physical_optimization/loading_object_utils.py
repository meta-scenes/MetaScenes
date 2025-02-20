import bpy
from mathutils import Vector, Matrix
import sys

sys.path.append('/home/huangyue/Mycodes/MetaScenes/scripts/physical_optimization')
from utils import set_origin_obj

def apply_one_obj(obj, trans_matrix):
    """
    Applies a transformation matrix to an object in Blender.

    Parameters:
    obj (bpy.types.Object): The object to transform.
    trans_matrix (list): A 4x4 transformation matrix.
    """
    loc, rot, scale = Matrix(trans_matrix).decompose()
    obj.location = loc
    obj.rotation_euler = rot.to_euler('XYZ')
    obj.scale = scale
    bpy.context.view_layer.update()

def load_one_obj(obj_path, inst_id, choose, matrix):
    """
    Loads a 3D object into Blender and applies transformations.

    Parameters:
    obj_path (str): Path to the 3D object file.
    inst_id (str): Instance identifier for the object.
    choose (str): Specifies whether to apply texture.
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

    # Remove unrealistic objects based on size constraints
    if obj.dimensions[2] > 5 or obj.dimensions[0] > 20 or obj.dimensions[1] > 7:
        bpy.data.objects.remove(obj, do_unlink=True)
        return is_texture

    if matrix:
        apply_one_obj(obj, matrix)

    set_origin_obj(obj)

    if choose in {'1', '3'}:
        is_texture = True
        mat = obj.data.materials[0] if obj.data.materials else bpy.data.materials.new(name=f"Material_{obj.name}")
        obj.data.materials.append(mat) if not obj.data.materials else None

        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")

        if bsdf:
            attribute_node = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")
            attribute_node.location = (-400, 200)
            mat.node_tree.links.new(bsdf.inputs['Base Color'], attribute_node.outputs['Color'])

    return is_texture

def load_object(one_scan, anno_info, inst_id_to_name, joint_matrix_all):
    """
    Loads objects from a given scene and applies transformations.

    Parameters:
    one_scan (str): Scene identifier.
    anno_info (dict): Annotation information containing object metadata.
    inst_id_to_name (dict): Mapping of instance IDs to object names.
    joint_matrix_all (dict): Precomputed transformation matrices.

    Returns:
    tuple: (List of loaded object IDs, List of objects with special textures)
    """
    objs_list = []
    special_texture = []
    insts_info = anno_info[one_scan]

    for inst_id, obj_info in insts_info.items():
        if '.' in inst_id or 'door' in inst_id_to_name[int(inst_id)]:
            continue

        obj_path = (joint_matrix_all[one_scan][inst_id]['mesh_path']
                    if inst_id in joint_matrix_all[one_scan] and joint_matrix_all[one_scan][inst_id]
                    else obj_info['mesh_path'])

        matrix = None if inst_id in joint_matrix_all[one_scan] and joint_matrix_all[one_scan][inst_id] else obj_info['matrix']

        is_texture = load_one_obj(obj_path, inst_id, obj_info['choose'], matrix)
        objs_list.append(inst_id)

        if is_texture:
            special_texture.append(inst_id)

    return objs_list, special_texture
