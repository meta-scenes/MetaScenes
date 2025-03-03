import bpy
import json
import os
import numpy as np
from mathutils import Matrix


def clear_all():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)

def set_origin_obj():
    for obj in bpy.data.objects:
        obj.select_set(True)
        mat = obj.matrix_world
        center = np.mean(obj.bound_box, axis=0)  # local
        center = center.reshape((3, 1))
        center_world = (np.array(mat)[:3, :3] @ center + np.array(mat)[:3, [3]]).reshape(-1)
        # set origin
        bpy.context.scene.cursor.location = center_world
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        if abs(obj.rotation_euler[0]) < np.pi / 2 / 3:
            obj.rotation_euler[0] = 0
        if abs(obj.rotation_euler[1]) < np.pi / 2 / 3:
            obj.rotation_euler[1] = 0
        # bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        obj.select_set(False)


def load_object(obj_path, inst_id, is_src=True):
    """
    Load an object into the Blender scene and apply necessary transformations.
    """
    print(f"Loading object: {obj_path}, Instance ID: {inst_id}")

    if obj_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
    elif obj_path.endswith('.ply'):
        bpy.ops.wm.ply_import(filepath=obj_path)
    elif obj_path.endswith('.glb'):
        print(f"Importing GLB: {obj_path}")
        bpy.ops.import_scene.gltf(filepath=obj_path)

        bpy.data.objects.remove(bpy.data.objects["world"], do_unlink=True)
        obj = bpy.context.selected_objects[-1]
        obj.name = inst_id

        bpy.data.objects[inst_id].rotation_quaternion[0] = -0.707
        bpy.data.objects[inst_id].rotation_quaternion[1] = 0.707

        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    obj = bpy.context.selected_objects[-1]
    obj.name = inst_id

    if obj.data.users > 1:
        print(f"Duplicating mesh for {obj.name}")
        new_mesh = obj.data.copy()
        obj.data = new_mesh

    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = 'PASSIVE' if is_src else 'ACTIVE'

    if not is_src:
        bpy.context.object.rigid_body.collision_shape = 'BOX'

    obj.rigid_body.linear_damping = 1
    obj.rigid_body.angular_damping = 1

def visualize_simulation_results(simu_matrix_info_path, dataset_path):
    """
    Load objects and apply transformation matrices from simulation results.
    This allows visualization of the final object positions in Blender.
    """
    clear_all()

    with open(simu_matrix_info_path, 'r', encoding='utf-8') as f:
        simu_matrix_info = json.load(f)
    simu_matrix_info = simu_matrix_info['matrix']
    for inst_id in simu_matrix_info:

        obj_path = os.path.join(dataset_path, one_scan, inst_id, f'{inst_id}.obj')
        load_object(obj_path, inst_id, is_src=True)
        set_origin_obj()
        obj = bpy.data.objects[inst_id]
        obj.matrix_world = Matrix(np.array(simu_matrix_info[inst_id]))

    print("Visualization complete. Check Blender scene for final object placements.")


if __name__ == "__main__":
    one_scan = 'Scene ID to visualize'
    simu_matrix_info_path = f'path/to/simulation/matrix/JSON/file.'
    dataset_path = "/path/to/Annotation_scenes/"
    visualize_simulation_results(simu_matrix_info_path, dataset_path)