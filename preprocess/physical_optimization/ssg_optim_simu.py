import os
import numpy as np
import json
import os.path as osp
import bpy
from mathutils import Matrix
import yaml

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


def run_simulation_and_get_positions(frame=10):
    """
    Run the physics simulation and capture object transformations at the target frame.
    """

    matrix_dict = {}

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame

    bpy.context.scene.frame_set(1)
    bpy.ops.object.select_all(action='DESELECT')

    print('=== Begin simulate ===')
    while bpy.context.scene.frame_current != frame:
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
        bpy.context.view_layer.update()
    print('=== End simulate ===')

    bpy.context.view_layer.update()
    for obj in bpy.context.scene.objects:
        if obj.name == 'Cube': continue
        if obj.type == 'MESH':
            trans_matrix = np.array(obj.matrix_world)
            matrix_dict[obj.name] = trans_matrix.tolist()

            new_matrix_world = obj.matrix_world.copy()
            obj.matrix_world = new_matrix_world
            bpy.context.view_layer.update()

    bpy.context.scene.frame_set(1)

    return matrix_dict


def construct_support_dict(one_scan, ssg_path, inst_name_dir):
    support_dict = {}
    ssg_path_one_scan = osp.join(ssg_path, f'{one_scan}_relationships.json')
    if not os.path.exists(ssg_path_one_scan): return {}
    ssg = json.load(open(ssg_path_one_scan, 'rb'))
    inst_id_to_name = json.load(open(osp.join(inst_name_dir, f'{one_scan}.json'), 'r'))

    for src_obj_id, tgt_obj_id, rels in ssg['relationship']:

        if rels in ['support']:
            if src_obj_id == 'planes': continue
            if '-' not in tgt_obj_id and inst_id_to_name[int(tgt_obj_id)] in ['chair', 'table', 'office chair']: continue
            if str(src_obj_id) not in support_dict: support_dict[str(src_obj_id)] = []
            support_dict[str(src_obj_id)].append(str(tgt_obj_id))
        elif rels in ['embed']:
            if str(src_obj_id) not in support_dict: support_dict[str(src_obj_id)] = []
            support_dict[str(src_obj_id)].append(str(tgt_obj_id))

        else:
            continue


    return support_dict


def clear_all():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)


def create_init_cube():
    """
    Create a ground plane cube for physics simulation.
    """
    bpy.ops.mesh.primitive_cube_add(size=1)
    cube = bpy.context.active_object
    cube.scale = (10.0, 10.0, 0.1)

    bpy.context.view_layer.objects.active = cube
    bpy.ops.rigidbody.object_add()
    cube.rigid_body.type = 'PASSIVE'
    return cube



def visualize_simulation_results(simu_matrix_info_path, dataset_path):
    """
    Load objects and apply transformation matrices from simulation results.
    This allows visualization of the final object positions in Blender.
    """
    clear_all()

    with open(simu_matrix_info_path, 'r', encoding='utf-8') as f:
        simu_matrix_info = json.load(f)

    for one_scan, groups in simu_matrix_info.items():

        for one_group, matrix_dict in groups.items():
            obj_path = os.path.join(dataset_path, one_scan, one_group, f'{one_group}.obj')
            load_object(obj_path, one_group, is_src=True)
            set_origin_obj()
            if one_group in matrix_dict:
                obj = bpy.data.objects[one_group]
                obj.matrix_world = Matrix(np.array(matrix_dict[one_group]))


            for tgt_inst_id, matrix in matrix_dict.items():

                if tgt_inst_id == one_group:
                    continue
                tgt_obj_path = os.path.join(dataset_path, one_scan, tgt_inst_id, f'{tgt_inst_id}.obj')
                load_object(tgt_obj_path, tgt_inst_id, is_src=False)
                set_origin_obj()
                obj = bpy.data.objects[tgt_inst_id]
                obj.matrix_world = Matrix(np.array(matrix))

    print("Visualization complete. Check Blender scene for final object placements.")


def run_simulation(config):
    """ Run the full simulation process """

    ssg_path = config["ssg_save_dir"]
    dataset_path = config["dataset_path"]
    local_opt_save_dir = config["local_opt_save_dir"]
    inst_name_dir = config["inst_name_dir"]
    scan_list = config["scan_list"]
    simulation_frames = config["simulation_frames"]
    print (scan_list)

    for one_scan in scan_list:

        simu_matrix_info = {}
        invalid_obj_info = {}

        delect_objs = []
        support_dict = construct_support_dict(one_scan, ssg_path, inst_name_dir)
        print(f"Processing scan: {one_scan} | Support dict: {support_dict}")

        for one_group, tgt_obj_list in support_dict.items():
            clear_all()
            cube = create_init_cube()

            # Load source object and simulate
            src_obj_path = os.path.join(dataset_path, one_scan, one_group, f'{one_group}.obj')
            load_object(src_obj_path, one_group, is_src=False)

            set_origin_obj()
            src_obj = bpy.data.objects[one_group]
            cube.location[2] = src_obj.location[2] - src_obj.dimensions[2] / 2 - 0.1
            matrix_dict = run_simulation_and_get_positions(frame=simulation_frames)
            src_obj.rigid_body.type = 'PASSIVE'

            # Load target objects
            for tgt_inst_id in tgt_obj_list:
                tgt_obj_path = os.path.join(dataset_path, one_scan, tgt_inst_id, f'{tgt_inst_id}.obj')
                load_object(tgt_obj_path, tgt_inst_id, is_src=False)

            set_origin_obj()
            matrix_dict = run_simulation_and_get_positions(frame=simulation_frames)

            # Validate object placement

            for tgt_inst_id in tgt_obj_list:
                tgt_obj = bpy.data.objects.get(tgt_inst_id)
                if not tgt_obj:
                    continue

                    # Check height relationship
                tgt_bottom = tgt_obj.location[2] - tgt_obj.dimensions[2] / 2
                src_top = src_obj.location[2] + src_obj.dimensions[2] / 2

                if tgt_bottom - src_top < -0.1:
                    del matrix_dict[tgt_inst_id]
                    bpy.data.objects.remove(tgt_obj, do_unlink=True)
                    print(f'Deleting {tgt_inst_id} due to height mismatch')
                    delect_objs.append(tgt_inst_id)
                    continue

                # Check XY alignment
                if not (src_obj.location[0] - src_obj.dimensions[0] / 2 <= tgt_obj.location[0] <= src_obj.location[0] +
                        src_obj.dimensions[0] / 2 and
                        src_obj.location[1] - src_obj.dimensions[1] / 2 <= tgt_obj.location[1] <= src_obj.location[1] +
                        src_obj.dimensions[1] / 2):
                    del matrix_dict[tgt_inst_id]
                    bpy.data.objects.remove(tgt_obj, do_unlink=True)
                    print(f'Deleting {tgt_inst_id} due to XY misalignment')
                    delect_objs.append(tgt_inst_id)
                    continue

            simu_matrix_info[one_group] = matrix_dict
            print(f'Processing {one_scan} - {one_group} done')

        invalid_obj_info = delect_objs

        # Save results
        output_file = os.path.join(local_opt_save_dir, f'{one_scan}_simu_matrix_info_support.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simu_matrix_info, f, ensure_ascii=False, indent=4)


        output_file_invalid = os.path.join(local_opt_save_dir, f'{one_scan}_invalid_objs.json')
        with open(output_file_invalid, 'w', encoding='utf-8') as f:
            json.dump(invalid_obj_info, f, ensure_ascii=False, indent=4)

    # Visualize results
    # visualize_simulation_results(output_file, dataset_path)



def load_config(config_path='/home/huangyue/Mycodes/MetaScenes/process/physical_optimization/config.yaml'):
    """ Load YAML configuration file """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """ Main function to load config and run simulation """
    config = load_config()
    run_simulation(config)


if __name__ == "__main__":
    main()
