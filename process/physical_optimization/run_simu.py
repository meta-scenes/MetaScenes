import os
import numpy as np
import bpy
import json
import os.path as osp
import random
import yaml
import sys
import math
sys.path.append('/home/huangyue/Mycodes/MetaScenes/process/physical_optimization')
from loading_object_utils import load_object
from joint_object_utils import get_joint_objs
from mcmc_utils import update_collision_loss, get_one_move, get_one_move_back
from add_physics_utils import create_rigidbody_constraint, add_rigid_body, add_one_obj_rigid_body
from layout_utils import heuristic_layout, roomplane_layout
from hanging_utils import refine_hanging, hanging_init
from utils import is_touching, is_inside, set_origin_obj, get_polygon_from_mesh, matrix_difference


def clear_all():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)

def remove_floating_meshes(hanging_obj_list):
    delete_cnt = 0
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    floating_meshes = []

    for obj1 in meshes:
        touching = False
        for obj2 in meshes:
            if obj1 != obj2 and is_touching(obj1, obj2):
                touching = True
                # print (obj1.name , 'is touching ... ', obj2.name )
                break
        if not touching:
            floating_meshes.append(obj1)

    for obj in floating_meshes:
        if obj.name in hanging_obj_list: continue
        print(f"Removed floating mesh: {obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)
        delete_cnt += 1

    return delete_cnt


def remove_outside(layout_type):
    delete_cnt = 0
    walls = [obj for obj in bpy.data.objects if (obj.type == 'MESH' and 'geometry_' in obj.name)]
    polygon = get_polygon_from_mesh(bpy.data.objects.get('planes'), layout_type)

    for obj in bpy.data.objects:
        if obj in walls: continue
        if obj.name == 'planes': continue
        if obj.type != 'MESH': continue

        if obj.location.z - obj.dimensions.z / 2 < -0.1:
            print('Removing down...', obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
            delete_cnt += 1
            continue

        if not is_inside(polygon, obj):
            print('Removing outside...', obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
            delete_cnt += 1
            continue

    return delete_cnt


def run_simulation_and_get_loss(frame=250):
    """
    Runs a physics simulation in Blender and calculates a loss function based on position and rotation changes.

    Parameters:
    - frame (int): The total number of frames to run the simulation.

    Returns:
    - total_loss (float): The computed loss based on positional and rotational differences.
    """

    # Set start and end frames for the simulation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame

    # Set start frame in Blender
    bpy.context.scene.frame_set(1)

    # Store initial transformation matrices of all mesh objects
    initial_matrices = {
        obj.name: obj.matrix_world.copy()
        for obj in bpy.context.scene.objects if obj.type == 'MESH'
    }


    print('=== Begin simulate ===')
    while bpy.context.scene.frame_current != frame:
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
        bpy.context.view_layer.update()
    print('=== End simulate ===')

    bpy.context.scene.frame_set(frame)  # Reset scene frame to the last frame of the simulation

    # Store final transformation matrices of all mesh objects after simulation
    final_matrices = {
        obj.name: obj.matrix_world.copy()
        for obj in bpy.context.scene.objects if obj.type == 'MESH'
    }

    # Compute total loss based on position and rotation differences
    total_loss = sum(
        sum(matrix_difference(initial_matrices[obj_name], final_matrices[obj_name]))
        for obj_name in initial_matrices
    ) / len(initial_matrices)

    # Apply updated object positions
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.matrix_world = obj.matrix_world.copy()
            bpy.context.view_layer.update()

    bpy.context.scene.frame_set(1)  # 重置场景帧

    return total_loss


def mcmc_opti(src_obj=None, max_step=50, layout_type='heuristic', hanging_objs=None):
    """
    Performs Markov Chain Monte Carlo (MCMC) optimization to minimize collision loss
    by iteratively adjusting object positions.

    Parameters:
    - src_obj (bpy.types.Object, optional): The source object used for collision loss calculations.
    - max_step (int): Maximum number of optimization steps.
    - layout_type (str): Type of layout, used for polygon extraction.
    - hanging_objs (list of str): List of objects that should not be moved.

    Returns:
    - None
    """

    # Deselect all objects before optimization
    bpy.ops.object.select_all(action='DESELECT')

    # Initialize loss tracking
    losses = []
    obj_list, init_collision_loss = update_collision_loss(src_obj)

    # Initialize optimization step count
    step = 0

    # Get the polygon representation of the layout
    polygon = get_polygon_from_mesh(bpy.data.objects.get('planes'), layout_type)

    # Run optimization loop
    while init_collision_loss > 0 and step < max_step:
        step += 1
        print(step, init_collision_loss)

        # Iterate over all objects in the scene
        for one_obj_name in obj_list:
            if one_obj_name in hanging_objs:
                continue  # Skip objects that should not be moved

            obj = bpy.data.objects[one_obj_name]

            # Apply a random movement to the object
            move_direction = random.randint(0, 3)
            get_one_move(obj, move_direction)
            bpy.context.view_layer.update()

            # Ensure the object remains inside the polygon boundary
            if not is_inside(polygon, obj):
                get_one_move_back(obj, -move_direction)
                bpy.context.view_layer.update()
                continue  # Skip further evaluation if the object moves outside

            # Recalculate collision loss after movement
            _, collision_loss_new = update_collision_loss(src_obj)

            # If the movement increases collision loss, revert the change
            if collision_loss_new > init_collision_loss:
                get_one_move_back(obj, -move_direction)
                bpy.context.view_layer.update()
            else:
                # Update the current loss value if improvement is achieved
                init_collision_loss = collision_loss_new


def joint_object(ssg, texture_list):
    bpy.ops.object.select_all(action='DESELECT')
    valid_objs, joint_dict = get_joint_objs(ssg=ssg, texture_list=texture_list)

    bpy.ops.object.select_all(action='DESELECT')
    for src_id in joint_dict:
        if src_id not in bpy.data.objects: continue
        obj = bpy.data.objects[src_id]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        obj.select_set(False)

    return joint_dict


def hanging_object_optimization(one_scan, anno_info, inst_id_to_name):
    bpy.ops.object.select_all(action='DESELECT')
    hanging_objs = hanging_init(one_scan, anno_info, inst_id_to_name)
    for obj in bpy.data.objects:
        set_origin_obj(obj)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name in hanging_objs: continue
        if obj.location[2] - obj.dimensions[2] / 2 < 0:
            obj.location[2] = obj.dimensions[2] / 2

    return hanging_objs


def layout_construction(one_scan, layout_type, layout_path):

    bpy.ops.object.select_all(action='DESELECT')
    if layout_type == 'heuristic':
        layout_info = json.load(open(f'{layout_path}/{one_scan}.json'))
        heuristic_layout(layout_info)
        for obj in bpy.data.objects:
            set_origin_obj(obj)

    elif layout_type == 'model':
        roomplane_layout(one_scan, layout_path=layout_path, scale_rate=1)
        for obj in bpy.data.objects:
            set_origin_obj(obj)
        layout_offset_x = -bpy.data.objects.get('planes').location.x
        layout_offset_y = -bpy.data.objects.get('planes').location.y
        for obj in bpy.data.objects:
            if obj.name in ['planes'] or obj.name.startswith("geometry_"):
                obj.location.x += layout_offset_x
                obj.location.y += layout_offset_y

    else:
        assert False


def global_optimization(hanging_objs, layout_type):

    planes = bpy.data.objects['planes']

    bpy.ops.object.select_all(action='DESELECT')
    add_rigid_body()

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith("geometry_"): continue
        if obj.type != 'MESH': continue
        if obj == planes: continue
        if obj.name in hanging_objs: continue
        tgt_obj = bpy.data.objects.get(obj.name)
        src_obj = planes
        if tgt_obj is None or src_obj is None: continue
        add_one_obj_rigid_body(tgt_obj, 'ACTIVE')
        create_rigidbody_constraint(
            obj1=src_obj,
            obj2=tgt_obj,
            constraint_type='GENERIC',
            use_limit_lin=(
                [-0.5, 0.5],
                [-0.5, 0.5],
                [-1, 0]
            ),
            use_limit_ang=(
                [math.radians(-10), math.radians(10)],
                [math.radians(-10), math.radians(10)],
                [0,0]
            )
        )

    simu_loss1 = run_simulation_and_get_loss(250)
    simu_loss2 = run_simulation_and_get_loss(150)

    bpy.ops.object.select_all(action='DESELECT')
    delete_hanging = refine_hanging(hanging_objs, layout_type)
    simu_loss3 = run_simulation_and_get_loss(150)

    return simu_loss1, simu_loss2, simu_loss3, delete_hanging


def split_objs(joint_obj_list):
    """
    Splits objects in the Blender scene based on vertex groups and stores their transformation matrices.

    Parameters:
    - joint_obj_list (dict): Dictionary where keys are combined object names and values are lists of sub-object IDs.

    Returns:
    - physical_matrix_info (dict): Dictionary mapping object names to their transformation matrices.
    """

    # Iterate through all objects in the scene
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if obj.name not in joint_obj_list:
            continue

        combined_object = bpy.data.objects[obj.name]
        bpy.context.view_layer.objects.active = combined_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')

        # Process each sub-object in the joint object
        for one_obj_id in joint_obj_list[obj.name]:
            bpy.context.view_layer.update()
            group_name = f'vert_{one_obj_id}'

            # Check if the vertex group exists
            if group_name not in combined_object.vertex_groups:
                continue

            bpy.ops.object.vertex_group_set_active(group=group_name)
            bpy.ops.object.vertex_group_select()

            # Check if any vertices are selected
            bpy.ops.object.mode_set(mode='OBJECT')
            selected_verts = [v for v in combined_object.data.vertices if v.select]
            bpy.ops.object.mode_set(mode='EDIT')

            print(one_obj_id, len(selected_verts), len(combined_object.data.vertices))

            if not selected_verts:
                continue

            # Separate the selected vertices into a new object
            bpy.ops.mesh.separate(type='SELECTED')
            print('Separation done')

            # Deselect all vertices for the next iteration
            bpy.ops.mesh.select_all(action='DESELECT')

            # Rename the separated object to match its ID
            bpy.data.objects[f'{obj.name}.001'].name = one_obj_id

        bpy.ops.object.mode_set(mode='OBJECT')

    # Store transformation matrices of all separated objects
    physical_matrix_info = {}
    bpy.ops.object.select_all(action='DESELECT')

    for obj in bpy.data.objects:
        if obj.type == 'MESH' and 'plane' not in obj.name and 'geometry' not in obj.name:
            # Select and set the object's origin
            obj.select_set(True)
            set_origin_obj(obj)

            # Store the transformation matrix
            physical_matrix_info[obj.name] = np.array(obj.matrix_world).tolist()
            print(f'Saving {obj.name} transformation matrix')

    return physical_matrix_info


def run(layout_type, one_scan, joint_matrix_all, invalid_objs, dataset_path, ssg, anno_info, inst_id_to_name, layout_dir, global_opt_save_dir):

    # 0. loading objects
    objs_list, texture_list = load_object(one_scan, joint_matrix_all, invalid_objs, dataset_path)

    # 1. joint small objects
    joint_dict = joint_object(ssg, texture_list)

    # 2. hanging group refined
    hanging_objs = hanging_object_optimization(one_scan, anno_info, inst_id_to_name)

    # 3. layout
    layout_construction(one_scan, layout_type, layout_dir)


    # 4. mcmc init
    mcmc_opti(layout_type=layout_type, hanging_objs=hanging_objs)

    # 5. global simulation
    simu_loss1, simu_loss2, simu_loss3, delete_hanging = global_optimization(hanging_objs, layout_type)


    # 6 removing outside objects and floating objectsni
    bpy.ops.object.select_all(action='DESELECT')
    delete_overlay = remove_outside(layout_type)

    bpy.ops.object.select_all(action='DESELECT')
    delete_floating = remove_floating_meshes(hanging_objs)

    simu_loss4 = run_simulation_and_get_loss(250)
    simu_loss = simu_loss1 + simu_loss2 + simu_loss3 + simu_loss4



    # 7 split objects
    matrix_dict = split_objs(joint_dict)


    # 8 saving results
    results = {}
    results['simu_loss'] = simu_loss
    results['layout_type'] = layout_type
    results['delete_overlay'] = delete_overlay
    results['delete_floating'] = delete_floating
    results['delete_hanging'] = delete_hanging
    results['matrix'] = matrix_dict


    output_file = os.path.join(global_opt_save_dir, f'{one_scan}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print ('Saving done')


def load_config(config_path):
    """ Load configuration from a YAML file. """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # Load config file
    config = load_config("/home/huangyue/Mycodes/MetaScenes/process/physical_optimization/config.yaml")

    # Load parameters from config
    dataset_path = config["dataset_path"]
    ssg_path = config["ssg_save_dir"]
    local_opt_save_dir = config["local_opt_save_dir"]
    layout_model_dir = config["layout_model_dir"]
    layout_heuristic_dir = config["layout_heuristic_dir"]
    inst_name_dir = config["inst_name_dir"]
    global_opt_save_dir = config["global_opt_save_dir"]
    scan_list = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_r2.json', 'rb'))#config["scan_list"]
    layout_type = config["layout_type"]


    anno_info = json.load(open(config["anno_info_path"], "rb"))

    for scan_idx, one_scan in enumerate(scan_list):

        joint_matrix_all = json.load(open(osp.join(local_opt_save_dir, f"{one_scan}_simu_matrix_info_support.json"), "rb"))
        invalid_objs = json.load(open(osp.join(local_opt_save_dir, f"{one_scan}_invalid_objs.json"), "rb"))
        if os.path.exists(os.path.join(global_opt_save_dir, f'{one_scan}.json')):
            print ('skip | ', one_scan)
            continue

        if layout_type == "model":
            layout_dir = layout_model_dir
        elif layout_type == "heuristic":
            layout_dir = layout_heuristic_dir
        else:
            layout_dir = ""
            assert False, "Invalid layout type"

        if layout_type == "model" and not os.path.exists(osp.join(layout_dir, f"{one_scan}.json")):
            print("Model layout construct failed, switching to heuristic layout instead!")
            continue

        ssg_path_one_scan = osp.join(ssg_path, f"{one_scan}_relationships.json")
        if not os.path.exists(ssg_path_one_scan):
            print ('ssg not exists! | ', one_scan)
            continue
        ssg = json.load(open(ssg_path_one_scan, "rb"))
        inst_id_to_name = json.load(open(osp.join(inst_name_dir, f"{one_scan}.json"), "r"))

        clear_all()
        run(layout_type=layout_type,
            one_scan = one_scan,
            joint_matrix_all = joint_matrix_all,
            invalid_objs = invalid_objs,
            dataset_path = dataset_path,
            ssg = ssg,
            anno_info =anno_info,
            inst_id_to_name = inst_id_to_name,
            layout_dir = layout_dir,
            global_opt_save_dir =  global_opt_save_dir
            )


if __name__ == '__main__':
    main()

