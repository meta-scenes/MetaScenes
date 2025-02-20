import os
import numpy as np
import bpy
import json
import os.path as osp
import random

import sys
sys.path.append('/home/huangyue/Mycodes/MetaScenes/scripts/physical_optimization')
from loading_object_utils import load_object
from joint_object_utils import select_joint_groups, get_joint_objs
from mcmc_utils import update_collision_loss, get_one_move, get_one_move_back
from add_physics_utils import create_rigidbody_constraint, add_rigid_body, add_one_obj_rigid_body
from layout_utils import heuristic_layout, roomplane_layout
from hanging_utils import refine_hanging, hanging_init_v2
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

    # 删除悬空的 Mesh 对象
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
    """运行物理模拟并计算损失函数"""
    # 设置开始和结束帧
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame

    # 设置起始帧
    bpy.context.scene.frame_set(1)

    # 记录初始位置和旋转矩阵
    initial_matrices = {}
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            initial_matrices[obj.name] = obj.matrix_world.copy()

    # 运行模拟
    print('=== Begin simulate ===')
    while bpy.context.scene.frame_current != frame:
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
        bpy.context.view_layer.update()
    print('=== End simulate ===')

    bpy.context.scene.frame_set(frame)  # 重置场景帧

    # 记录模拟后的位置和旋转矩阵
    final_matrices = {}
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            final_matrices[obj.name] = obj.matrix_world.copy()

    # 计算位置和旋转差异，并构建损失函数
    total_loss = 0
    for obj_name in initial_matrices:
        pos_diff, rot_diff = matrix_difference(initial_matrices[obj_name], final_matrices[obj_name])
        total_loss += pos_diff + rot_diff  # 可以根据需要调整这两个差异的权重

    total_loss = total_loss / len(initial_matrices)

    # apply object position
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # apply_one_obj(obj, obj.matrix_world)
            new_matrix_world = obj.matrix_world.copy()
            obj.matrix_world = new_matrix_world
            bpy.context.view_layer.update()

    bpy.context.scene.frame_set(1)  # 重置场景帧

    print(total_loss)
    return total_loss


def mcmc_opti(src_obj=None, max_step=50, layout_type='heuristic', hanging_objs=[]):

    bpy.ops.object.select_all(action='DESELECT')
    losses = []
    obj_list, init_collision_loss = update_collision_loss(src_obj)
    step = 0
    polygon = get_polygon_from_mesh(bpy.data.objects.get('planes'), layout_type)
    while init_collision_loss > 0 and step < max_step:
        step += 1
        print(step, init_collision_loss)
        for one_obj_name in obj_list:
            if one_obj_name in hanging_objs: continue
            obj = bpy.data.objects[one_obj_name]

            c = random.randint(0, 3)
            get_one_move(obj, c)
            bpy.context.view_layer.update()
            if not is_inside(polygon, obj):
                get_one_move_back(obj, -c)
                bpy.context.view_layer.update()
                continue

            _, collision_loss_new = update_collision_loss(src_obj)

            if collision_loss_new > init_collision_loss:
                get_one_move_back(obj, -c)
                bpy.context.view_layer.update()
            else:
                init_collision_loss = collision_loss_new


def joint_object(texture_list):
    bpy.ops.object.select_all(action='DESELECT')
    max_obj_group = select_joint_groups(support_info[one_scan])
    valid_objs, joint_dict = get_joint_objs(max_obj_group=max_obj_group, ssg=ssg, ssg_sm=ssg_sm, texture_list=texture_list)
    print (joint_dict)
    bpy.ops.object.select_all(action='DESELECT')
    for src_id in joint_dict:
        if src_id not in bpy.data.objects: continue
        obj = bpy.data.objects[src_id]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        obj.select_set(False)


def hanging_object_optimization(one_scan, anno_info, inst_id_to_name):
    bpy.ops.object.select_all(action='DESELECT')
    hanging_objs = hanging_init_v2(one_scan, anno_info, inst_id_to_name)
    for obj in bpy.data.objects:
        set_origin_obj(obj)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name in hanging_objs: continue
        if obj.location[2] - obj.dimensions[2] / 2 < 0:
            obj.location[2] = obj.dimensions[2] / 2

    return hanging_objs


def layout_construction(one_scan, layout_type):

    bpy.ops.object.select_all(action='DESELECT')
    if layout_type == 'heuristic':
        layout_info = json.load(open(f'/home/huangyue/Mycodes/MetaScenes/scripts/layout_estimation/heuristic_layout/{one_scan}.json'))
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
            use_limit_lin=(  # 使用元组传递线性限制
                [-0.5, 0.5],  # X轴限制
                [-0.5, 0.5],  # Y轴限制
                [-1, 0]  # Z轴限制
            ),
            use_limit_ang=(  # 使用元组传递角度限制
                [0, 0],  # X轴角度限制
                [0, 0],  # Y轴角度限制
                [0, 0]  # Z轴角度限制
            )
        )

    simu_loss1 = run_simulation_and_get_loss(250)
    simu_loss2 = run_simulation_and_get_loss(150)

    bpy.ops.object.select_all(action='DESELECT')
    delete_hanging = refine_hanging(hanging_objs, layout_type)
    simu_loss3 = run_simulation_and_get_loss(150)

    return simu_loss1, simu_loss2, simu_loss3, delete_hanging


def run(layout_type):

    # 0. loading objects
    objs_list, texture_list = load_object(one_scan, anno_info, inst_id_to_name, joint_matrix_all)

    # 1. joint small objects
    joint_object(texture_list)

    assert False

    # 2. hanging group refined
    hanging_objs = hanging_object_optimization(one_scan, anno_info, inst_id_to_name)

    # 3. layout
    layout_construction(one_scan, layout_type)

    # 4. mcmc init
    mcmc_opti(layout_type=layout_type, hanging_objs=hanging_objs)

    # 5. global simulation
    simu_loss1, simu_loss2, simu_loss3, delete_hanging = global_optimization(hanging_objs, layout_type)



    # 7 removing outside objects and floating objects
    bpy.ops.object.select_all(action='DESELECT')
    delete_overlay = remove_outside(layout_type)

    # 6.2 floating
    bpy.ops.object.select_all(action='DESELECT')
    delete_floating = remove_floating_meshes(hanging_objs)

    simu_loss4 = run_simulation_and_get_loss(250)
    simu_loss = simu_loss1 + simu_loss2 + simu_loss3 + simu_loss4

    print(joint_dict)


    return simu_loss, delete_overlay + delete_floating + delete_hanging


if __name__ == '__main__':

    ssg_path = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scannet_ssg/'
    inst_name_dir = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scan_data/instance_id_to_name/'
    layout_path = '/mnt/fillipo/huangyue/recon_sim/3_recon/Layout/roomFormer_planes/'
    obj_root = '/mnt/fillipo/huangyue/recon_sim/7_anno_v3/'
    support_info = json.load(open('/mnt/fillipo/huangyue/recon_sim/7_anno_v3/metadata_support_yandan_v5.json', 'rb'))
    anno_info = json.load(open('/mnt/fillipo/huangyue/recon_sim/7_anno_v2/anno_info_ranking_v2.json', 'rb'))
    anno_info_sm = json.load(open('/mnt/fillipo/huangyue/recon_sim/7_anno_v2/anno_info_ranking_v3_sm.json', 'rb'))
    joint_matrix_all = json.load(open('/mnt/fillipo/huangyue/recon_sim/7_anno_v3/metadata_yandan_v5.json', 'rb'))

    scan_list_sim = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_sim.json', 'rb'))
    scan_list_sim_id = [s.split('_')[0] for s in scan_list_sim]
    scan_list = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_707.json', 'rb'))



    for scan_idx, one_scan in enumerate(scan_list):
        scan_id = one_scan.split('_')[0]
        if scan_id in scan_list_sim_id: continue
        # if one_scan in loss_stage: continue
        if one_scan not in ['scene0006_00']: continue
        if not os.path.exists(f'/mnt/fillipo/huangyue/recon_sim/3_recon/Layout/roomFormer_planes/{one_scan}.json'): continue

        # if one_scan in loss_stage: continue

        print(scan_idx, one_scan)
        ssg = json.load(open(osp.join(ssg_path, one_scan, 'relationships_v3.json'), 'rb'))
        if not os.path.exists(osp.join(ssg_path, one_scan, 'relationships_v4_sm.json')): continue
        ssg_sm = json.load(open(osp.join(ssg_path, one_scan, 'relationships_v4_sm.json'), 'rb'))
        inst_id_to_name = json.load(open(osp.join(inst_name_dir, f'{one_scan}.json'), 'r'))
        print(inst_id_to_name)


        clear_all()

        simu_loss, delete_objs = run(layout_type='heuristic')


        #
        # bpy.ops.object.select_by_type(type='MESH')
        # bpy.ops.export_scene.gltf(filepath = scene_path, use_selection=True)
        # print ('Export done')
        #
        #
        # json.dump(loss_stage, open(os.path.join(json_save_path), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


