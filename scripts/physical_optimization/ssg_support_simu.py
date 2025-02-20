import os

import numpy as np
import trimesh
import json
import os.path as osp
import bpy




def load_object(obj_path, inst_id, is_src = True):
    print (obj_path)
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[-1]
    obj.name = inst_id
    if is_src:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = 'PASSIVE'
    else:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = 'ACTIVE'
        bpy.context.object.rigid_body.collision_shape = 'CONVEX_HULL'
        # 设置刚体阻尼值
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
        obj.rotation_euler = (0.0, 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        obj.select_set(False)

def run_simulation_and_get_positions(frame=10):

    matrix_dict = {}

    """运行物理模拟并获取指定帧的物体位置"""
    # 设置开始和结束帧
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame

    # 设置起始帧
    bpy.context.scene.frame_set(1)

    # 确保所有物体处于非活动状态
    bpy.ops.object.select_all(action='DESELECT')

    print('=== Begin simulate ===')
    while bpy.context.scene.frame_current != frame:
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
        bpy.context.view_layer.update()
    print('=== End simulate ===')

    # 获取并更新每个物体在指定帧的位置
    # scene = trimesh.Scene()
    bpy.context.view_layer.update()
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj_path = os.path.join(obj_path_root, one_scan, obj.name, f'{obj.name}.obj')
            obj_mesh = trimesh.load(obj_path)

            pivot = np.array(obj_mesh.bounding_box.centroid)

            trans_matrix = np.array(obj.matrix_world)
            trans_matrix[:3,3] -= pivot

            matrix_dict[obj.name] = trans_matrix.tolist()

            print(obj, trans_matrix)
            # scene.add_geometry(obj_mesh)
    # scene.show()

    bpy.context.scene.frame_set(frame)

    return matrix_dict



def construct_support_dict(one_scan):
    support_dict = {}
    print(scan_idx, one_scan)
    ssg = json.load(open(osp.join(ssg_path, one_scan, 'relationships.json'), 'rb'))
    ssg_v2 = json.load(open(osp.join(ssg_path, one_scan, 'relationships_v2.json'), 'rb'))
    inst_id_to_name = json.load(open(osp.join(inst_name_dir, f'{one_scan}.json'), 'r'))
    hanging_obj_list = []

    for src_obj_id, tgt_obj_id, rels in ssg[one_scan]['relationships']:
        if rels in ['hanging on', 'hung on']:
            hanging_obj_list.append(src_obj_id)

    for src_obj_id, tgt_obj_id, rels in ssg_v2['relationship']:
        if int(tgt_obj_id) < 0: continue
        if rels in ['support']:
            # if inst_id_to_name[int(src_obj_id)] == 'floor': continue
            if src_obj_id == 'planes': continue
            if int(tgt_obj_id) in hanging_obj_list: continue
            if inst_id_to_name[int(tgt_obj_id)] in ['chair', 'table', 'office chair']: continue

            if str(src_obj_id) not in support_dict:
                support_dict[str(src_obj_id)] = []
            support_dict[str(src_obj_id)].append(str(tgt_obj_id))

    obj_list = os.listdir(osp.join(obj_path_root, one_scan))
    obj_list.sort()
    for obj in obj_list:
        if '-' not in obj: continue
        if '_' in obj: continue
        inst_id, _, _ = obj.split('-')
        if inst_id not in support_dict:
            support_dict[inst_id] = []
        support_dict[inst_id].append(obj)

    return support_dict

def clear_all():

    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)


ssg_path = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scannet_ssg/'
obj_path_root = '/mnt/fillipo/huangyue/recon_sim/7_anno_v3/'
scan_list_sim = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_sim.json', 'rb'))
scan_list_sim_id = [s.split('_')[0] for s in scan_list_sim]
scan_list = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_707.json', 'rb'))
inst_name_dir = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scan_data/instance_id_to_name/'



simu_matrix_info = {}

for scan_idx, one_scan in enumerate(scan_list_sim):

    # if one_scan not in [ 'scene0126_00']: continue

    simu_matrix_info[one_scan] = {}
    support_dict = construct_support_dict(one_scan)

    for one_group in support_dict:
        clear_all()
        tgt_obj_list = support_dict[one_group]

        src_obj_path = os.path.join(obj_path_root, one_scan, one_group, f'{one_group}.obj')
        if not os.path.exists(src_obj_path):
            print ('WRONG ', src_obj_path)
            continue
        load_object(src_obj_path, one_group)


        for tgt_inst_id in tgt_obj_list:
            tgt_obj_path = os.path.join(obj_path_root, one_scan, tgt_inst_id, f'{tgt_inst_id}.obj')
            if not os.path.exists(tgt_obj_path): continue
            load_object(tgt_obj_path, tgt_inst_id, is_src=False)

        set_origin_obj()
        matrix_dict = run_simulation_and_get_positions(frame = 150)
        simu_matrix_info[one_scan][one_group] = matrix_dict

        print ('done')

json.dump(simu_matrix_info, open(os.path.join(obj_path_root, 'simu_matrix_info_v2.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=4)












