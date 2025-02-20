import os

import numpy as np
import trimesh
import json
import os.path as osp

# moving A->B
def move_towards(A, B, step = 0.1):
    A = np.array(A)
    B = np.array(B)
    B[2] = A[2]
    direction = B - A
    distance = np.linalg.norm(direction)
    direction_normalized = direction / distance

    if distance <= step:
        A = B
    else:
        A = A + direction_normalized * step

    return A

def is_inside(src_obj, tgt_obj):
    src_box = src_obj.bounding_box
    tgt_box = tgt_obj.bounding_box
    sl, sw, sh = src_box.extents
    tl, tw, th = tgt_box.extents
    move_to = tgt_box.centroid[2] - src_box.centroid[2] - sh / 2 - th / 2
    tgt_obj.apply_translation([0, 0, -move_to])

    # 计算目标对象在XY平面的四个角
    tgt_vertices = tgt_box.vertices
    tgt_xy_corners = tgt_vertices[:, :2]  # 仅取XY坐标

    # 计算源对象在XY平面的投影范围
    src_x_min = src_box.vertices[:, 0].min()
    src_x_max = src_box.vertices[:, 0].max()
    src_y_min = src_box.vertices[:, 1].min()
    src_y_max = src_box.vertices[:, 1].max()


    all_inside = np.all(
        (src_x_min <= tgt_xy_corners[:, 0]) & (tgt_xy_corners[:, 0] <= src_x_max) &
        (src_y_min <= tgt_xy_corners[:, 1]) & (tgt_xy_corners[:, 1] <= src_y_max)
    )
    return all_inside


def move_to_support(src_obj, tgt_obj, src_obj_id, tgt_obj_id):
    # if tgt_obj_id in ['52']:
    #     debug = 1
    # move to Z support
    sl, sw, sh = src_obj.bounding_box.extents
    tl, tw, th = tgt_obj.bounding_box.extents
    src_center = src_obj.bounding_box.centroid
    tgt_center = tgt_obj.bounding_box.centroid

    move_to = tgt_center[2] - src_center[2] - sh/2 - th/2
    tgt_obj.apply_translation([0,0,-move_to])
    assert tgt_obj_id not in ssg_refined_info[one_scan]
    ssg_refined_info[one_scan][tgt_obj_id] = np.array([0, 0, -move_to]).tolist()


    # move to XY support
    all_inside = is_inside(src_obj, tgt_obj)

    step = 10
    while not all_inside and step>0:
        src_box = src_obj.bounding_box
        tgt_box = tgt_obj.bounding_box
        src_center_XY = src_box.centroid
        tgt_center_XY = tgt_box.centroid
        tgt_center_update = move_towards(tgt_center_XY, src_center_XY)
        tgt_obj.apply_translation([tgt_center_update[0] - tgt_center_XY[0], tgt_center_update[1] - tgt_center_XY[1], 0])
        all_inside = is_inside(src_obj, tgt_obj)
        step -= 1
        ssg_refined_info[one_scan][tgt_obj_id][0] += tgt_center_update[0] - tgt_center_XY[0]
        ssg_refined_info[one_scan][tgt_obj_id][1] += tgt_center_update[1] - tgt_center_XY[1]

    return tgt_obj


def refine_pair(src_obj_id, tgt_obj_id, one_scan):
    src_obj_path = osp.join(obj_path_root, one_scan, src_obj_id, f'{src_obj_id}.obj')
    tgt_obj_path = osp.join(obj_path_root, one_scan, tgt_obj_id, f'{tgt_obj_id}.obj')

    if not (os.path.exists(src_obj_path) and os.path.exists(tgt_obj_path)): return

    src_obj = trimesh.load(src_obj_path)
    tgt_obj = trimesh.load(tgt_obj_path)


    tgt_obj = move_to_support(src_obj, tgt_obj, src_obj_id, tgt_obj_id)

    scene = trimesh.Scene()
    scene.add_geometry(src_obj)
    scene.add_geometry(tgt_obj)

    # scene.show()

    tgt_obj.export(tgt_obj_path)
    print ('export done ', tgt_obj_path)

    return

def refine_pair_on_floor( tgt_obj_id, one_scan):
    tgt_obj_path = osp.join(obj_path_root, one_scan, tgt_obj_id, f'{tgt_obj_id}.obj')

    if not os.path.exists(tgt_obj_path): return

    tgt_obj = trimesh.load(tgt_obj_path)

    tl, tw, th = tgt_obj.bounding_box.extents
    tgt_center = tgt_obj.bounding_box.centroid

    move_to = tgt_center[2] - th / 2
    tgt_obj.apply_translation([0, 0, -move_to])
    ssg_refined_info[one_scan][tgt_obj_id] = np.array([0, 0, -move_to]).tolist()

    # tgt_obj.show()

    tgt_obj.export(tgt_obj_path)
    print ('export done ', tgt_obj_path)

    return

ssg_path = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scannet_ssg/'
obj_path_root = '/mnt/fillipo/huangyue/recon_sim/7_anno_v3/'
scan_list_sim = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_sim.json', 'rb'))
scan_list_sim_id = [s.split('_')[0] for s in scan_list_sim]
scan_list = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_707.json', 'rb'))
inst_name_dir = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scan_data/instance_id_to_name/'
ssg_refined_info = {}

for scan_idx, one_scan in enumerate(scan_list_sim):
    # if one_scan not in ['scene0126_00']: continue
    print(scan_idx, one_scan)
    ssg = json.load(open(osp.join(ssg_path, one_scan, 'relationships.json'), 'rb'))
    ssg_v2 = json.load(open(osp.join(ssg_path, one_scan, 'relationships_v2.json'), 'rb'))
    inst_id_to_name = json.load(open(osp.join(inst_name_dir, f'{one_scan}.json'), 'r'))
    print(inst_id_to_name)
    floor_id = -1
    if one_scan not in ssg_refined_info:
        ssg_refined_info[one_scan] = {}

    hanging_obj_list = []

    for src_obj_id, tgt_obj_id, rels in ssg[one_scan]['relationships']:
        if rels in ['hanging on', 'hung on']:
            hanging_obj_list.append(src_obj_id)

    on_floor_objs = []
    for src_obj_id, tgt_obj_id, rels in ssg_v2['relationship']:
        if rels in ['support']:
            #if inst_id_to_name[int(src_obj_id)] == 'floor':
            if int(tgt_obj_id) in hanging_obj_list: continue
            if src_obj_id == 'planes':
                refine_pair_on_floor(str(tgt_obj_id), one_scan)
                on_floor_objs.append(int(tgt_obj_id))



    for src_obj_id, tgt_obj_id, rels in ssg_v2['relationship']:
        if rels in ['support']:
            # if inst_id_to_name[int(src_obj_id)] == 'floor': continue
            if src_obj_id == 'planes': continue
            if int(tgt_obj_id) in hanging_obj_list: continue
            if int(tgt_obj_id) in on_floor_objs: continue
            if inst_id_to_name[int(tgt_obj_id)] in ['chair', 'table', 'office chair']: continue
            if tgt_obj_id in ssg_refined_info[one_scan]: continue
            refine_pair(str(src_obj_id), str(tgt_obj_id), one_scan)

    obj_list = os.listdir(osp.join(obj_path_root, one_scan))
    # obj_list.sort()
    for obj in obj_list:
        if not('.glb' not in obj and '-' in obj): continue
        inst_id, _, _ = obj.split('-')
        refine_pair(inst_id, obj, one_scan)

json.dump(ssg_refined_info, open(os.path.join(obj_path_root, 'ssg_refined_info_v2.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=4)





