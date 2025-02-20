import os
import random

import numpy as np
import trimesh
import json
import os.path as osp
import bpy


class BoundingBox:
    def __init__(self, cx, cy, cz, width, height, depth):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.width = width
        self.height = height
        self.depth = depth

    @property
    def x_min(self):
        return self.cx - self.width / 2

    @property
    def y_min(self):
        return self.cy - self.height / 2

    @property
    def z_min(self):
        return self.cz - self.depth / 2

    @property
    def x_max(self):
        return self.cx + self.width / 2

    @property
    def y_max(self):
        return self.cy + self.height / 2

    @property
    def z_max(self):
        return self.cz + self.depth / 2

    def volume(self):
        return self.width * self.height * self.depth

    def intersection_volume(self, other):
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        z_min = max(self.z_min, other.z_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        z_max = min(self.z_max, other.z_max)

        if x_min < x_max and y_min < y_max and z_min < z_max:
            return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        else:
            return 0

def compute_collision_loss(bboxes):
    total_volume = sum([bbox.volume() for bbox in bboxes])
    total_intersection_volume = 0

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            total_intersection_volume += bboxes[i].intersection_volume(bboxes[j])

    collision_loss = total_intersection_volume / total_volume
    return collision_loss


def load_object(obj_path, inst_id, is_src = True):
    print (obj_path)
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[-1]
    obj.name = inst_id
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


def get_one_move(obj, case, step = 0.05):

    if case == 1:
        obj.location.x += step
    elif case == 2:
        obj.location.x -= step
    elif case == 3:
        obj.location.y += step
    elif case == 4:
        obj.location.y -= step

    return 0

def get_one_move_back(obj, case, step = 0.05):

    if case == -1:
        obj.location.x -= step
    elif case == -2:
        obj.location.x += step
    elif case == -3:
        obj.location.y -= step
    elif case == -4:
        obj.location.y += step

    return 0


def update_collision_loss(src_obj):
    bboxes = []
    obj_list = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.name not in ['Cube', 'planes'] and obj!= src_obj and 'geometry' not in obj.name:

            bboxes.append(BoundingBox(obj.location.x, obj.location.y,obj.location.z, obj.dimensions[0], obj.dimensions[1], obj.dimensions[2] ))
            obj_list.append(obj.name)
    if len(bboxes) <= 1:
        return [], 0

    init_collision_loss = compute_collision_loss(bboxes)


    return obj_list, init_collision_loss


def mcmc_init(src_id, tgt_list):
    loc_dict = {}
    src_obj = bpy.data.objects[src_id]
    loc_dict[src_id] = src_obj.location.copy()
    for tgt_id in tgt_list:
        if tgt_id not in bpy.data.objects: continue
        tgt_obj = bpy.data.objects[tgt_id]
        loc_dict[tgt_id] = tgt_obj.location.copy()
        tgt_obj.location.z = src_obj.location.z + tgt_obj.dimensions[2]/2 + src_obj.dimensions[2]/2

    mcmc_opti(src_obj)

    for inst_id in loc_dict:
        obj = bpy.data.objects[inst_id]
        loc_new = obj.location.copy()
        move = loc_new - loc_dict[inst_id]
        loc_dict[inst_id] = np.array(move).tolist()
        print (inst_id, move)

    return loc_dict


def mcmc_opti(src_obj, max_step = 50):

    obj_list, init_collision_loss = update_collision_loss(src_obj)
    step = 0
    while init_collision_loss > 0.03 and step < max_step:
        step += 1
        for one_obj_name in obj_list:
            obj = bpy.data.objects[one_obj_name]

            c = random.randint(0,3)
            get_one_move(obj, c)
            if obj.location.x - obj.dimensions[0] / 2 < src_obj.location.x - src_obj.dimensions[0] / 2  or obj.location.x + obj.dimensions[0] / 2 > src_obj.location.x + src_obj.dimensions[0] / 2 or obj.location.y - obj.dimensions[1] / 2 < src_obj.location.y - src_obj.dimensions[1] / 2 or obj.location.y + obj.dimensions[1] / 2 > src_obj.location.y + src_obj.dimensions[1] / 2:
                get_one_move_back(obj, -c)
                continue

            _, collision_loss_new = update_collision_loss(src_obj)

            if collision_loss_new > init_collision_loss:
                get_one_move_back(obj, -c)
            else:
                init_collision_loss = collision_loss_new



def run_simulation_and_export(one_scan, frame=10):

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


    bpy.context.scene.frame_set(frame)


    # apply object position
    for obj in bpy.context.scene.objects:
       if obj.type == 'MESH' and obj.name != 'Cube':
           # print(obj.name, obj.matrix_world)
           new_matrix_world = obj.matrix_world.copy()
           obj.matrix_world = new_matrix_world
           bpy.context.view_layer.update()
           matrix_dict[obj.name] = np.round(np.array(obj.matrix_world.copy()),4).tolist()
           # print (obj.name , obj.matrix_world, matrix_dict[obj.name])

    bpy.context.scene.frame_set(1)

    return matrix_dict

def construct_support_dict(inst_id_to_name, support_dict_raw):
    support_dict = {}
    fix_dict = {}
    embed_dict = {}

    for src_id, tgt_id, rels in support_dict_raw['relationship']:
        if rels == 'support':
            if src_id not in support_dict:
                support_dict[src_id] = []
            support_dict[src_id].append(tgt_id)
        elif rels == 'embed':
            # todo sink
            if inst_id_to_name[int(tgt_id)] in ['sink']:
                if src_id not in fix_dict:
                    fix_dict[src_id] = []
                fix_dict[src_id].append(tgt_id)

            if src_id not in embed_dict:
                embed_dict[src_id] = []
            embed_dict[src_id].append(tgt_id)
        else:
            if src_id not in fix_dict:
                fix_dict[src_id] = []
            fix_dict[src_id].append(tgt_id)




    print (support_dict)
    for src_id in support_dict:
        if src_id == 'planes': continue
        for tgt_id in support_dict[src_id]:
            if tgt_id in support_dict:
                support_dict[src_id].extend(support_dict[tgt_id])
                support_dict[tgt_id] = []

    for src_id in list(support_dict.keys()):
        if support_dict[src_id] == []:
            del support_dict[src_id]


    return support_dict, fix_dict, embed_dict

def clear_all():

    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)


def run_simu_one_group(one_scan, one_group, one_dict, on_planes_objs, is_move, is_join):

    tgt_obj_list = one_dict[one_group]

    src_obj_path = os.path.join(obj_path_root, one_scan, one_group, f'{one_group}.obj')
    if not os.path.exists(src_obj_path):  return
    load_object(src_obj_path, one_group)

    for tgt_inst_id in tgt_obj_list:
        tgt_obj_path = os.path.join(obj_path_root, one_scan, tgt_inst_id, f'{tgt_inst_id}.obj')
        if not os.path.exists(tgt_obj_path): continue
        load_object(tgt_obj_path, tgt_inst_id, is_src=False)

    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD',
                                    location=(0, 0, -0.05), scale=(5, 5, 0.1))
    obj = bpy.data.objects["Cube"]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = 'PASSIVE'
    # 设置刚体阻尼值
    obj.rigid_body.linear_damping = 1
    obj.rigid_body.angular_damping = 1

    set_origin_obj()

    if is_join:
        for obj in bpy.data.objects:
            if obj.name == 'Cube':
                obj.select_set(False)
            else:
                obj.select_set(True)

        bpy.context.view_layer.objects.active = bpy.data.objects[one_group]
        bpy.ops.object.join()
        bpy.data.objects[one_group].select_set(False)
        set_origin_obj()


    if is_move:
        trans_dict = mcmc_init(one_group, tgt_obj_list)


    if one_group not in on_planes_objs:
        bpy.data.objects[one_group].rigid_body.type = 'PASSIVE'


    matrix_dict = run_simulation_and_export(one_scan, frame=250)


    return matrix_dict




def run_simu():
    simu_matrix_info = {}

    for scan_idx, one_scan in enumerate(scan_list_sim):

        # if one_scan not in ['scene0092_01']: continue
        inst_id_to_name = json.load(open(osp.join(inst_name_dir, f'{one_scan}.json'), 'r'))
        simu_matrix_info[one_scan] = {}
        simu_matrix_info[one_scan]['simu'] = {}
        simu_matrix_info[one_scan]['trans'] = {}
        support_dict_raw = json.load(open(os.path.join(ssg_path, one_scan, 'relationships_v3.json')))
        support_dict, fix_dict, embed_dict = construct_support_dict(inst_id_to_name, support_dict_raw)
        print('show me dict')
        print(support_dict, fix_dict, embed_dict)
        simu_matrix_info[one_scan]['support_dict'] = support_dict
        simu_matrix_info[one_scan]['fix_dict'] = fix_dict
        simu_matrix_info[one_scan]['embed_dict'] = embed_dict

        if 'planes' in support_dict:
            on_planes_objs = support_dict['planes']
        else:
            on_planes_objs = []

        for one_group in support_dict:
            clear_all()
            if one_group in ['planes']: continue
            matrix_dict = run_simu_one_group(one_scan, one_group, support_dict, on_planes_objs, is_move=True, is_join=False)
            simu_matrix_info[one_scan]['simu'].update(matrix_dict)

        for one_group in embed_dict:
            clear_all()
            matrix_dict = run_simu_one_group(one_scan, one_group, embed_dict, on_planes_objs, is_move=False, is_join=False)
            simu_matrix_info[one_scan]['simu'].update(matrix_dict)

        for one_group in fix_dict:
            clear_all()
            # if one_group in support_dict: continue # todo bug
            matrix_dict = run_simu_one_group(one_scan, one_group, fix_dict, on_planes_objs, is_move=False,  is_join=True)
            simu_matrix_info[one_scan]['simu'].update(matrix_dict)



    json.dump(simu_matrix_info, open(os.path.join(obj_path_root, 'joint_matrix.json'), 'w', encoding='utf-8'),
             ensure_ascii=False, indent=4)



def apply_one_obj(obj_path, trans_matrix, move_vector):

    mesh = trimesh.load(obj_path, force='mesh')

    translate_matrix = np.eye(4)
    translate_matrix[:3, 3] = np.round(np.array(move_vector),2)
    mesh.apply_transform(translate_matrix)

    pivot = np.array(mesh.bounding_box.centroid)

    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -pivot

    translation_back = np.eye(4)
    translation_back[:3, 3] = pivot

    final_matrix = translation_back @ trans_matrix @ translation_to_origin # @ translate_matrix

    return mesh.apply_transform(final_matrix)


def run_show():
    joint_matrix = json.load(open(os.path.join(obj_path_root, 'joint_matrix.json'),'rb'))

    for scan_idx, one_scan in enumerate(scan_list_sim):

        if one_scan not in ['scene0009_02']: continue

        support_dict_raw = json.load(open(os.path.join(ssg_path, one_scan, 'relationships_v3.json')))
        support_dict = construct_support_dict(support_dict_raw)
        print(support_dict)

        for one_group in support_dict:
            scene = trimesh.Scene()
            tgt_obj_list = support_dict[one_group]
            #if one_group == '3':continue
            src_obj_path = os.path.join(obj_path_root, one_scan, one_group, f'{one_group}.obj')
            if not os.path.exists(src_obj_path):
                print('WRONG ', src_obj_path)
                continue
            # scene.add_geometry(apply_one_obj(src_obj_path, joint_matrix[one_scan]['simu'][one_group], joint_matrix[one_scan]['trans'][one_group]))
            scene.add_geometry(trimesh.load(src_obj_path))


            for tgt_inst_id in tgt_obj_list:
                tgt_obj_path = os.path.join(obj_path_root, one_scan, tgt_inst_id, f'{tgt_inst_id}.obj')
                # if tgt_inst_id not in joint_matrix[one_scan]['simu']: continue
                # scene.add_geometry(apply_one_obj(tgt_obj_path, joint_matrix[one_scan]['simu'][tgt_inst_id], joint_matrix[one_scan]['trans'][tgt_inst_id]))
                scene.add_geometry(trimesh.load(tgt_obj_path))
            scene.show()



if __name__ == '__main__':
    ssg_path = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scannet_ssg/'
    obj_path_root = '/mnt/fillipo/huangyue/recon_sim/7_anno_v3/'
    scan_list_sim = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_sim.json', 'rb'))
    scan_list_sim_id = [s.split('_')[0] for s in scan_list_sim]
    scan_list = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_707.json', 'rb'))
    inst_name_dir = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scan_data/instance_id_to_name/'
    ssg_path = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scannet_ssg/'

    # run_show()
    run_simu()
















