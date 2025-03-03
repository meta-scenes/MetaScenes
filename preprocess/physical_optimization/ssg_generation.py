import os
import random

import trimesh
import json
import os.path as osp
import math
import numpy as np


def axis_align(obj):
    # obj[1], obj[2] = obj[2], obj[1]
    return obj

def distance_XY(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def is_support (obj1, obj2, padding = 0.04):
    # obj2 supports obj1
    # obj1 = trimesh.load(obj1_path)
    # obj2 = trimesh.load(obj2_path)
    # obj1.show()
    # obj2.show()
    obj1_loc = axis_align(obj1.bounding_box.centroid.copy())
    obj2_loc = axis_align(obj2.bounding_box.centroid.copy())
    obj1_size = axis_align(obj1.bounding_box.extents.copy())
    obj2_size = axis_align(obj2.bounding_box.extents.copy())

    case1 = abs((obj1_loc[2] - obj1_size[2]/2) - (obj2_loc[2] + obj2_size[2]/2)) <= padding # and ((obj1_loc[2] - obj1_size[2]/2) - (obj2_loc[2] + obj2_size[2]/2)) >= -obj2_size[2]/2
    case2 = (distance_XY(obj1_loc, obj2_loc) < max(obj2_size[0], obj2_size[1]))
    case3 = obj1_loc[0] <= obj2_loc[0]+obj2_size[0]/2 and obj1_loc[0] >= obj2_loc[0]-obj2_size[0]/2 and obj1_loc[1] <= obj2_loc[1]+obj2_size[1]/2 and obj1_loc[1] >= obj2_loc[1]-obj2_size[1]/2
    case4 = (obj1_size[0] * obj1_size[1] * obj1_size[2] ) <= (obj2_size[0] * obj2_size[1] * obj2_size[2] )
    #print ('case1', abs((obj1_loc[2] - obj1_size[2]/2) - (obj2_loc[2] + obj2_size[2]/2)), 'case2 ', distance_XY(obj1_loc, obj2_loc), max(obj2_size[0], obj2_size[1]))


    if case1 and case3 and case4:
        # print ('support')
        # obj1 = trimesh.load(obj1_path)
        # obj2 = trimesh.load(obj2_path)
        # scene = trimesh.Scene()
        # scene.add_geometry(obj1)
        # scene.add_geometry(obj2)
        # scene.show()
        return True

    return False


def is_embed (obj1, obj2):
    # obj1 in obj2
    # obj1 = trimesh.load(obj1_path)
    # obj2 = trimesh.load(obj2_path)
    # obj1.show()
    # obj2.show()
    obj1_loc = axis_align(obj1.bounding_box.centroid.copy())
    obj2_loc = axis_align(obj2.bounding_box.centroid.copy())
    obj1_size = axis_align(obj1.bounding_box.extents.copy())
    obj2_size = axis_align(obj2.bounding_box.extents.copy())

    case1 = ((obj1_loc[2] - obj1_size[2]/2) - (obj2_loc[2] + obj2_size[2]/2)) > -obj1_size[2] and ((obj1_loc[2] - obj1_size[2]/2) - (obj2_loc[2] + obj2_size[2]/2))<-obj1_size[2]/2
    case3 = obj1_loc[0] <= obj2_loc[0]+obj2_size[0]/2 and obj1_loc[0] >= obj2_loc[0]-obj2_size[0]/2 and obj1_loc[1] <= obj2_loc[1]+obj2_size[1]/2 and obj1_loc[1] >= obj2_loc[1]-obj2_size[1]/2
    case4 = (obj1_size[0] * obj1_size[1] * obj1_size[2] ) <= (obj2_size[0] * obj2_size[1] * obj2_size[2] )
    #print ('case1', abs((obj1_loc[2] - obj1_size[2]/2) - (obj2_loc[2] + obj2_size[2]/2)), 'case2 ', distance_XY(obj1_loc, obj2_loc), max(obj2_size[0], obj2_size[1]))


    if case1 and case3 and case4:
        # print ('embed')
        # obj1 = trimesh.load(obj1_path)
        # obj2 = trimesh.load(obj2_path)
        # scene = trimesh.Scene()
        # scene.add_geometry(obj1)
        # scene.add_geometry(obj2)
        # scene.show()
        return True

    return False

def is_support_floor (obj1, padding = 0.04):
    # obj2 supports obj1
    # obj1 = trimesh.load(obj1_path)
    # obj2 = trimesh.load(obj2_path)
    # obj1.show()
    # obj2.show()
    obj1_loc = axis_align(obj1.bounding_box.centroid.copy())
    obj1_size = axis_align(obj1.bounding_box.extents.copy())

    case1 = abs(obj1_loc[2] - obj1_size[2]/2) < 0.2

    if case1:
        # print ('floor support')
        # obj1.show()
        return True

    return False

def is_inside (obj1, obj2, padding = 0.04):
    # obj1 in obj2
    # obj1 = trimesh.load(obj1_path)
    # obj2 = trimesh.load(obj2_path)
    # obj1.show()
    # obj2.show()
    obj1_loc = axis_align(obj1.bounding_box.centroid.copy())
    obj2_loc = axis_align(obj2.bounding_box.centroid.copy())
    obj1_size = axis_align(obj1.bounding_box.extents.copy())
    obj2_size = axis_align(obj2.bounding_box.extents.copy())

    case1 = True
    for den in range(3):
        if not (obj1_loc[den] + obj1_size[den]/2 <= obj2_loc[den] + obj2_size[den]/2 and obj1_loc[den] - obj1_size[den]/2 >= obj2_loc[den] - obj2_size[den]/2):
            case1 = False

    case4 = (obj1_size[0] * obj1_size[1] * obj1_size[2] ) <= (obj2_size[0] * obj2_size[1] * obj2_size[2] )
    #print ('case1', abs((obj1_loc[2] - obj1_size[2]/2) - (obj2_loc[2] + obj2_size[2]/2)), 'case2 ', distance_XY(obj1_loc, obj2_loc), max(obj2_size[0], obj2_size[1]))

    if case1 and case4:
        # print ('Inside')
        # obj1 = trimesh.load(obj1_path)
        # obj2 = trimesh.load(obj2_path)
        # scene = trimesh.Scene()
        # scene.add_geometry(obj1)
        # scene.add_geometry(obj2)
        # scene.show()
        return True

    return False

def run(one_scan, ssg_path, inst_name_dir, DATASET_PATH):

    support_dict = {
        'scan': one_scan,
        'relationship': []
    }

    save_path = osp.join(ssg_path, f'{one_scan}_relationships.json')
    inst_id_to_name = json.load(open(osp.join(inst_name_dir, f'{one_scan}.json'), 'r'))

    objs_list = os.listdir(os.path.join(DATASET_PATH, one_scan))
    objs_list.sort()
    done_list = []

    planes_scene = trimesh.Scene()
    for inst_id in objs_list:
        if '-' in inst_id: continue

        if inst_id_to_name[int(inst_id)] in ['mirror', 'window', 'curtain']: continue

        obj2_path = os.path.join(DATASET_PATH, one_scan, inst_id, f'{inst_id}.obj')
        obj2 = trimesh.load(obj2_path, force='mesh')

        if is_support_floor(obj2):
            support_dict['relationship'].append(['planes', (inst_id.split('.')[0]), 'support'])
            planes_scene.add_geometry(obj2)

    # planes_scene.show()
    for i in range(len(objs_list)):

        if '-' in objs_list[i]: continue
        if inst_id_to_name[int(objs_list[i])] in ['mirror', 'window', 'whiteboard', 'picture', 'radiator',
                                                  'curtain']: continue

        obj1_path = os.path.join(DATASET_PATH, one_scan, objs_list[i], f'{objs_list[i]}.obj')
        obj1 = trimesh.load(obj1_path)


        for j in range(len(objs_list)):
            if i == j: continue
            if objs_list[j] in done_list: continue

            obj2_path = os.path.join(DATASET_PATH, one_scan, objs_list[j], f'{objs_list[j]}.obj')
            obj2 = trimesh.load(obj2_path, force='mesh')


            if 'geometry' in obj1_path or 'geometry' in obj2_path: continue

            # if 'planes.glb' in obj1_path or 'planes.glb' in obj2_path: continue
            ### for small object
            if '-' in objs_list[j]:
                src_inst = objs_list[j].split('-')[0]
                support_dict['relationship'].append(
                    [src_inst, (objs_list[j].split('.')[0]), 'support'])
                done_list.append(objs_list[j])
                continue

            # print (inst_id_to_name[int(objs_list[i])], inst_id_to_name[int(objs_list[j])])
            if is_support(obj1, obj2):
                support_dict['relationship'].append(
                    [(objs_list[j].split('.')[0]), (objs_list[i].split('.')[0]), 'support'])
                continue

            if is_inside(obj1, obj2):
                support_dict['relationship'].append(
                    [(objs_list[j].split('.')[0]), (objs_list[i].split('.')[0]), 'inside'])
                continue

            if is_embed(obj1, obj2):
                support_dict['relationship'].append(
                    [(objs_list[j].split('.')[0]), (objs_list[i].split('.')[0]), 'embed'])
                continue

            # if is_support(obj1, obj2) or is_support(obj2, obj1):
            #     print ('support')
            #     scene = trimesh.Scene()
            #     scene.add_geometry(obj1)
            #     scene.add_geometry(obj2)
            #     scene.show()
            #
            # if is_inside(obj1, obj2) or is_inside(obj2, obj1):
            #     print ('inside')
            #     scene = trimesh.Scene()
            #     scene.add_geometry(obj1)
            #     scene.add_geometry(obj2)
            #     scene.show()


    json.dump(support_dict, open(save_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)

    return save_path






def show(one_scan):
    ssg = json.load(open(osp.join(ssg_path,  f'{one_scan}_relationships.json'),'rb'))


    for item in ssg['relationship']:
        src, tgt, rels = item
        if src == 'planes': continue
        scene = trimesh.Scene()

        obj1_path = os.path.join(DATASET_PATH, one_scan, src, f'{src}.obj')
        obj1 = trimesh.load(obj1_path)


        obj2_path = os.path.join(DATASET_PATH, one_scan, tgt, f'{tgt}.obj')
        obj2 = trimesh.load(obj2_path)

        print (item)
        scene.add_geometry(obj1)
        scene.add_geometry(obj2)
        scene.show()


def main(scan_list, ssg_path, inst_name_dir, DATASET_PATH):
    save_path_list = []
    for one_scan in scan_list:

        if os.path.exists(f'/mnt/fillipo/huangyue/recon_sim/11_release/SSG/{one_scan}_relationships.json'):
            print ('skipping')
            continue
        print(f'Processing {one_scan}')
        save_path = run(one_scan, ssg_path, inst_name_dir, DATASET_PATH)

        save_path_list.append(save_path)

    return save_path_list







if __name__ == '__main__':

    scan_list_sim = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_sim.json', 'rb'))
    scan_list_sim_id = [s.split('_')[0] for s in scan_list_sim]
    scan_list = json.load(open('/mnt/fillipo/huangyue/recon_sim/scans_707.json', 'rb'))
    inst_name_dir = '/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scan_data/instance_id_to_name/'

    ssg_path = '/mnt/fillipo/huangyue/recon_sim/11_release/SSG/'
    DATASET_PATH = '/mnt/fillipo/huangyue/recon_sim/11_release/Annotation_scenes/'

    for scan_idx, one_scan in enumerate(scan_list):
        scan_id = one_scan.split('_')[0]
        if scan_id in scan_list_sim_id: continue
        if one_scan not in ['scene0006_00', 'scene0001_00']: continue

        save_path = run(one_scan, ssg_path, inst_name_dir, DATASET_PATH)







