import json
import trimesh
import os


metadata_path = '/mnt/fillipo/huangyue/recon_sim/7_anno_v2/anno_info_ranking_v2.json'
metadata_path_sm = '/mnt/fillipo/huangyue/recon_sim/7_anno_v2/anno_info_ranking_v3_sm.json'
save_path = '/mnt/fillipo/huangyue/recon_sim/11_release/Annotation_scenes/'
metadata = json.load(open(metadata_path, 'rb'))
metadata_sm = json.load(open(metadata_path_sm, 'rb'))
for one_scan in metadata:
    print (one_scan)
    scene = trimesh.Scene()
    # if one_scan not in ['scene0006_00']: continue
    save_path_one_scan = os.path.join(save_path, one_scan)
    os.makedirs(save_path_one_scan, exist_ok=True)
    objs_info = metadata[one_scan]
    objs_info_sm = []
    if one_scan in metadata_sm:
        objs_info_sm = metadata_sm[one_scan]

    for inst_id in objs_info:

        one_obj = objs_info[inst_id]
        mesh_path = one_obj['mesh_path']
        matrix = one_obj['matrix']

        mesh = trimesh.load(mesh_path, force='mesh')

        mesh.apply_transform(matrix)
        os.makedirs(os.path.join(save_path_one_scan, inst_id) ,exist_ok=True)
        mesh.export(os.path.join(save_path_one_scan, inst_id, f'{inst_id}.obj'))
        #scene.add_geometry(mesh)


    for inst_id in objs_info_sm:

        one_obj = objs_info_sm[inst_id]
        mesh_path = one_obj['mesh_path']
        matrix = one_obj['matrix']


        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_transform(matrix)
        os.makedirs(os.path.join(save_path_one_scan, inst_id), exist_ok=True)
        mesh.export(os.path.join(save_path_one_scan, inst_id, f'{inst_id}.obj'))
        #scene.add_geometry(mesh)


    #scene.show()


    # ## show
    # objs = os.listdir(save_path_one_scan)
    # for one_mesh in objs:
    #     mesh = trimesh.load(os.path.join(save_path_one_scan, one_mesh, f'{one_mesh}.obj'))
    #     scene.add_geometry(mesh)
    #
    # scene.show()



