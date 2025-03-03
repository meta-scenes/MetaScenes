
import os
import json
import glob
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
import numpy as np
import trimesh
from dataset_3d import sample_surface
import pickle
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
def get_train_and_test_data():
    datapath = []
    classes = {}
    cls_idx = 0
    for one_scan in scan_list:
        inst_id_to_name = json.load(open(
            f'/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scan_data/instance_id_to_name/{one_scan}.json',
            'rb'))
        for method_id in methods:
            method_name = methods[method_id]['name']
            if not os.path.exists(os.path.join(method_dir, method_name)): continue
            for inst_id, label in enumerate(inst_id_to_name):
                inst_id = str(inst_id)
                if method_id in ['6']:
                    template_path = methods[method_id]['output_path'].replace('*ONESCAN*', one_scan).replace(
                        '*INSTID*', inst_id)
                    for idx in range(5):
                        mesh_path = template_path.replace('*FRAMEID*', str(idx))
                        cls_name = f'{label}-{inst_id}-{method_id}-{one_scan}'
                        if not os.path.exists(mesh_path): continue
                        datapath.append([cls_name, mesh_path])
                        if cls_name not in classes:
                            classes[cls_name] = cls_idx
                            cls_idx += 1
                else:
                    template_path = methods[method_id]['mesh_path'].replace('*ONESCAN*', one_scan).replace(
                        '*INSTID*', inst_id).replace('*FRAMEID*', '*')
                    mesh_path = glob.glob(template_path)
                    if len(mesh_path) == 0:
                        continue
                    cls_name = f'{label}-{inst_id}-{method_id}-{one_scan}'
                    datapath.append([cls_name, mesh_path[0]])
                    if cls_name not in classes:
                        classes[cls_name] = cls_idx
                        cls_idx += 1

    # random.shuffle(datapath)
    return datapath, classes

def run(fn):

    cls = classes[fn[0]]
    cls = np.array([cls]).astype(np.int32)
    method_id = fn[0].split('-')[-2]
    mesh = trimesh.load(fn[1], force='mesh')
    if method_id in ['1', '2', '3', '6']:
        rotate_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rotate_matrix)
        # mesh.show()
    try:
        assert len(mesh.vertices) > 15
        xyz, rgb = sample_surface(mesh, points_pre_area=1000)
        xyz = xyz.astype(np.float32)
        rgb = rgb.astype(np.float32)

    except Exception as e:
        print(e)
        xyz = mesh.vertices.astype(np.float32)
        rgb = np.ones_like(xyz) * 255

    point_set = np.hstack((xyz, rgb))


    point_set = farthest_point_sample(point_set, npoints)

    return point_set, cls, fn[1]



if __name__ == "__main__":


    ### Generating test dataset
    root = '/mnt/fillipo/huangyue/recon_sim/'
    methods = json.load(open('./recon_method.json','rb'))
    scan_list = ['scene0000_00', 'scene0009_02', 'scene0019_00']
    method_dir = os.path.join(root, '3_recon')
    npoints = 10000
    save_path = os.path.join(root, '9_model', 'dataloader',
                                          'Ti_%d_%dpts_fps.dat' % (len(scan_list), npoints))
    datapath, classes = get_train_and_test_data()

    with parallel_backend('multiprocessing', n_jobs=20):
        outputs = Parallel()(
            delayed(run)(fn=fn) for fn in tqdm(datapath))

    list_of_points, list_of_labels, list_of_datapath = zip(*outputs)

    list_of_points = list(list_of_points)
    list_of_labels = list(list_of_labels)
    list_of_datapath = list(list_of_datapath)

    with open(save_path, 'wb') as f:
        pickle.dump([list_of_points, list_of_labels, list_of_datapath], f)
        print('done')
