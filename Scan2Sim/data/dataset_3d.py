

import torch
import numpy as np
import torch.utils.data as data

import yaml
from easydict import EasyDict

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
import pickle
from PIL import Image
import glob
import trimesh



def get_bbox_from_rgb(img, white_threshold=8):
    """
    计算 RGB 图像中非白色区域的边界框，允许背景颜色接近白色。

    :param img: 输入的 RGB 图像
    :param white_threshold: 与白色的距离阈值，值越大，容忍的颜色偏差越大
    :return: 物体区域的边界框 (left, upper, right, lower)
    """
    # 将图像转换为 numpy 数组
    img_np = np.array(img)

    # 定义白色的 RGB 值
    white = np.array([255, 255, 255])

    # 计算每个像素的颜色与白色的差距
    color_diff = np.abs(img_np - white)

    # 如果每个通道的差距都小于阈值，则认为是接近白色的像素（背景）
    mask = np.any(color_diff > white_threshold, axis=-1)

    # 如果整个图像都是接近白色的像素，则返回 None
    if not mask.any():
        return None

    # 获取非白色区域的边界框 (min_row, min_col, max_row, max_col)
    coords = np.argwhere(mask)
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0) + 1

    # 返回边界框 (left, upper, right, lower)
    return (top_left[1], top_left[0], bottom_right[1], bottom_right[0])




def pil_loader(path, center=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)

        if img.mode == 'RGBA':
            alpha = img.split()[3]
            bbox = alpha.getbbox()
        elif img.mode == 'RGB':
            bbox = get_bbox_from_rgb(img)
        else:
            raise ValueError("Unsupported image mode!")

        if bbox:
            # 裁剪出非白色/非透明区域
            img_cropped = img.crop(bbox)

            # 计算缩放比例，使裁剪后的图像尽量铺满背景
            background = Image.new('RGB', img.size, (255, 255, 255))
            bg_w, bg_h = background.size
            cropped_w, cropped_h = img_cropped.size

            # 计算缩放比例，保持宽高比
            scale = min(bg_w / cropped_w, bg_h / cropped_h)
            new_size = (int(cropped_w * scale), int(cropped_h * scale))
            img_resized = img_cropped.resize(new_size)

            # 计算将缩放后的图像粘贴到背景上的位置，居中粘贴
            offset = ((bg_w - new_size[0]) // 2, (bg_h - new_size[1]) // 2)

            # 将缩放后的图像粘贴到白色背景上
            if img.mode == 'RGB':
                background.paste(img_resized, offset)
            else:
                background.paste(img_resized, offset, mask=img_resized.split()[3])

            return background.convert('RGB')
        else:
            print("The image has no non-white or non-transparent areas.")
            return img.convert('RGB')





def farthest_point_sample(point, npoint):
    """
    Input:
        point: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        sampled_points: sampled pointcloud data, [npoint, D]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    point = torch.tensor(point, dtype=torch.float32).to(device)
    N, D = point.shape
    xyz = point[:, :3]
    centroids = torch.zeros((npoint,), dtype=torch.long).to(device)
    distance = torch.ones((N,), dtype=torch.float32).to(device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1, keepdim=True)

    sampled_points = point[centroids, :]
    return sampled_points.cpu().numpy()

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, :3]
        rotated_data[k, :, :3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:] = batch_data[k, :, 3:]
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def barycentric_coordinates(A, B, C, P):
    # Compute the vectors
    v0 = B - A
    v1 = C - A
    v2 = P - A

    # Compute dot products
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    # Compute the denominator of the barycentric coordinates
    denom = d00 * d11 - d01 * d01

    # Compute the barycentric coordinates
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w

def sample_surface(plydata, points_pre_area = 1000):

    total_points = round(plydata.area * points_pre_area)
    sampled_v, face_indices = trimesh.sample.sample_surface_even(plydata, total_points)

    if isinstance(plydata.visual, trimesh.visual.ColorVisuals):
        # For ColorVisuals, directly get vertex colors
        vertex_colors = plydata.visual.vertex_colors[:, :3]
        face_vertex_colors = vertex_colors[plydata.faces[face_indices]]
        bary_coords = trimesh.triangles.points_to_barycentric(plydata.triangles[face_indices], sampled_v)
        vertex_colors = np.einsum('ij,ijk->ik', bary_coords, face_vertex_colors) # / 255.
        vertices = sampled_v

    else:
        interpolated_uvs = np.zeros((sampled_v.shape[0], 2))
        out_segment = []
        for i, point in enumerate(sampled_v):
            A, B, C = plydata.vertices[plydata.faces[face_indices[i]]]
            ind = np.argmin(np.linalg.norm(np.array([A, B, C]) - point, axis=1))
            out_segment.append(plydata.faces[face_indices[i]][ind])
            uv_A, uv_B, uv_C = plydata.visual.uv[plydata.faces[face_indices[i]]]
            alpha, beta, gamma = barycentric_coordinates(A, B, C, point)
            interpolated_uvs[i] = alpha * uv_B + beta * uv_C + gamma * uv_A

        uv = np.array(interpolated_uvs)
        sampled_c = plydata.visual.material.to_color(uv)[:, :3] # / 255
        vertices = sampled_v
        vertex_colors = sampled_c

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(vertices)
    # pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
    # o3d.visualization.draw_geometries([pcd])
    #
    # print(vertices.shape, vertex_colors.shape)
    assert vertices.shape == vertex_colors.shape
    return vertices, vertex_colors


import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


@DATASETS.register_module()
class Ti_anno(data.Dataset):
    def __init__(self, config):

        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.image_path = config.IMAGE_PATH
        self.pc_path = config.PC_PATH
        self.sample_points_num = self.npoints
        self.whole = config.get('whole')

        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)['ti']

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Ti_anno')

        if self.subset == 'train':
            lines = 600
        else:
            lines = 107

        self.save_path = os.path.join(self.data_root, '9_model', f'{self.subset}_{lines}_real_image_ranking.dat')
        assert os.path.exists(self.save_path)

        with open(self.save_path, 'rb') as f:
            self.file_list = pickle.load(f)

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='Ti_anno')

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True

        self.use_height = config.use_height

        if self.augment:
            print("using augmented point clouds.")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        pc_xyz = pc[:, :3]
        centroid = np.mean(pc_xyz, axis=0)
        pc_xyz = pc_xyz - centroid
        m = np.max(np.sqrt(np.sum(pc_xyz ** 2, axis=1)))
        pc_xyz = pc_xyz / m
        pc[:, :3] = pc_xyz
        return pc


    def random_sample(self, pc, num):
        if len(pc) >= num:
            np.random.shuffle(self.permutation)
            return pc[self.permutation[:num]]
        else:
            repeated_pc = np.tile(pc, (num // len(pc) + 1, 1))
            np.random.shuffle(repeated_pc)
            return repeated_pc[:num]

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data_list = {}
        for modality in ['image_pc_data', 'text_pc_data']:
            image_pcd_path_list = sample[modality]['pcd_path_list']
            data_list[modality] = []
            for one_path in image_pcd_path_list:
                one_path = os.path.join(self.pc_path, os.path.basename(one_path))
                data = IO.get(one_path).astype(np.float32)

                if self.uniform and self.sample_points_num < data.shape[0]:
                    data = farthest_point_sample(data, self.sample_points_num)
                else:
                    data = self.random_sample(data, self.sample_points_num)

                data = self.pc_norm(data)
                data[:, 3:] = data[:, 3:]/255.0

                if self.augment and self.subset == 'train':
                    data_xyz = data[:, :3]
                    data_xyz = random_point_dropout(data_xyz[None, ...])
                    data_xyz = random_scale_point_cloud(data_xyz)
                    data_xyz = shift_point_cloud(data_xyz)
                    data_xyz = rotate_perturbation_point_cloud(data_xyz)
                    data_xyz = rotate_point_cloud(data_xyz)
                    data[:, :3] = data_xyz
                    data = data.squeeze()

                if self.use_height:
                    self.gravity_dim = 1
                    height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                               self.gravity_dim:self.gravity_dim + 1].min()
                    data = np.concatenate((data, height_array), axis=1)
                    data = torch.from_numpy(data).float()
                else:
                    data = torch.from_numpy(data).float()

                data_list[modality].append(data)
            data_list[modality] = torch.stack(data_list[modality], dim=0)


        cls_gt = torch.tensor(sample['cls_list'], dtype=torch.float32)
        inst_id = sample['inst_id']
        one_scan = sample['image_pc_data']['one_scan_list'][0]
        label = sample['label']


        caption = f'A photo of a {label}.'
        tokenized_captions = []
        tokenized_captions.append(self.tokenizer(caption))
        tokenized_captions = torch.stack(tokenized_captions)

        picked_image_addr = os.path.join(self.image_path, one_scan, f'{inst_id}_*.png')
        if len(glob.glob(picked_image_addr)) > 0:
            picked_image_addr = glob.glob(picked_image_addr)[0]
            image = pil_loader(picked_image_addr)
            image = self.train_transform(image)
        else:
            # # only do text-encoder
            default_image_size = (3, 224, 224)
            image = torch.zeros(default_image_size)


        return inst_id, one_scan, tokenized_captions, data_list, image, cls_gt

    def __len__(self):
        return len(self.file_list)



import collections.abc as container_abcs
int_classes = int
# from torch import string_classes
string_classes = str

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(batch, list):
        batch = [example for example in batch if example[4] is not None]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)

        # Determine the target shape for the output
        target_shape = [len(batch)] + list(batch[0].shape)  # Example: [64, 1, 77]

        # Resize the `out` tensor to zero elements to avoid resizing warnings
        if out is not None and out.numel() != 0:
            out.resize_(0)

        # Create a new tensor if `out` is not provided
        if out is None:
            out = torch.empty(target_shape, dtype=elem.dtype, device=elem.device)

        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_type, train_transform=None):
        if dataset_type == 'train':
            self.dataset_name = args.pretrain_dataset_name
        elif dataset_type == 'val':
            self.dataset_name = args.validate_dataset_name
        else:
            raise ValueError("not supported dataset type.")
        with open('./data/dataset_catalog.json', 'r') as f:
            self.dataset_catalog = json.load(f)
            self.dataset_usage = self.dataset_catalog[self.dataset_name]['usage']
            self.dataset_split = self.dataset_catalog[self.dataset_name][self.dataset_usage]
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.dataset_type = dataset_type
        if 'colored' in args.model.lower():
            self.use_colored_pc = True
        else:
            self.use_colored_pc = False
        if args.npoints == 10000:
            self.use_10k_pc = True
        else:
            self.use_10k_pc = False
        self.build_3d_dataset(args, self.dataset_config_dir)

    def build_3d_dataset(self, args, config):
        config = cfg_from_yaml_file(config)
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.args = args
        config.use_height = args.use_height
        config.npoints = args.npoints
        config.use_colored_pc = self.use_colored_pc
        config.use_10k_pc = self.use_10k_pc
        config_others = EasyDict({'subset': self.dataset_type, 'whole': True})
        self.dataset = build_dataset_from_cfg(config, config_others)
