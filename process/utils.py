import numpy as np
from sklearn.cluster import KMeans
import random
from rembg import remove
import pickle
from PIL import Image
import math
import open3d as o3d
from scipy.ndimage import binary_erosion, binary_dilation

FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

orgInstID_to_id = {id : id - 1 for id in range(1, 257)}
orgInstID_to_id[0] = -100
LARGE_OBJ_LIST = ['wall', 'floor', 'bed', 'ceiling', 'table', 'bookshelf', 'hood', 'stair', 'rail', 'mattress', 'carpet', 'piano', 'bedframe', 'beam']
MED_OBJ_LIST = ['chair', 'box', 'cabinet', 'desk', 'dresser', 'shelf', 'couch', 'sink', 'toilet', 'nightstand', 'frige', 'bathtub', 'counter', 'stove', 'bench', 'seat', 'container', 'stairs', 'oven', 'bin', 'bar', 'rack', 'ladder', 'stand', 'door', 'clothes', 'computer', 'suitcase', 'ottoman', 'board', 'closet', 'shower', 'machine', 'structure', 'shelves', 'fireplace', 'luggage', 'case', 'bicycle', 'guitar', 'lamp', 'board', 'printer', 'statue', 'urinal', 'wood', 'board', 'platform', 'plants', 'elevator', 'chandelier', 'house', 'furnace', 'media' ]


def check_converge(vertices, obj_label):
    converge = False

    x_diff = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    y_diff = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    z_diff = np.max(vertices[:, 2]) - np.min(vertices[:, 2])

    if x_diff < 0.2 or y_diff < 0.2 or z_diff < 0.2: # slice
        converge = True
    # if x_diff < 0.5 and y_diff < 0.5: # small cube
    #     converge = True
    if x_diff < 1.0 and y_diff < 0.5: # small obj
        converge = True
    if x_diff < 0.5 and y_diff < 1.0: # small obj
        converge = True

    if any([_obj_l in obj_label for _obj_l in MED_OBJ_LIST]):
        if x_diff < 1.5 and y_diff < 1.5: # cube
            converge = True
        if x_diff < 2.6 and y_diff < 1.3: # 1:2
            converge = True
        if x_diff < 1.3 and y_diff < 2.6: # 2:1
            converge = True
    if any([_obj_l in obj_label for _obj_l in LARGE_OBJ_LIST]):
        if x_diff < 2.5 or y_diff < 2.5: # slice
            converge = True

    return converge, x_diff, y_diff

def display_inlier_outlier(inlier_cloud, obj_label, x_diff, y_diff, _iter, converge, extra=None):
    new_vertices = np.asarray(inlier_cloud.points)
    x_diff_new = np.max(new_vertices[:, 0]) - np.min(new_vertices[:, 0])
    y_diff_new = np.max(new_vertices[:, 1]) - np.min(new_vertices[:, 1])

    print(f"Showing outliers (blue) and inliers (gray) after {_iter} iterations with {converge} convergence: ")
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    geo_list = [inlier_cloud]
    if extra:
        for _e in extra:
            _e.paint_uniform_color([0, 0, 1])
        geo_list.extend(extra)
    o3d.visualization.draw_geometries(geo_list, window_name=f"{obj_label} {_iter} iterations with {converge} convergence {x_diff:.2f}-{x_diff_new:.2f} {y_diff:.2f}-{y_diff_new:.2f}")

def remove_outlier(o3d_pcd, obj_label, _iter=0):
    vertices = np.asarray(o3d_pcd.points)
    vert_shape = vertices.shape[0]
    if vert_shape < 1000:
        return o3d_pcd, None
    nb_pts_large = [0.001, 0.001, 0.002, 0.004]
    nb_pts = [0.005, 0.005, 0.007, 0.009]
    nb_nb = [0.1, 0.2, 0.3, 0.4]
    if _iter % 2 == 0:
        if any([_obj_l in obj_label for _obj_l in LARGE_OBJ_LIST]):
            pts_thres = nb_pts_large[int(_iter / 2)]
        else:
            pts_thres = nb_pts[int(_iter / 2)]
        assert int(pts_thres*vert_shape) > 0, str(int(pts_thres*vert_shape)) + '-' + str(vert_shape)
        cl, ind = o3d_pcd.remove_radius_outlier(nb_points=int(pts_thres*vert_shape), radius=0.1) # remove most of small outliers, bed not good
    else:
        neighbor_thres = nb_nb[int(_iter / 2)]
        cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=int(neighbor_thres*vert_shape), std_ratio=2.0)

    if len(ind) == vertices.shape[0]:
        return o3d_pcd, None
    else:
        extra = o3d_pcd.select_by_index(ind, invert=True)
        o3d_pcd = o3d_pcd.select_by_index(ind)
        return o3d_pcd, extra

def remove_outlier_iter(vertices, obj_label, visualize=False):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(vertices)
    # o3d.visualization.draw_geometries([o3d_pcd])
    _, x_diff, y_diff = check_converge(vertices, obj_label)
    # if converge:
    #     print ('Is converge init')
    #     return o3d_pcd

    extra = []
    _iter = 0
    converge = False
    # while converge or _iter < 8:
    while _iter < 8:
        o3d_pcd, _e = remove_outlier(o3d_pcd, obj_label, _iter=_iter)
        if _e:
            extra.append(_e)
        vertices = np.asarray(o3d_pcd.points)
        if vertices.shape[0] == 0:
            return None
        converge, _, _ = check_converge(vertices, obj_label)
        _iter += 1
        if converge:
            break
    if visualize:
        display_inlier_outlier(o3d_pcd, obj_label, x_diff, y_diff, _iter, converge, extra)
    if converge:
        return o3d_pcd
    else:
        return None


def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)

def proj_mat_func(fov, aspectRatio, znear, zfar):
    proj_w = 1 / np.tan(fov)
    proj_h = aspectRatio / np.tan(fov)

    m_proj = np.array([[proj_w, 0, 0, 0],
                        [0, proj_h, 0, 0],
                        [0, 0, zfar / (zfar - znear), zfar * znear / (znear - zfar)],
                        [0,   0,  1,  0]])
    return m_proj

def get_proj_mat(camera_to_world, align_mat, int_color_mat, color_w, color_h):
    m_view = np.linalg.inv(camera_to_world)
    znear, zfar = 0.1, 15
    fx, fy = int_color_mat[0, 0], int_color_mat[1, 1]
    fov = np.arctan(color_w / 2 / fx) # half fov in randius
    aspect = color_w / color_h
    m_proj = proj_mat_func(fov, aspect, znear, zfar)

    proj_mat = np.matmul(m_proj, m_view)
    if align_mat is not None:
        proj_mat = np.matmul(np.linalg.inv(align_mat.transpose()), proj_mat.transpose())
    return proj_mat

def get_proj_pts(pt_3d, proj_mat, img_w, img_h):
    pts = np.ones((pt_3d.shape[0], 4), dtype = pt_3d.dtype)
    pts[:, 0:3] = pt_3d
    pt_2d = np.dot(pts, proj_mat)

    out_pt = []
    z = []
    for pt_idx in range(len(pt_2d)):
        pt = pt_2d[pt_idx]

        pt_x = int((pt[0] / pt[3] + 1) / 2 * img_w)
        pt_y = int((pt[1] / pt[3] + 1) / 2 * img_h)
        if pt_x < 0 or pt_x >= img_w or pt_y < 0 or pt_y >= img_h or pt[2] < 0:
            continue
        out_pt.append([pt_x, pt_y])
        z.append(pts[pt_idx][2])

    return out_pt, z

def get_proj_pts_tiny(pt_3d, proj_mat, img_w, img_h, bbox_t):
    pts = np.ones((pt_3d.shape[0], 4), dtype = pt_3d.dtype)
    pts[:, 0:3] = pt_3d
    pt_2d = np.dot(pts, proj_mat)
    out_pt = []
    z = []
    for pt_idx in range(len(pt_2d)):
        pt = pt_2d[pt_idx]

        # pt_x = int((pt[0] / pt[3] + 1) / 2 * 550)
        # pt_y = int((pt[1] / pt[3] + 1) / 2 * 550)
        pt_x = int((pt[0] / pt[3] + 1) / 2 * img_w)
        pt_y = int((pt[1] / pt[3] + 1) / 2 * img_h)

        if pt_x > bbox_t[1] and pt_x < bbox_t[3] and pt_y > bbox_t[0] and pt_y < bbox_t[2]:
            out_pt.append(pt)
            z.append(pts[pt_idx][2])

    if out_pt == []:
        return []

    pts_tiny = np.dot(out_pt, np.linalg.inv(proj_mat))
    return pts_tiny


def rotate_vect_x(v, theta):
    R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    v_rotated = np.dot(R, v)
    return v_rotated

def rotate_vect_y(v, theta):
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    v_rotated = np.dot(R, v)
    return v_rotated

def rotate_vect_z(v, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    v_rotated = np.dot(R, v)
    return v_rotated
def random_sample(pcd, sample_tps):
    if len(pcd) > sample_tps:
        return random.sample(pcd, sample_tps)
    else:
        return pcd

def kmeans_sample(pcd, sam_pts_max):
    if len(pcd) > sam_pts_max:
        kmeans_sample = KMeans(n_clusters= sam_pts_max, random_state=0, n_init= 'auto'). fit(pcd)
        return kmeans_sample.cluster_centers_
    else:
        return pcd

def pred_bbox(image):
    image_nobg = remove(image.convert('RGBA'), alpha_matting=True)
    alpha = np.asarray(image_nobg)[:,:,-1]
    x_nonzero = np.nonzero(alpha.sum(axis=0))
    y_nonzero = np.nonzero(alpha.sum(axis=1))
    if len(x_nonzero[0]) == 0 or len(y_nonzero[0]) == 0:
        return 0,0,0,0
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    return x_min, y_min, x_max, y_max


def calculate_iou(bbox1, bbox2):
    # 获取bbox1的坐标
    x1_min, y1_min, x1_max, y1_max = bbox1
    # 获取bbox2的坐标
    x2_min, y2_min, x2_max, y2_max = bbox2

    # 计算相交区域的坐标
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    # 计算相交区域的面积
    intersection_area = x_overlap * y_overlap

    # 计算bbox1和bbox2的面积
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算并集的面积
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算交并比
    iou = intersection_area / union_area

    return iou

def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def is_black_image(img):
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixels[x, y] != (0, 0, 0):
                return False
    return True

def is_white_image(img):
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixels[x, y] != (255, 255, 255):
                return False
    return True


def resize_square_img(img):
    resize_rate = 968. / 1296.
    w, h = img.size
    inpaint_img_ = img.resize((w, int(w * resize_rate)))
    inpaint_bg = Image.new('RGB', (w, h), (255, 255, 255))
    inpaint_bg.paste(inpaint_img_, box=(0, 0))
    return inpaint_bg

def are_lines_perpendicular(line1, line2):

    if line1[1][0] - line1[0][0] != 0:
        slope1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0])
    else:
        slope1 = np.inf


    if line2[1][0] - line2[0][0] != 0:
        slope2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])
    else:
        slope2 = np.inf


    return not(slope1 == slope2)

def get_dist (point1, point2):
    distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)
    return distance

def remove_noise_from_points(points, img_shape, iterations=1):
    """
    移除坐标点中的噪点

    参数:
    points (list of list of int): 点的坐标列表，例如 [[x1, y1], [x2, y2], ...]
    img_shape (tuple of int): 图像的形状，例如 (height, width)
    iterations (int): 形态学操作的迭代次数

    返回:
    list of list of int: 处理后的坐标点列表
    """
    # 将坐标点转换为二值图像
    binary_img = np.zeros(img_shape, dtype=np.uint8)
    for x, y in points:
        binary_img[y, x] = 1

    # 定义一个3x3的核
    structure = np.ones((3, 3), dtype=np.uint8)

    # 应用腐蚀和膨胀操作
    eroded_img = binary_erosion(binary_img, structure=structure, iterations=iterations).astype(np.uint8)
    processed_img = binary_dilation(eroded_img, structure=structure, iterations=iterations).astype(np.uint8)

    # 从处理后的二值图像中提取坐标
    processed_points = np.argwhere(processed_img == 1)
    processed_points = [[x, y] for y, x in processed_points]

    return processed_points

rotate_func_map = {
    'x' : rotate_vect_x,
    'y' : rotate_vect_y,
    'z' : rotate_vect_z,
}

