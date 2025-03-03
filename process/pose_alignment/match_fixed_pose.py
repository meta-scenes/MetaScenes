import os
import os.path as osp
import numpy as np
import json
import trimesh


class FixedPose:
    def __init__(self, cfg):
        """
        Initialize FixedPose with configuration settings.

        Args:
            cfg (dict): Configuration dictionary containing paths and parameters.
        """
        self.cfg = cfg
        self.method = cfg["method"]
        self.dataset_path = cfg["dataset_path"]
        self.asset_path = cfg["asset_path"]
        self.output_path = cfg["output_path"]
        self.inst_name_dir = cfg["inst_name_dir"]

    def load_axis_align_matrix(self, scan_id):
        """
        Load the axis alignment matrix for a given scan.

        Args:
            scan_id (str): The scan identifier.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        matrix_file = osp.join(self.dataset_path, scan_id, f"{scan_id}.txt")
        with open(matrix_file, "r") as f:
            for line in f:
                if "axisAlignment" in line:
                    matrix_values = [float(x) for x in line.split("=")[1].strip().split()]
                    return np.array(matrix_values).reshape((4, 4))

        return np.eye(4)

    def get_mesh_bbox(self, mesh):
        """
        Compute the oriented bounding box of a mesh.

        Args:
            mesh (trimesh.Trimesh): The input mesh.

        Returns:
            tuple: (length, width, height, bounding box)
        """
        bbox = mesh.bounding_box_oriented
        bounds = bbox.bounds
        dimensions = bounds[1] - bounds[0]

        bbox.visual.face_colors = [0, 1, 0]
        return (*dimensions, bbox)

    def bbox_iou(self, box1, box2):
        """
        Compute the IoU (Intersection over Union) of two 2D bounding boxes.

        Args:
            box1 (list): [x_min, y_min, x_max, y_max]
            box2 (list): [x_min, y_min, x_max, y_max]

        Returns:
            float: IoU value.
        """
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def run(self, scan_id, save_output_mesh=True, show_output_mesh=False):
        """
        Process a single scan, aligning and transforming generated meshes to match ground truth.

        Args:
            scan_id (str): The scan identifier.
            save_output_mesh (bool): Whether to save the aligned meshes.
            show_output_mesh (bool): Whether to visualize the meshes.
        """
        output_dir = osp.join(self.output_path, self.method, scan_id)
        os.makedirs(output_dir, exist_ok=True)

        axis_align_matrix = self.load_axis_align_matrix(scan_id)
        inst_id_to_name = json.load(open(osp.join(self.inst_name_dir, scan_id, f"{scan_id}.json"), "r"))

        assets = os.listdir(self.asset_path)
        assets.sort()

        obj_path = osp.join(self.dataset_path, scan_id, "obj")

        transformation_data = {}

        for one_asset in assets:
            _, _, inst_id = one_asset.split('_')
            label = inst_id_to_name[int(inst_id)]
            gen_mesh_path = osp.join(self.asset_path, f'{scan_id}_{inst_id}', f"{inst_id}.obj")
            gt_mesh_path = osp.join(obj_path, f"{inst_id}-{label}.obj")

            if not (osp.exists(gen_mesh_path) and osp.exists(gt_mesh_path)):
                continue

            pred_mesh = trimesh.load(gen_mesh_path)
            gt_mesh = trimesh.load(gt_mesh_path)
            gt_mesh.apply_transform(axis_align_matrix)

            # Store the initial transformation matrix
            transformation_matrix = np.eye(4)

            # Compute bounding boxes
            gt_l, gt_w, gt_h, _ = self.get_mesh_bbox(gt_mesh)
            pred_l, pred_w, pred_h, _ = self.get_mesh_bbox(pred_mesh)

            # Scale height
            pred_mesh.vertices *= [1.0, 1.0, gt_h / pred_h]
            transformation_matrix[2, 2] *= gt_h / pred_h

            # Scale length/width
            scale_factor = max(gt_l, gt_w) / max(pred_l, pred_w)
            pred_mesh.vertices *= [scale_factor, scale_factor, 1.0]
            transformation_matrix[0, 0] *= scale_factor
            transformation_matrix[1, 1] *= scale_factor

            # Translate to match centroids
            translation = gt_mesh.centroid - pred_mesh.centroid
            pred_mesh.apply_translation(translation)
            transformation_matrix[:3, 3] += translation

            # Find best rotation
            gt_bbox_2d = [gt_mesh.centroid[0] - gt_l / 2, gt_mesh.centroid[1] - gt_w / 2,
                          gt_mesh.centroid[0] + gt_l / 2, gt_mesh.centroid[1] + gt_w / 2]

            max_iou, best_rotation = -1, None
            for i in range(4):
                delta_angle = np.pi / 2 * i
                temp_mesh = pred_mesh.copy()
                rotation_matrix = trimesh.transformations.rotation_matrix(delta_angle, [0, 0, 1], temp_mesh.centroid)
                temp_mesh.apply_transform(rotation_matrix)

                pred_bounds = temp_mesh.bounding_box.bounds
                pred_bbox_2d = [temp_mesh.centroid[0] - (pred_bounds[1][0] - pred_bounds[0][0]) / 2,
                                temp_mesh.centroid[1] - (pred_bounds[1][1] - pred_bounds[0][1]) / 2,
                                temp_mesh.centroid[0] + (pred_bounds[1][0] - pred_bounds[0][0]) / 2,
                                temp_mesh.centroid[1] + (pred_bounds[1][1] - pred_bounds[0][1]) / 2]

                iou = self.bbox_iou(gt_bbox_2d, pred_bbox_2d)
                if iou > max_iou:
                    best_rotation, max_iou = rotation_matrix, iou

            if best_rotation is not None:
                pred_mesh.apply_transform(best_rotation)
                transformation_matrix = best_rotation @ transformation_matrix

            # Save transformation matrix for the mesh
            transformation_data[inst_id] = transformation_matrix.tolist()

            if save_output_mesh:
                mesh_save_path = osp.join(self.output_path, scan_id, f"{inst_id}.ply")
                os.makedirs(osp.dirname(mesh_save_path), exist_ok=True)
                pred_mesh.export(mesh_save_path)

        # Save the transformation data as a JSON file
        transformation_file_path = osp.join(self.output_path, self.method, scan_id, "transformations.json")
        with open(transformation_file_path, "w") as f:
            json.dump(transformation_data, f, indent=4)

        if show_output_mesh:
            show_mesh_list = []
            for inst_id in transformation_data:
                gen_mesh_path = osp.join(self.asset_path, f'{scan_id}_{inst_id}', f"{inst_id}.obj")
                if not osp.exists(gen_mesh_path):
                    continue

                pred_mesh = trimesh.load(gen_mesh_path)
                transformation_matrix = np.array(transformation_data[inst_id])
                pred_mesh.apply_transform(transformation_matrix)

                show_mesh_list.append(pred_mesh)

            trimesh.Scene(show_mesh_list).show()

