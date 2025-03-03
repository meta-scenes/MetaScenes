import os
import numpy as np
import os.path as osp
import json
import trimesh
import glob


class FreePose:
    def __init__(self, cfg):
        """
        Initialize the FreePose class.

        Args:
            cfg (dict): Configuration dictionary containing paths and parameters.
        """
        self.cfg = cfg
        self.method = cfg["method"]
        self.dataset_path = cfg["dataset_path"]
        self.asset_path = cfg["asset_path"]
        self.output_path = cfg["output_path"]
        self.inst_name_dir = cfg["inst_name_dir"]
        self.init_pose = cfg["init_pose"]  # 4x4 transformation matrix

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

    def transform_mesh_ignore_z(self, transform):
        """
        Apply transformation to mesh while ignoring Z-axis rotation.

        Args:
            mesh (trimesh.Trimesh): The input mesh.
            transform (np.ndarray): 4x4 transformation matrix.

        Returns:
            trimesh.Trimesh: Transformed mesh.
        """
        transform_xy = np.eye(4)
        transform_xy[:2, :2] = transform[:2, :2]
        transform_xy[:2, 3] = transform[:2, 3]
        return transform_xy

    def get_mesh_bbox(self, mesh):
        """
        Compute the oriented bounding box of a mesh.

        Args:
            mesh (trimesh.Trimesh): The input mesh.

        Returns:
            tuple: (length, width, height, bounding box)
        """
        bbox = mesh.bounding_box_oriented
        dimensions = bbox.bounds[1] - bbox.bounds[0]
        bbox.visual.face_colors = [0, 1, 0]
        return (*dimensions, bbox)

    def scale_gt_pred_mesh(self, gt_mesh, pred_mesh):
        """
        Compute the scale ratio to align generated mesh with ground truth mesh.

        Args:
            gt_mesh (trimesh.Trimesh): Ground truth mesh.
            pred_mesh (trimesh.Trimesh): Generated mesh.

        Returns:
            float: Scale factor.
        """
        gt_l, gt_w, gt_h, _ = self.get_mesh_bbox(gt_mesh)
        pred_l, pred_w, pred_h, _ = self.get_mesh_bbox(pred_mesh)
        return max(gt_l, gt_w, gt_h) / max(pred_l, pred_w, pred_h)

    def parse_asset_path(self, scan, inst_id):
        """
        Retrieve generated mesh path and corresponding frame index.

        Args:
            scan (str): Scan identifier.
            inst_id (str): Instance identifier.

        Returns:
            tuple: (Generated mesh path, Frame index)
        """
        gen_mesh_files = glob.glob(osp.join(self.asset_path, f'{scan}_{inst_id}', f"{inst_id}_*.obj"))
        if not gen_mesh_files:
            raise FileNotFoundError(f"No generated mesh found for {inst_id} in {scan}.")
        gen_mesh_path = gen_mesh_files[0]
        frame_idx = osp.basename(gen_mesh_path).split("_")[1].split('.')[0]
        return gen_mesh_path, frame_idx

    def run(self, scan_id, save_output_mesh=True, show_output_mesh=False):
        """
        Process a single scan and align generated meshes.

        Args:
            scan (str): Scan identifier.
            save_output_mesh (bool): Whether to save the aligned meshes.
            show_output_mesh (bool): Whether to visualize the meshes.
        """
        output_dir = osp.join(self.output_path, self.method, scan_id)
        os.makedirs(output_dir, exist_ok=True)
        axis_align_matrix = self.load_axis_align_matrix(scan_id)
        inst_id_to_name = json.load(open(osp.join(self.inst_name_dir, scan_id, f"{scan_id}.json"), "r"))
        org_obj_path = osp.join(self.dataset_path, scan_id, "obj")

        assets = os.listdir(self.asset_path)
        assets.sort()

        transformation_data = {}
        for one_asset in assets:
            _, _, inst_id = one_asset.split('_')
            label = inst_id_to_name[int(inst_id)]
            if label in ["floor", "wall", "ceiling"]:
                continue

            gen_mesh_path, frame_idx = self.parse_asset_path(scan_id, inst_id)
            gt_mesh_path = osp.join(org_obj_path, f"{inst_id}-{label}.obj")


            if not (osp.exists(gen_mesh_path) and osp.exists(gt_mesh_path)):
                continue
            pred_mesh = trimesh.load(gen_mesh_path)
            gt_mesh = trimesh.load(gt_mesh_path)

            # Register and align the mesh
            pose_path = osp.join(self.dataset_path, scan_id, f"{scan_id}_pose", f"{frame_idx}.txt")
            camera_pose = np.loadtxt(pose_path)
            transform_ignore_z = camera_pose @ self.init_pose
            transform_xy = self.transform_mesh_ignore_z(transform_ignore_z)
            pred_mesh.apply_transform(transform_xy)
            pred_mesh.apply_transform(axis_align_matrix)
            rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
            pred_mesh.apply_transform(rotation_matrix)
            gt_mesh.apply_transform(axis_align_matrix)

            # Store the transformation matrix
            transformation_matrix = rotation_matrix @ axis_align_matrix @ transform_xy


            # Scale the generated mesh
            scale_factor = self.scale_gt_pred_mesh(gt_mesh, pred_mesh)
            pred_mesh.apply_scale(scale_factor)
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_matrix[1, 1] = scale_matrix[2, 2] = scale_factor
            transformation_matrix = scale_matrix @ transformation_matrix

            # Translate to match centroids
            translation = gt_mesh.centroid - pred_mesh.centroid
            pred_mesh.apply_translation(translation)
            transformation_matrix[:3, 3] += translation


            transformation_data[inst_id] = transformation_matrix.tolist()


            if save_output_mesh:
                mesh_save_path = osp.join(output_dir, f"{inst_id}.obj")
                os.makedirs(osp.dirname(mesh_save_path), exist_ok=True)
                pred_mesh.export(mesh_save_path)

        # Save the transformation matrix data
        transformation_file_path = osp.join(output_dir, "transformations.json")
        with open(transformation_file_path, "w") as f:
            json.dump(transformation_data, f, indent=4)

        if show_output_mesh:
            show_mesh_list = []
            for inst_id, transformation_matrix in transformation_data.items():
                gen_mesh_path, _ = self.parse_asset_path(scan_id, inst_id)
                pred_mesh = trimesh.load(gen_mesh_path)
                pred_mesh.apply_transform(np.array(transformation_matrix))
                show_mesh_list.append(pred_mesh)

            trimesh.Scene(show_mesh_list).show()

