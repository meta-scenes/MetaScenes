import trimesh
import json
import os
import argparse
import numpy as np

def show_simulation_results(simu_matrix_info_path, dataset_path, one_scan, export_dir=None):
    """
    Load objects, apply transformation matrices from simulation results, and visualize them.
    Optionally export transformed objects as .obj files.

    Args:
        simu_matrix_info_path (str): Path to the simulation matrix JSON file.
        dataset_path (str): Base directory containing dataset object files.
        one_scan (str): Scene ID.
        export_path (str, optional): If provided, saves transformed objects as .obj files.
    """
    # Load simulation matrix information
    with open(simu_matrix_info_path, 'r', encoding='utf-8') as f:
        simu_matrix_info = json.load(f)
    simu_matrix_info = simu_matrix_info['matrix']

    # Create a Trimesh scene
    scene = trimesh.Scene()

    # Ensure export directory exists if exporting is enabled
    export_path = os.path.join(export_dir, one_scan)
    if export_path:
        os.makedirs(export_path, exist_ok=True)

    for inst_id, matrix in simu_matrix_info.items():
        obj_path = os.path.join(dataset_path, one_scan, inst_id, f'{inst_id}.obj')

        if not os.path.exists(obj_path):
            print(f"Warning: Object file not found: {obj_path}")
            continue

        # Load the mesh
        mesh = trimesh.load(obj_path)

        # Center object before transformation
        current_center = mesh.bounding_box.centroid
        mesh.apply_translation(-current_center)

        # Apply transformation matrix
        matrix_np = np.array(matrix)
        mesh.apply_transform(matrix_np)

        # Add to scene
        scene.add_geometry(mesh)

        # Export transformed object if export path is provided
        if export_path:
            export_file = os.path.join(export_path, f"{inst_id}.obj")
            mesh.export(export_file)
            print(f"Exported: {export_file}")

    # Show the scene in an interactive viewer
    scene.show()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize and optionally export transformed 3D objects.")

    parser.add_argument('--one_scan', required=True, help="Scene ID to visualize (e.g., scene0001_00).")
    parser.add_argument('--simu_matrix_info_path', required=True, help="Path to simulation matrix JSON file.")
    parser.add_argument('--dataset_path', required=True, help="Base directory containing dataset object files.")
    parser.add_argument('--export_dir', default=None, help="Optional directory to export transformed objects as .obj files.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    show_simulation_results(args.simu_matrix_info_path, args.dataset_path, args.one_scan, args.export_dir)
