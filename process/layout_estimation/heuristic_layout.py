import trimesh
import json
import os
import argparse


def generate_floor_from_scene(one_scan, dataset_path, output_json_path, is_save=True, visualize=False, padding=0):
    """
    Generate the floor from a given scene based on object bounding boxes.

    Parameters:
    one_scan (str): The name of the scan in the dataset.
    dataset_path (str): Path to annotated scenes.
    output_json_path (str): Path to save the generated floor data in JSON format.
    is_save (bool): Whether to save the floor data as a JSON file. Default is True.
    visualize (bool): Whether to visualize the floor and scene. Default is False.
    padding (float): Padding to extend the floor boundary. Default is 0 (no padding).

    The function generates a floor for the scene by calculating its bounding box, optionally adding padding,
    and saves the floor data as a JSON file. If visualize=True, it also displays the floor and the scene.
    """

    scene = trimesh.Scene()
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    dataset_one_scan = os.path.join(one_scan, dataset_path)
    objs_list = os.listdir(dataset_one_scan)
    objs_list.sort()
    for inst_id in objs_list:
        mesh_path = os.path.join(dataset_one_scan, inst_id, f'{inst_id}.obj')
        mesh = trimesh.load(mesh_path, force='mesh')
        scene.add_geometry(mesh)

        # Use the bounding box to calculate the scene's boundaries
        bbox = mesh.bounding_box.bounds
        min_x, min_y = min(min_x, bbox[0][0]), min(min_y, bbox[0][1])
        max_x, max_y = max(max_x, bbox[1][0]), max(max_y, bbox[1][1])

    # Apply padding to the floor's bounding box
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    # Define the four corners of the floor
    vert = [
        [min_x, min_y, 0],
        [max_x, min_y, 0],
        [max_x, max_y, 0],
        [min_x, max_y, 0]
    ]

    # Define the quadrilateral face (the floor)
    quads = [[0, 1, 2, 3]]

    # Save the floor data as a JSON file
    if is_save:
        floor_data = {"vert": vert, "quads": quads}
        with open(output_json_path, "w") as f:
            json.dump(floor_data, f, indent=4)

    if visualize:
        # Create a floor with a thickness of 1
        floor_width = max_x - min_x
        floor_height = max_y - min_y
        floor_thickness = 1  # Thickness of 1 unit

        floor_mesh = trimesh.creation.box(extents=[floor_width, floor_height, floor_thickness])

        # Move the floor so its bottom is at z=0
        floor_mesh.apply_translation([(min_x + max_x) / 2, (min_y + max_y) / 2, -floor_thickness / 2])

        # Add the floor to the scene
        scene.add_geometry(floor_mesh)

        # Visualize the scene with the floor
        scene.show()


def show_one_scene(one_scan, metadata):
    """
    Display a single scene based on the provided scan name and metadata.

    Parameters:
    one_scan (str): The name of the scan in the dataset.
    metadata (dict): Metadata containing object info for the scene.

    The function loads and visualizes the mesh objects in the given scene.
    """

    scene = trimesh.Scene()

    objs_info = metadata[one_scan]
    for inst_id in objs_info:
        one_obj = objs_info[inst_id]
        mesh_path = one_obj['mesh_path']
        matrix = one_obj['matrix']
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_transform(matrix)
        scene.add_geometry(mesh)

    # Visualize the scene
    scene.show()


def parse_args():
    """
    Parse command-line arguments.

    Returns:
    argparse.Namespace: A namespace object containing the command-line arguments.

    Example usage:
    python script.py --scans scene0001_00 --anno_path /path/to/annotated/scenes --output_dir /path/to/output

    """
    parser = argparse.ArgumentParser(description="Generate floor from scene data.")

    parser.add_argument('--scans', nargs='+', required=True, help="List of scans to process (e.g., scene0001_00).")
    parser.add_argument('--dataset_path', required=True, help="Path to the annotation scene.")
    parser.add_argument('--output_dir', required=True, help="Directory to save the output JSON files.")
    parser.add_argument('--padding', type=float, default=0, help="Padding to apply to the floor boundaries.")
    parser.add_argument('--visualize', action='store_true', help="Visualize the scene and floor.")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()


    # Process each scan in the list
    for one_scan in args.scans:
        output_json_path = os.path.join(args.output_dir, f'{one_scan}.json')
        if one_scan not in metadata: continue
        generate_floor_from_scene(one_scan, args.dataset_path, output_json_path, is_save=True, visualize=args.visualize,
                                  padding=args.padding)
        print (f'Processing {one_scan} done')
