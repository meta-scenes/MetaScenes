import os
import argparse
import trimesh
import numpy as np
import json


def rotate_z_90(verts):
    """Rotate vertices around the Z-axis by 90 degrees around the origin."""
    rotation_matrix = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    # Apply the rotation to all vertices (rotate x and y, keep z unchanged)
    rotated_verts = verts.copy()  # Make a copy of the vertices array to avoid modifying the original
    rotated_verts[:, :2] = np.dot(verts[:, :2], rotation_matrix[:2, :2])  # Rotate only x and y, keep z as is

    return rotated_verts


def translate_to_origin(verts):
    """Translate the floor to the origin by shifting the center of the floor to the origin."""
    # Find the floor vertices (those with z=0)
    floor_verts = verts[verts[:, 2] == 0]

    # Calculate the min and max for x and y coordinates of the floor
    min_x, min_y = floor_verts[:, :2].min(axis=0)
    max_x, max_y = floor_verts[:, :2].max(axis=0)

    # Calculate the center of the floor based on min/max coordinates
    floor_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, 0])

    # Calculate the translation vector (move the center to origin)
    translation_vector = -floor_center  # Shift by the negative of the center

    # Apply the translation to all vertices
    translated_verts = verts + translation_vector  # Shift all vertices by the translation vector

    return translated_verts


def load_scene(scan_path, show_small_objects, show_bbox):
    """Load a 3D scene with optional small objects and bounding boxes."""
    show_list = []
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"Scan path not found: {scan_path}")

    for obj_name in os.listdir(scan_path):
        # Skip small objects if the flag is not set
        if '-' in obj_name and not show_small_objects:
            continue

        obj_path = os.path.join(scan_path, obj_name, f'{obj_name}.obj')
        if not os.path.exists(obj_path):
            print(f"Warning: {obj_path} not found, skipping.")
            continue

        mesh = trimesh.load(obj_path)
        show_list.append(mesh)

        # Add bounding box if enabled
        if show_bbox:
            obb = mesh.bounding_box  # Oriented Bounding Box (OBB)
            bbox = trimesh.creation.box(extents=obb.extents)  # Create box mesh

            # Convert bbox to wireframe by extracting edges
            bbox_edges = trimesh.load_path(bbox.vertices[bbox.edges_unique])

            # Apply the transformation of the OBB
            bbox_edges.apply_transform(obb.primitive.transform)

            # Add wireframe bbox to the scene
            show_list.append(mesh)


    return show_list


def load_layout_heuristic(json_path, wall_height=3.0):
    """Load and visualize floor and walls from layout.json using trimesh."""


    # Load JSON file
    with open(json_path, 'r') as f:
        layout_data = json.load(f)

    verts = np.array(layout_data["vert"])  # Vertices (corner points)
    quads = layout_data["quads"]  # Face indices (quad indices)

    show_list = []

    ## --- Step 1: Create the Floor ---
    # Compute floor center and size
    min_xy = verts.min(axis=0)
    max_xy = verts.max(axis=0)
    floor_size = max_xy - min_xy

    # Create floor quadrilateral from the first four vertices
    floor_vertices = np.array([
        [verts[0][0], verts[0][1], 0],  # Bottom-left
        [verts[1][0], verts[1][1], 0],  # Bottom-right
        [verts[2][0], verts[2][1], 0],  # Top-right
        [verts[3][0], verts[3][1], 0]  # Top-left
    ])

    # Define two triangles to form a rectangular plane
    floor_faces = np.array([
        [0, 1, 2],  # First triangle
        [0, 2, 3]  # Second triangle
    ])

    # Create the floor mesh
    floor = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)

    # Apply floor translation (center it)
    show_list.append(floor)

    ## --- Step 2: Create the Walls ---
    for i in range(len(verts)):
        p1, p2 = verts[i], verts[(i + 1) % len(verts)]  # Get consecutive edge points
        edge_center = (p1 + p2) / 2  # Wall center
        edge_length = np.linalg.norm(p2 - p1)  # Wall length

        # Create wall using two triangles for each wall face (quad with wall height)
        wall_vertices = np.array([
            [p1[0], p1[1], 0],  # Bottom-left
            [p2[0], p2[1], 0],  # Bottom-right
            [p2[0], p2[1], wall_height],  # Top-right
            [p1[0], p1[1], wall_height]  # Top-left
        ])

        # Define two triangles to form a rectangular wall plane
        wall_faces = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3]  # Second triangle
        ])

        # Create the wall mesh
        wall = trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces)

        show_list.append(wall)

    return show_list


def load_layout_model(json_path, wall_height=3.0):
    """Load and visualize floor and walls from layout.json using trimesh."""

    # Load JSON file
    with open(json_path, 'r') as f:
        layout_data = json.load(f)

    verts = np.array(layout_data["verts"])  # Vertices (corner points)
    quads = layout_data["quads"]  # Face indices (quad indices)

    verts = translate_to_origin(verts)
    verts = rotate_z_90(verts)


    show_list = []

    # ## --- Step 1: Create the Floor ---
    #
    # # Create the floor using the vertices with z=0
    # # Loop through quads and find the ones with z=0 for floor creation
    # for quad in quads:
    #     # Check if the quad consists of floor vertices (z=0)
    #     if np.all(verts[quad][:, 2] == 0.0):
    #         quad_vertices = verts[quad]  # Get the floor vertices
    #         print('quad_vertices ', quad_vertices)
    #
    #         # Use Delaunay triangulation to split the quad into triangles
    #         floor_2d = quad_vertices[:, :2]  # Only consider x and y for triangulation
    #         delaunay = Delaunay(floor_2d)
    #
    #         # Extract the faces from Delaunay triangulation
    #         floor_faces = delaunay.simplices
    #
    #         # Create the floor geometry
    #         floor = trimesh.Trimesh(vertices=quad_vertices, faces=floor_faces)
    #         scene.add_geometry(floor)

    ## --- Step 2: Create the Walls ---
    for i in range(len(verts)):
        p1, p2 = verts[i], verts[(i + 1) % len(verts)]  # Get consecutive edge points
        edge_center = (p1 + p2) / 2  # Wall center
        edge_length = np.linalg.norm(p2 - p1)  # Wall length

        # Create wall vertices (quad with height)
        wall_vertices = np.array([
            [p1[0], p1[1], 0],  # Bottom-left
            [p2[0], p2[1], 0],  # Bottom-right
            [p2[0], p2[1], wall_height],  # Top-right
            [p1[0], p1[1], wall_height]  # Top-left
        ])

        # Define two triangles to form the rectangular wall plane
        wall_faces = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3]  # Second triangle
        ])

        # Create the wall mesh
        wall = trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces)

        # Apply wall translation (position the wall in the scene)
        # wall.apply_translation([edge_center[0], edge_center[1], wall_height / 2])
        show_list.append(wall)

    return show_list


# Function to parse command-line arguments
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize a 3D scene with optional bounding boxes and small objects.")

    # Argument: scan, used to specify the scene ID to visualize (e.g., scene0001_00)
    parser.add_argument('--scan', required=True, help="Scene ID to visualize (e.g., scene0001_00).")

    # Argument: save_path, specifies the base directory containing the annotation results
    parser.add_argument('--save_path', required=True, help="Base directory containing the annotation results.")

    # Argument: layout_path_heuristic, the path to the heuristic-based layout
    parser.add_argument('--layout_path_heuristic', required=True, help="Path to heuristic-based layout.")

    # Argument: layout_path_model, the path to the model-based layout
    parser.add_argument('--layout_path_model', required=True, help="Path to model-based layout.")

    # Argument: show_small_objects, a boolean flag to include small objects (filenames containing '-')
    parser.add_argument('--show_small_objects', action='store_true',
                        help="Include small objects (filenames containing '-').")

    # Argument: show_bbox, a boolean flag to display bounding boxes for objects
    parser.add_argument('--show_bbox', action='store_true', help="Display bounding boxes for objects.")

    # Argument: layout_type, specifies the type of layout to load: heuristic-based or model-based
    parser.add_argument('--layout_type', choices=['heuristic', 'model'], default='model',
                        help="Type of layout to load: 'heuristic' or 'model' (default: 'model').")

    return parser.parse_args()


# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Get the scan path
    scan_path = os.path.join(args.save_path, args.scan)

    # Load the object show list based on whether to include small objects and bounding boxes
    object_show_list = load_scene(scan_path, args.show_small_objects, args.show_bbox)

    # Choose layout loading method based on the layout_type argument
    if args.layout_type == 'heuristic':
        layout_show_list = load_layout_heuristic(args.layout_path_heuristic)
    elif args.layout_type == 'model':
        layout_show_list = load_layout_model(args.layout_path_model)
    else:
        layout_show_list = []
        assert 'Error Type'

    # Create a new scene
    scene = trimesh.Scene()

    # Add object geometries to the scene
    scene.add_geometry(object_show_list)

    # Add layout geometries to the scene
    scene.add_geometry(layout_show_list)

    # Show the scene
    scene.show()