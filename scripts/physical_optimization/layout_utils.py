
import bpy
import bmesh
import json
import os


WALL_THICKNESS = 0.1
WALL_HEIGHT = 3
WALL_TEXTURE_PATH = '/mnt/fillipo/huangyue/recon_sim/layout_textures/wall/carpet_Paper002_2K_Color_crop0.jpg'
FLOOR_TEXTURE_PATH = '/mnt/fillipo/huangyue/recon_sim/layout_textures/floor/wood_Wood026_1K-PNG_crop0.jpg'


def create_mesh(verts, mesh_name):
    """
    Creates a new mesh object from a list of vertices and applies thickness.

    Parameters:
    - verts (list of tuple): List of vertex coordinates (x, y, z).
    - mesh_name (str): Name of the new mesh object.

    Returns:
    - bpy.types.Object: The created mesh object.
    """
    # Create a new mesh and object
    mesh_data = bpy.data.meshes.new(mesh_name)
    mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)

    # Add the object to the scene
    bpy.context.collection.objects.link(mesh_obj)

    # Create a BMesh object and add vertices
    bm = bmesh.new()
    bm_verts = [bm.verts.new(vert) for vert in verts]

    bm.verts.ensure_lookup_table()

    # Create a face using the vertices
    bm.faces.new(bm_verts)

    # Update the BMesh and write to mesh data
    bm.to_mesh(mesh_data)
    bm.free()

    # Update mesh to appear in the viewport
    mesh_obj.data.update()

    # Apply thickness to the mesh
    add_thickness(mesh_obj, thickness=WALL_THICKNESS)

    mesh_obj.select_set(True)

    return mesh_obj


def add_thickness(obj, thickness):
    """
    Applies a solidify modifier to add thickness to an object.

    Parameters:
    - obj (bpy.types.Object): The object to modify.
    - thickness (float): The thickness value.

    Returns:
    - None
    """
    # Ensure the object is selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Switch to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Apply all transformations
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Add a solidify modifier
    solidify_modifier = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_modifier.thickness = thickness
    solidify_modifier.offset = 0

    # Apply the modifier
    bpy.ops.object.modifier_apply(modifier=solidify_modifier.name)

    # Ensure the object remains in object mode
    bpy.ops.object.mode_set(mode='OBJECT')



def add_texture_to_cube(cube, texture_file):
    """
    Assigns a texture to a cube using a material with a Principled BSDF shader.

    Parameters:
    - cube (bpy.types.Object): The cube object to apply the texture to.
    - texture_file (str): Path to the texture file.

    Returns:
    - None
    """
    # Create a new material with nodes enabled
    material = bpy.data.materials.new(name="CubeMaterial")
    material.use_nodes = True

    # Get the material node tree
    nodes = material.node_tree.nodes

    # Clear default nodes
    nodes.clear()

    # Create an image texture node and load the image
    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.image = bpy.data.images.load(filepath=texture_file)

    # Create a Principled BSDF shader node
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Create a material output node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')

    # Link texture color to the BSDF shader base color
    material.node_tree.links.new(texture_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    # Link BSDF shader to the material output
    material.node_tree.links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Assign the material to the cube
    if cube.data.materials:
        cube.data.materials[0] = material
    else:
        cube.data.materials.append(material)


def create_wall_cube(length, location, name, dir, padding, wall_thickness, wall_height):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    wall_cube = bpy.context.object
    wall_cube.name = name

    if dir == 'x':
        wall_cube.scale = (length + padding, wall_thickness / 2, wall_height)
    else:
        wall_cube.scale = (wall_thickness / 2, length + padding, wall_height)

    add_texture_to_cube(wall_cube,
                        WALL_TEXTURE_PATH)



def roomplane_layout(one_scan, layout_path, scale_rate=1.01):
    one_scan_layout = json.load(open(os.path.join(layout_path, f'{one_scan}.json'), 'rb'))
    verts = one_scan_layout['verts']
    quads = one_scan_layout['quads']

    for i, quad in enumerate(quads):
        vert = [verts[idx] for idx in quad]
        for one_vert in vert:
            if one_vert[2] > 0.5: one_vert[2] = WALL_HEIGHT  # wall_high

        create_mesh(vert, f'geometry_{i}')
        if i == 0: continue
        wall_cube = bpy.data.objects.get(f'geometry_{i}')
        bpy.context.view_layer.objects.active = wall_cube
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0)
        bpy.ops.object.mode_set(mode='OBJECT')
        add_texture_to_cube(wall_cube,
                            WALL_TEXTURE_PATH)

    bpy.ops.transform.resize(value=(scale_rate, scale_rate, scale_rate))
    bpy.ops.transform.rotate(value=-1.5708, orient_axis='Z')

    planes = bpy.data.objects.get('geometry_0')
    planes.name = 'planes'
    assert planes is not None

    bpy.context.view_layer.objects.active = planes
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0)
    bpy.ops.object.mode_set(mode='OBJECT')
    add_texture_to_cube(planes,
                        WALL_TEXTURE_PATH)
    add_texture_to_cube(planes, FLOOR_TEXTURE_PATH)


def heuristic_layout(floor_data, padding=0.1):
    """
    Constructs a floor and walls in Blender using the given floor data.

    Parameters:
    floor_data (dict): Floor data containing vertices and quads.
    padding (float): Extra padding for floor expansion.

    The function creates a floor plane and four surrounding walls in Blender.
    """

    # Extract floor vertices
    vert = floor_data["vert"]

    # Compute floor boundaries
    min_x, min_y = vert[0][0], vert[0][1]
    max_x, max_y = vert[2][0], vert[2][1]

    # Compute cube dimensions and position
    cube_width = max_x - min_x
    cube_length = max_y - min_y
    cube_height = 0.05  # Floor thickness
    cube_x = (min_x + max_x) / 2
    cube_y = (min_y + max_y) / 2

    # Create the floor cube
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cube_x, cube_y, -cube_height / 2))
    cube = bpy.context.object
    cube.scale = (cube_width + padding, cube_length + padding, cube_height)
    cube.name = 'planes'

    # Apply texture to the floor
    add_texture_to_cube(cube, FLOOR_TEXTURE_PATH)

    # Define wall parameters
    wall_height = WALL_HEIGHT
    wall_thickness = WALL_THICKNESS

    # Create walls around the floor
    create_wall_cube(cube_length, (min_x - wall_thickness / 2, cube_y, wall_height / 2), 'geometry_1', 'y', padding,
                     wall_thickness, wall_height)  # Left wall

    create_wall_cube(cube_length, (max_x + wall_thickness / 2, cube_y, wall_height / 2), 'geometry_2', 'y', padding,
                     wall_thickness, wall_height)  # Right wall

    create_wall_cube(cube_width, (cube_x, min_y - wall_thickness / 2, wall_height / 2), 'geometry_3', 'x', padding,
                     wall_thickness, wall_height)  # Front wall

    create_wall_cube(cube_width, (cube_x, max_y + wall_thickness / 2, wall_height / 2), 'geometry_4', 'x', padding,
                     wall_thickness, wall_height)  # Back wall