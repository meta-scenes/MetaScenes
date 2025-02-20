
import bpy
import bmesh
from mathutils import Vector
import json
import os


WALL_THICKNESS = 0.1
WALL_HEIGHT = 3
WALL_TEXTURE_PATH = '/mnt/fillipo/huangyue/recon_sim/layout_textures/wall/carpet_Paper002_2K_Color_crop0.jpg'
FLOOR_TEXTURE_PATH = '/mnt/fillipo/huangyue/recon_sim/layout_textures/floor/wood_Wood026_1K-PNG_crop0.jpg'

def create_mesh(verts, mesh_name):
    # 创建一个新的Mesh对象
    mesh_data = bpy.data.meshes.new(mesh_name)
    mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)

    # 将新创建的对象添加到场景中
    bpy.context.collection.objects.link(mesh_obj)

    # 创建一个BMesh对象并将顶点数据加入到BMesh中
    bm = bmesh.new()

    # 添加顶点
    bm_verts = []
    for vert in verts:
        bm_verts.append(bm.verts.new(vert))

    bm.verts.ensure_lookup_table()

    # 添加四边形
    bm.faces.new(bm_verts)

    # 更新BMesh并写入到Mesh数据中
    bm.to_mesh(mesh_data)
    bm.free()

    # 更新Mesh以显示在视图中
    mesh_obj.data.update()

    add_thickness(mesh_obj, thickness=WALL_THICKNESS)

    mesh_obj.select_set(True)


def add_thickness(obj, thickness):
    # 进入对象模式并选中对象
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # 确保处于对象模式
    bpy.ops.object.mode_set(mode='OBJECT')

    # 应用所有变换
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 添加固化修改器
    solidify_modifier = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_modifier.thickness = thickness
    solidify_modifier.offset = 0

    # 应用修改器
    bpy.ops.object.modifier_apply(modifier=solidify_modifier.name)

    # 确保处于对象模式
    bpy.ops.object.mode_set(mode='OBJECT')



def add_texture_to_cube(cube, texture_file):
    # 创建新的材质
    material = bpy.data.materials.new(name="CubeMaterial")
    material.use_nodes = True

    # 获取材质节点树
    nodes = material.node_tree.nodes

    # 删除默认的Diffuse节点
    nodes.clear()

    # 添加图像纹理节点
    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.image = bpy.data.images.load(
        filepath=texture_file)

    # 添加Principled BSDF节点
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')

    # 添加材质输出节点
    output_node = nodes.new(type='ShaderNodeOutputMaterial')

    # 链接节点
    material.node_tree.links.new(texture_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    material.node_tree.links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # 将材质分配给Cube
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