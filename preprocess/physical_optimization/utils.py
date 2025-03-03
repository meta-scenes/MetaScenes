

import numpy as np
import bpy

from mathutils import Vector, Quaternion, Matrix



def is_point_in_polygon(point, polygon):
    """
    使用射线法判断一个点是否在多边形内。
    point: 要检查的点，Vector2D类型 (x, y)
    polygon: 多边形的顶点列表，List[Vector2D]
    """
    num = len(polygon)
    j = num - 1
    odd_nodes = False

    for i in range(num):
        vi = polygon[i]
        vj = polygon[j]

        if ((vi.y < point.y and vj.y >= point.y) or (vj.y < point.y and vi.y >= point.y)) and \
                (vi.x + (point.y - vi.y) / (vj.y - vi.y) * (vj.x - vi.x) < point.x):
            odd_nodes = not odd_nodes
        j = i

    return odd_nodes


def get_polygon_from_mesh(mesh_obj, layout_type):
    """
    从给定的Mesh对象中获取一个平面上的多边形。
    假设Mesh是一个平面，只有一个面片。
    """
    polygon = []
    mesh = mesh_obj.data

    if layout_type == 'model':
        face = mesh.polygons[0]
    elif layout_type == 'heuristic':
        face = mesh.polygons[4]
    else:
        assert False

    for vert_idx in face.vertices:
        # 获取世界坐标系下的顶点位置
        vert_co = mesh_obj.matrix_world @ mesh.vertices[vert_idx].co
        polygon.append(vert_co.xy)  # 获取XY平面的坐标

    return polygon


def is_inside(polygon, mesh_obj):
    # 计数在面片内的顶点数
    count = 0
    count_mesh = 0
    # 遍历mesh_obj的顶点，判断它们是否在polygon中
    vertice_cnt = len(mesh_obj.data.vertices)
    step = int(vertice_cnt / 1000) + 1

    for vert_idx, vertex in enumerate(mesh_obj.data.vertices):
        if vert_idx % step != 0: continue
        point = mesh_obj.matrix_world @ vertex.co
        point = point.xy  # 获取顶点的XY坐标
        if is_point_in_polygon(point, polygon):
            # print (point , 'inside')
            count += 1
        count_mesh += 1

    # print(f"Number of vertices inside the polygon: {count}")
    # print(count / len(mesh_obj.data.vertices))

    return count / count_mesh > 0.8



def is_touching(obj1, obj2):
    """
    检查两个对象的包围盒是否接触。
    :param obj1: 第一个对象
    :param obj2: 第二个对象
    :return: 如果两个对象的包围盒接触，返回 True，否则返回 False。
    """

    bbox1 = [obj1.matrix_world @ Vector(corner) for corner in obj1.bound_box]
    bbox2 = [obj2.matrix_world @ Vector(corner) for corner in obj2.bound_box]

    bbox1_min = round_vector(Vector((min(v[0] for v in bbox1), min(v[1] for v in bbox1), min(v[2] for v in bbox1))))
    bbox1_max = round_vector(Vector((max(v[0] for v in bbox1), max(v[1] for v in bbox1), max(v[2] for v in bbox1))))

    bbox2_min = round_vector(Vector((min(v[0] for v in bbox2), min(v[1] for v in bbox2), min(v[2] for v in bbox2))))
    bbox2_max = round_vector(Vector((max(v[0] for v in bbox2), max(v[1] for v in bbox2), max(v[2] for v in bbox2))))

    # print ('Calulate touching ...')
    # print (bbox1_min, bbox1_max, bbox2_min, bbox2_max)
    # print((bbox1_min.x <= bbox2_max.x and bbox1_max.x >= bbox2_min.x))
    # print((bbox1_min.y <= bbox2_max.y and bbox1_max.y >= bbox2_min.y))
    # print((bbox1_min.z <= bbox2_max.z and bbox1_max.z >= bbox2_min.z))

    return (bbox1_min.x <= bbox2_max.x and bbox1_max.x >= bbox2_min.x) and \
        (bbox1_min.y <= bbox2_max.y and bbox1_max.y >= bbox2_min.y) and \
        (bbox1_min.z <= bbox2_max.z and bbox1_max.z >= bbox2_min.z)

def round_vector(vec, decimals=3):
    return Vector([round(v, decimals) for v in vec])

def matrix_difference(m1, m2):
    """
    计算两个矩阵之间的差异，位置和旋转部分。
    :param m1: 第一个矩阵
    :param m2: 第二个矩阵
    :return: 位置差异和旋转差异
    """
    pos_diff = np.linalg.norm(np.array(m1.translation) - np.array(m2.translation))
    rot_diff = np.linalg.norm(m1.to_3x3() - m2.to_3x3())
    return pos_diff, rot_diff



def is_intersection(obj1, obj2):
    # obj1 is wall, obj2 is target object
    new_obj = obj2.copy()
    new_obj.data = obj2.data.copy()
    bpy.context.collection.objects.link(new_obj)

    bool_mod = new_obj.modifiers.new(type="BOOLEAN", name="bool_mod")
    bool_mod.operation = 'INTERSECT'
    bool_mod.object = obj1


    bpy.context.view_layer.objects.active = new_obj
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)

    if len(new_obj.data.polygons) == 0:
        bpy.data.objects.remove(new_obj, do_unlink=True)
        return False
    else:
        bpy.data.objects.remove(new_obj, do_unlink=True)
        return True


def set_origin_obj(obj):

    obj.select_set(True)
    mat = obj.matrix_world
    center = np.mean(obj.bound_box, axis=0)  # local
    center = center.reshape((3, 1))
    center_world = (np.array(mat)[:3, :3] @ center + np.array(mat)[:3, [3]]).reshape(-1)
    # set origin
    bpy.context.scene.cursor.location = center_world
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    if abs(obj.rotation_euler[0]) < np.pi / 2 / 3:
        obj.rotation_euler[0] = 0
    if abs(obj.rotation_euler[1]) < np.pi / 2 / 3:
        obj.rotation_euler[1] = 0
    # bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    obj.select_set(False)