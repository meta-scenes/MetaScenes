
import numpy as np
import bpy

from mathutils import Vector, Quaternion, Matrix

import math
import sys
sys.path.append('/home/huangyue/Mycodes/MetaScenes/scripts/physical_optimization')
from utils import is_touching, is_intersection
from layout_utils import WALL_HEIGHT
from add_physics_utils import create_rigidbody_constraint, add_one_obj_rigid_body


def refine_hanging(hanging_obj_list, layout_type):
    hanging_obj_with_wall = {}
    walls_list = []
    delete_objs_cnt = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name.startswith("geometry_"):
            if obj.location.z > WALL_HEIGHT * 2/3: continue
            walls_list.append(obj)

    for one_obj_name in hanging_obj_list:
        if one_obj_name not in bpy.data.objects: continue
        hanging_obj = bpy.data.objects[one_obj_name]
        rotate_one_hanging_obj(hanging_obj)

        obj_ori = get_obj_orientation(hanging_obj)
        coli_wall = check_collision(hanging_obj, walls_list, obj_ori, layout_type)

        if coli_wall is None:
            bpy.data.objects.remove(hanging_obj, do_unlink=True)
            delete_objs_cnt+=1

            continue

        hanging_obj_with_wall[hanging_obj.name] = {
            'wall': coli_wall,
            'obj_ori': obj_ori
        }
        print('Find wall ', hanging_obj, obj_ori, coli_wall)

    if layout_type == 'model':
        move_to_wall(hanging_obj_with_wall)
    else:
        for hanging_obj in hanging_obj_with_wall:
            one_wall = hanging_obj_with_wall[hanging_obj]['wall']
            create_rigidbody_constraint(
                obj1=bpy.data.objects[one_wall],
                obj2=bpy.data.objects[hanging_obj],
                constraint_type='GENERIC',
                use_limit_lin=(
                    [-0.2, 0.2],
                    [-0.2, 0.2],
                    [0, 0]
                ),
                use_limit_ang=(
                    [0,0],
                    [0,0],
                    None
                )
            )
            add_one_obj_rigid_body(bpy.data.objects[hanging_obj], 'ACTIVE')


    return delete_objs_cnt


def rotate_one_hanging_obj(obj):
    """
    Finds the optimal rotation around the Y-axis for a hanging object to minimize its bounding box volume.

    This function iterates through a range of rotation angles around the Y-axis, creating a temporary
    copy of the object for each angle, applying the transformation, and computing the bounding box volume.
    The angle that results in the smallest bounding box volume is then applied to the original object.

    Parameters:
    - obj (bpy.types.Object): The Blender object to be rotated.

    Returns:
    - None (The function directly modifies the input object's rotation).
    """

    min_volume = float('inf')  # Initialize minimum volume with a large number
    y_res = 0  # Best rotation angle around Y-axis

    for y in range(-5, 5):
        # Create a temporary copy of the object
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        bpy.context.collection.objects.link(new_obj)

        # Apply rotation around the Y-axis
        new_obj.rotation_euler[0] = math.radians(y)
        new_obj.select_set(True)
        bpy.context.view_layer.objects.active = new_obj
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        # Compute the bounding box in world coordinates
        bound_box = [new_obj.matrix_world @ Vector(corner) for corner in new_obj.bound_box]
        bbox_min = [min(bound_box, key=lambda x: x[i])[i] for i in range(3)]
        bbox_max = [max(bound_box, key=lambda x: x[i])[i] for i in range(3)]
        bbox_size = [bbox_max[i] - bbox_min[i] for i in range(3)]

        # Compute volume of the bounding box
        volume = bbox_size[0] * bbox_size[1] * bbox_size[2]

        # Remove the temporary object
        bpy.data.objects.remove(new_obj, do_unlink=True)

        # Update the best rotation angle if a smaller volume is found
        if volume < min_volume:
            y_res = y
            min_volume = volume

    # Apply the optimal rotation to the original object
    obj.rotation_euler[0] = math.radians(y_res)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)



def hanging_init(one_scan, anno_info, inst_id_to_name):
    hanging_group = []
    for obj in bpy.data.objects:
        inst_id = obj.name
        if '-' in inst_id: continue
        if anno_info[one_scan][inst_id]['hanging']:
            hanging_group.append(inst_id)
            rotate_one_hanging_obj(bpy.data.objects[inst_id])
        elif inst_id_to_name[int(obj.name)] in ['mirror', 'window', 'whiteboard', 'picture', 'radiator', 'curtain']:
            hanging_group.append(inst_id)
            rotate_one_hanging_obj(bpy.data.objects[inst_id])
        else:
            continue

    return hanging_group


def get_obj_orientation(obj):

    mesh = obj.data
    vertices = [v.co for v in mesh.vertices]

    x_coords = np.array([v.x for v in vertices])
    y_coords = np.array([v.y for v in vertices])

    x_variance = np.var(x_coords)
    y_variance = np.var(y_coords)

    if x_variance < y_variance:
        return 'X'
    else:
        return 'Y'


def is_touch_with_walls(tgt_obj, layout_type):
    for wall in bpy.data.objects:
        if wall.type == 'MESH' and wall.name.startswith("geometry_"):  # 假设墙面对象以"geometry_"开头命名
            if wall.location.z > WALL_HEIGHT * 2/3: continue
            if layout_type == 'model':
                if is_touching(wall, tgt_obj):
                    return wall.name
            elif layout_type == 'heuristic':
                if is_intersection(wall, tgt_obj):
                    return wall.name

    return ''


def move_to_wall(hanging_obj_with_wall):
    for src_id in hanging_obj_with_wall:
        coli_wall_name = hanging_obj_with_wall[src_id]['wall']
        obj_ori = hanging_obj_with_wall[src_id]['obj_ori']
        src_obj = bpy.data.objects[src_id]
        coli_wall = bpy.data.objects[coli_wall_name]
        if obj_ori == 'Y':
            if coli_wall.location.y > 0:
                src_obj.location.y = coli_wall.location.y - coli_wall.dimensions[1] / 2 - src_obj.dimensions[1] / 2
            else:
                src_obj.location.y = coli_wall.location.y + coli_wall.dimensions[1] / 2 + src_obj.dimensions[1] / 2
        elif obj_ori == 'X':
            if coli_wall.location.x > 0:
                src_obj.location.x = coli_wall.location.x - coli_wall.dimensions[0] / 2 - src_obj.dimensions[0] / 2
            else:
                src_obj.location.x = coli_wall.location.x + coli_wall.dimensions[0] / 2 + src_obj.dimensions[0] / 2
        else:
            pass


def check_collision(target, walls, ori, layout_type):
    step_size = 0.1
    max_steps = 10

    if ori == 'Y':
        init_loc = target.location.copy().y
        target.dimensions.y = max(0.03, target.dimensions.y)
    else:
        init_loc = target.location.copy().x
        target.dimensions.x = max(0.03, target.dimensions.x)

    for step in range(max_steps):

        if ori == 'Y':
            target.location.y = init_loc + step * step_size
            bpy.context.view_layer.update()
            res = is_touch_with_walls(target, layout_type)

            if res != '':
                return res

            target.location.y = init_loc - step * step_size
            bpy.context.view_layer.update()
            res = is_touch_with_walls(target, layout_type)

            if res != '':
                return res

        else:
            target.location.x = init_loc + step * step_size
            bpy.context.view_layer.update()
            res = is_touch_with_walls(target, layout_type)

            if res != '':
                return res

            target.location.x = init_loc - step * step_size
            bpy.context.view_layer.update()
            res = is_touch_with_walls(target, layout_type)

            if res != '':
                return res

