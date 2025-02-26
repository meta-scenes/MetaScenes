
import bpy


class BoundingBox:
    def __init__(self, cx, cy, cz, width, height, depth):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.width = width
        self.height = height
        self.depth = depth

    @property
    def x_min(self):
        return self.cx - self.width / 2

    @property
    def y_min(self):
        return self.cy - self.height / 2

    @property
    def z_min(self):
        return self.cz - self.depth / 2

    @property
    def x_max(self):
        return self.cx + self.width / 2

    @property
    def y_max(self):
        return self.cy + self.height / 2

    @property
    def z_max(self):
        return self.cz + self.depth / 2

    def volume(self):
        return self.width * self.height * self.depth

    def intersection_volume(self, other):
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        z_min = max(self.z_min, other.z_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        z_max = min(self.z_max, other.z_max)

        if x_min < x_max and y_min < y_max and z_min < z_max:
            return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        else:
            return 0

def compute_collision_loss(bboxes):
    total_volume = sum([bbox.volume() for bbox in bboxes])
    total_intersection_volume = 0

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            total_intersection_volume += bboxes[i].intersection_volume(bboxes[j])

    collision_loss = total_intersection_volume / total_volume
    return collision_loss



def get_one_move(obj, case, step = 0.05):

    if case == 1:
        obj.location.x += step
    elif case == 2:
        obj.location.x -= step
    elif case == 3:
        obj.location.y += step
    elif case == 4:
        obj.location.y -= step

    return 0

def get_one_move_back(obj, case, step = 0.05):

    if case == -1:
        obj.location.x -= step
    elif case == -2:
        obj.location.x += step
    elif case == -3:
        obj.location.y -= step
    elif case == -4:
        obj.location.y += step

    return 0


def update_collision_loss(src_obj):
    bboxes = []
    obj_list = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.name not in ['Cube', 'planes'] and obj!= src_obj and 'geometry' not in obj.name:

            bboxes.append(BoundingBox(obj.location.x, obj.location.y,obj.location.z, obj.dimensions[0], obj.dimensions[1], obj.dimensions[2] ))
            obj_list.append(obj.name)
    if len(bboxes) <= 1:
        return [], 0

    init_collision_loss = compute_collision_loss(bboxes)

    return obj_list, init_collision_loss
