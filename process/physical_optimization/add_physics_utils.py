import bpy


def set_constraint_limits(constraint, limits, limit_type):
    """Set the limits for a given constraint."""
    axes = ['x', 'y', 'z']
    for i, axis in enumerate(axes):
        limit_attr = f'use_limit_lin_{axis}' if limit_type == 'linear' else f'use_limit_ang_{axis}'
        limit_lower_attr = f'limit_lin_{axis}_lower' if limit_type == 'linear' else f'limit_ang_{axis}_lower'
        limit_upper_attr = f'limit_lin_{axis}_upper' if limit_type == 'linear' else f'limit_ang_{axis}_upper'

        # Check if the limits are specified (i.e., not False or None)
        if limits[i] is not None and limits[i] is not False:
            setattr(constraint, limit_attr, True)
            setattr(constraint, limit_lower_attr, limits[i][0])
            setattr(constraint, limit_upper_attr, limits[i][1])
        else:
            setattr(constraint, limit_attr, False)


def create_rigidbody_constraint(obj1, obj2, constraint_type='FIXED',
                                use_limit_lin=(None, None, None),
                                use_limit_ang=(None, None, None),
                                disable_collisions=False):

    constraint_name = f"constraint_{obj1.name}_{obj2.name}"

    if constraint_name not in bpy.data.objects:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=obj2.location)
        empty = bpy.context.object
        empty.name = constraint_name
    else:
        empty = bpy.data.objects[constraint_name]

    # Set the active object
    bpy.context.view_layer.objects.active = empty

    if constraint_type == 'FIXED':
        bpy.ops.rigidbody.constraint_add(type='FIXED')
        empty.rigid_body_constraint.object1 = obj1
        empty.rigid_body_constraint.object2 = obj2


    elif constraint_type == 'GENERIC':
        bpy.ops.rigidbody.constraint_add(type='GENERIC')
        empty.rigid_body_constraint.object1 = obj1
        empty.rigid_body_constraint.object2 = obj2
        constraint = empty.rigid_body_constraint

        # Set the constraints for linear and angular limits
        set_constraint_limits(constraint, use_limit_lin, 'linear')
        set_constraint_limits(constraint, use_limit_ang, 'angular')

        constraint.disable_collisions = disable_collisions

    bpy.context.view_layer.objects.active = None


def add_rigid_body():
    """Add rigid body physics to all objects except cameras and lights."""
    for obj in bpy.context.scene.objects:
        if obj.type == 'EMPTY' or obj.name in ['Camera', 'Light']:
            continue
        add_one_obj_rigid_body(obj, 'PASSIVE')


def add_one_obj_rigid_body(obj, rigid_type='PASSIVE'):
    """Add rigid body to a single object with specified type."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = rigid_type

    if rigid_type == 'ACTIVE':
        obj.rigid_body.collision_shape = 'CONVEX_HULL'
        obj.rigid_body.linear_damping = 1
        obj.rigid_body.angular_damping = 1
