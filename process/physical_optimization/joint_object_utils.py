import bpy


def get_joint_vert_group(joint_dict, src_id):
    """
    Creates and assigns a vertex group for objects in a joint relationship.

    Parameters:
    joint_dict (dict): Dictionary containing joint relationships.
    src_id (str): Source object ID.
    """
    for one_obj_id in joint_dict[src_id]:
        group_name = f'vert_{one_obj_id}'
        if one_obj_id not in bpy.data.objects:
            continue

        one_obj = bpy.data.objects[one_obj_id]
        one_obj.select_set(True)
        bpy.context.view_layer.objects.active = one_obj
        bpy.ops.object.mode_set(mode='OBJECT')

        # Assign all vertices to a new vertex group
        vg1 = one_obj.vertex_groups.new(name=group_name)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.mode_set(mode='OBJECT')  # Switch back to object mode


def process_relationship(src_id, tgt_obj_list, texture_list):
    """
    Processes relationships by merging target objects into the source object.

    Parameters:
    - src_id (str): Source object ID.
    - tgt_obj_list (list): List of target object IDs.
    - texture_list (list): List of objects with special textures.

    Returns:
    - tuple: (List of valid object IDs, Dictionary of joint relationships)
    """

    if src_id == 'planes' or src_id not in bpy.data.objects:
        return [], {}

    joint_dict = {src_id: []}
    valid_ids = [src_id]

    bpy.data.objects[src_id].select_set(True)
    fake_src_id = src_id  # Default to src_id, but may change based on texture presence

    for tgt_inst_id in tgt_obj_list:
        if tgt_inst_id not in bpy.data.objects:
            continue

        if tgt_inst_id in texture_list:
            fake_src_id = tgt_inst_id

        bpy.data.objects[tgt_inst_id].select_set(True)
        joint_dict[src_id].append(tgt_inst_id)
        valid_ids.append(tgt_inst_id)

    get_joint_vert_group(joint_dict, src_id)
    bpy.context.view_layer.objects.active = bpy.data.objects[fake_src_id]
    bpy.ops.object.join()
    bpy.data.objects[fake_src_id].select_set(False)
    bpy.data.objects[fake_src_id].name = src_id
    bpy.data.objects[src_id].data.name = src_id

    return valid_ids, joint_dict


def get_joint_objs(ssg, texture_list):
    """
    Processes objects based on support, inside, and embed relationships.

    Parameters:
    - ssg (dict): Scene relationship structure.
    - texture_list (list): List of objects with special textures.

    Returns:
    - tuple: (List of valid object IDs, Dictionary of joint relationships)
    """

    relationship_types = {
        'support': {},
        'inside': {},
        'embed': {}
    }

    # Collect relationships
    for src, tgt, rel in ssg.get('relationship', []):
        if rel in relationship_types:
            relationship_types[rel].setdefault(src, []).append(tgt)

    print(f"| Support dict: {relationship_types['support']}")
    print(f"| Fix dict: {relationship_types['inside']}")
    print(f"| Embed dict: {relationship_types['embed']}")

    valid_ids_all = []
    joint_dict_all = {}

    # Process each relationship type
    for rel_type in ['support', 'inside', 'embed']:
        for src_id, tgt_obj_list in relationship_types[rel_type].items():
            valid_ids, joint_dict = process_relationship(src_id, tgt_obj_list, texture_list)
            valid_ids_all.extend(valid_ids)
            joint_dict_all.update(joint_dict)

    return valid_ids_all, joint_dict_all