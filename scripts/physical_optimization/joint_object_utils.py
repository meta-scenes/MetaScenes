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


def select_joint_groups(support_info_one_scan):
    """
    Selects the largest object group for each support or embed relationship.

    Parameters:
    support_info_one_scan (dict): Scene support information.

    Returns:
    dict: Dictionary containing the largest object group for each relationship type.
    """
    max_obj_group = {'support': {}, 'embed': {}}

    for one_rel in ['support', 'embed']:
        for src_id, tgt_list in support_info_one_scan[one_rel].items():
            src_base_id = src_id.split('_')[0] if '_' in src_id else src_id

            if src_base_id not in max_obj_group[one_rel] or len(max_obj_group[one_rel][src_base_id]) < len(tgt_list):
                max_obj_group[one_rel][src_base_id] = tgt_list

    print('max_obj_group:', max_obj_group)
    return max_obj_group


def get_joint_objs(max_obj_group, ssg, ssg_sm, texture_list):
    """
    Processes objects based on support, inside, and embed relationships.

    Parameters:
    max_obj_group (dict): Largest object groups for relationships.
    ssg (dict): Scene relationship structure.
    ssg_sm (dict): Secondary scene relationship structure.
    texture_list (list): List of objects with special textures.

    Returns:
    tuple: (List of valid object IDs, Dictionary of joint relationships)
    """
    support_dict, fix_dict, embed_dict = {}, {}, {}

    # Collect relationships
    for src, tgt, rel in ssg['relationship'] + ssg_sm['relationship']:
        if rel == 'support':
            support_dict.setdefault(src, []).append(tgt)
        elif rel == 'inside':
            fix_dict.setdefault(src, []).append(tgt)
        else:  # embed relationship
            embed_dict.setdefault(src, []).append(tgt)

    print(support_dict, fix_dict, embed_dict)
    joint_dict = {}
    valid_ids = []

    def process_relationship(src_id, tgt_obj_list, rel_type):
        """
        Processes relationships by merging target objects into the source object.

        Parameters:
        src_id (str): Source object ID.
        tgt_obj_list (list): List of target object IDs.
        rel_type (str): Relationship type ('support', 'inside', or 'embed').
        """
        if src_id == 'planes' or src_id not in bpy.data.objects:
            return

        valid_ids.append(src_id)
        bpy.data.objects[src_id].select_set(True)
        joint_dict.setdefault(src_id, [])

        fake_src_id = src_id  # Default to src_id, but may change based on texture presence

        for tgt_inst_id in tgt_obj_list:
            if tgt_inst_id not in bpy.data.objects:
                continue

            # Ensure only valid objects are merged
            if rel_type in max_obj_group and tgt_inst_id not in max_obj_group[rel_type].get(src_id, []):
                bpy.data.objects.remove(bpy.data.objects[tgt_inst_id], do_unlink=True)
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

    # Process each relationship type
    for src_id, tgt_obj_list in support_dict.items():
        process_relationship(src_id, tgt_obj_list, 'support')

    for src_id, tgt_obj_list in fix_dict.items():
        process_relationship(src_id, tgt_obj_list, 'inside')

    for src_id, tgt_obj_list in embed_dict.items():
        process_relationship(src_id, tgt_obj_list, 'embed')

    return valid_ids, joint_dict
