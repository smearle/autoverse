import bpy

def delete_scene_objects(scene=None, exclude={}):
    """Delete a scene and all its objects."""
    # if not scene:
        # Use current scene if no argument given
        # scene = bpy.context.scene
    # Select all objects in the scene
    for obj in scene.objects:
        if obj not in exclude:
            obj.select_set(True)
    # Delete selected objects
    bpy.ops.object.delete()
    # Remove orphaned data blocks
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    