#BOBOB
# Run this from blender to auto reload the current text file

import bpy

def reload(text):
    fp = bpy.path.abspath(text.filepath)
    text.clear()
    with open(fp) as f:
        text.write(f.read())
    return False

class DrawingClass:
    lock = False
    def __init__(self, context, prop):
        self.prop = prop
        self.handle = bpy.types.SpaceTextEditor.draw_handler_add(
                   self.draw_text_callback,(context,),
                   'WINDOW', 'POST_PIXEL')

    def draw_text_callback(self, context):
        if self.lock:
            return
        space = context.space_data
        text = space.text

        if text and text.is_modified:  # if is_modified(text):
            print("reloading text")
            bpy.ops.text.reload()
            '''
            self.lock + True
            self.lock = reload(text)
            '''

    def remove_handle(self):
         bpy.types.SpaceTextEditor.draw_handler_remove(self.handle, 'WINDOW')

if __name__ == "__main__":
    context = bpy.context             
    dc = DrawingClass(context, "Draw This On Screen")