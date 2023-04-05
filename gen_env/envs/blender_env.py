import bpy
import gym
import random

import numpy as np
from gen_env.envs.blender.utils import delete_scene_objects

from gen_env.envs.play_env import PlayEnv
from gen_env.tiles import colors

class BlenderRender(gym.Wrapper):
    def __init__(self, env: PlayEnv):
        super().__init__(env)
        self.env = env

        self.mats = {}
        for c_name in colors:
            mat = bpy.data.materials.new(name=c_name)
            mat.diffuse_color = tuple(colors[c_name]) + (1.0,)
            self.mats[c_name] = mat

        scene = bpy.context.scene

        # Add some basic lighting to the scene
        light_data = bpy.data.lights.new(name="Light", type='SUN')
        light = bpy.data.objects.new(name="Light", object_data=light_data)
        scene.collection.objects.link(light)
        light.location = (0, 0, 10)
        self.sun = light 

    def is_done(self):
        return self.env._done


    def render(self, mode='human'):
        delete_scene_objects(bpy.context.scene, exclude={self.sun})
        # Place a cube wherever there is a wall
        for tile in self.env.tiles:
            bin_arr = self.map[tile.idx]
            for x, y in np.argwhere(bin_arr):
                x, y = x * 2, y * 2
                z = tile.idx * 2
                self.draw_cube(x, y, z, tile.color_name)

            if tile.is_player:
                bin_arr = self.map[tile.idx]
                for x, y in np.argwhere(bin_arr):
                    x, y = x * 2, y * 2
                    self.draw_cone(x, y, 2, tile.color)

    def draw_cone(self, x, y, z, color):
        # Create a new pyramid
        bpy.ops.mesh.primitive_cone_add(location=(x, y, z))
        # Get the new pyramid
        new_pyramid = bpy.context.object
        # Set the color
        # new_pyramid.data.materials.append(color)

    def draw_cube(self, x, y, z, color):
        # Create a new cube
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z))
        # Get the new cube
        new_cube = bpy.context.object
        # Set the color
        mat = self.mats[color]
        new_cube.active_material = mat