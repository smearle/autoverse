import bpy
import gym
import random

from gen_env.envs.play_env import PlayEnv


class BlenderRender(gym.Wrapper):

    def render(self, mode='human'):
        # Generate random coordinates at which to place the cube
        x = random.randint(-5, 5)
        y = random.randint(-5, 5)
        z = random.randint(-5, 5)

        # Place a random cube in blender
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z))