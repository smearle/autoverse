# BEGIN HACK (If we can't sneak our conda env into blender, we can install necessary packages in blender's python env)
from functools import partial
import os
import sys
import subprocess
import time

# INSTALL = True
INSTALL = False

def install_requirements():
    """Install required packages from `requirements.txt`. Used when running scripts with the Python version packaged 
    with Blender."""
    # Upgrade pip
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

    # Load list of requirements from `requirements.txt`
    with open('requirements_blenderpy.txt') as f:
        required = f.read().splitlines()

    # Install required packages
    for r in required:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', r])
    
    subprocess.check_call([sys.executable, 'setup.py', 'develop']) 

if INSTALL:
    install_requirements()
    # Add parent directory to path
    # print(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # # root_dir = os.path.dirname(parent_dir)
    # print(parent_dir)
    # sys.path.append(parent_dir)
# END HACK


import bpy
from gen_env.auto_reload import DrawingClass
from gen_env.envs.blender.utils import delete_scene_objects
from gen_env.envs.blender_env import BlenderRender
from gen_env.envs.play_env import PlayEnv
from gen_env.games import GAMES


# FIXME: Hydra seems to clash with blender (segfault). Load the config manually?
# def main(cfg):
#     game, height, width = cfg.game, cfg.height, cfg.width
#     cfg.game = cfg.game
def main():
    game = 'maze'
    height, width = 10, 10

    game = GAMES[game]
    env: PlayEnv = game.make_env(height, width)
    env = BlenderRender(env)
    env.reset()
    env._done = False

    def timer_callback(env, scene):
        print('tick', time.time())
        if env._done:
        # if env.is_done():
            delete_scene_objects(scene)
            env.reset()
            env._done = False
        else:
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            env._done = done
            env.render()
            # sys.exit()

        # # Deselct all objects
        # bpy.ops.object.select_all(action='DESELECT')

        return 1e-10
        # return 1

    # Register a simple timer that prints the current time
    bpy.app.timers.register(partial(timer_callback, env, bpy.context.scene))

    print('Done.')



if __name__ == '__main__':
    context = bpy.context             
    dc = DrawingClass(context, "Draw This On Screen")

    main()