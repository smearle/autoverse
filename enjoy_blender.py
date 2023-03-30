
import hydra
import bpy
from gen_env.auto_reload import DrawingClass

from gen_env.envs.blender_env import BlenderRender
from gen_env.envs.play_env import PlayEnv
from gen_env.games import GAMES


@hydra.main(version_base="1.3", config_path="gen_env/configs", config_name="human")
def main(cfg):
    game, height, width = cfg.game, cfg.height, cfg.width
    cfg.game = cfg.game

    game = GAMES[game]
    env: PlayEnv = game.make_env(height, width)
    env = BlenderRender(env)

    while True:
        done = False
        obs = env.reset()
        env.render(mode='human')
        while not done:
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render(mode='human')

    

if __name__ == '__main__':
    context = bpy.context             
    dc = DrawingClass(context, "Draw This On Screen")

    main()