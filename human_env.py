import argparse
from enum import Enum

import hydra
import numpy as np
import pygame

from gen_env.configs.config import Config
from gen_env.evo.individual import Individual
from gen_env.utils import init_base_env, load_game_to_env
from gen_env.envs.play_env import PlayEnv, Rule, TileType
from gen_env.games import *


@hydra.main(config_path="gen_env/configs", config_name="human")
def main(cfg: Config):
    game, map_shape = cfg.game, cfg.map_shape
    cfg.game = cfg.game
    if isinstance(game, str):
        if cfg.load_game is not None:
            # Then it's a filepath, so we are loading up an evolved game, saved to a yaml.
            fname = cfg.load_game
            env = init_base_env(cfg)
            individual = Individual.load(fname, cfg)
            load_game_to_env(env, individual)
        else:
            game_file = globals()[game]
            game_def = game_file.make_env()
            env = PlayEnv(
                cfg=cfg, height=map_shape[0], width=map_shape[1],
                **game_def
            )
            # env: PlayEnv = game.make_env(height, width)
    else:
        env: PlayEnv = game.make_env()

    # Set numpy seed
    np.random.seed(0)
    env.reset()
    env.render(mode='pygame')
    done = False

    running = True

    while running:
        # env.step(env.action_space.sample())
        # env.render()
        env.tick_human()

    pygame.quit()

if __name__ == "__main__":
    main()