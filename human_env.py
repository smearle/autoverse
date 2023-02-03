import argparse
from enum import Enum

from fire import Fire
import hydra
import numpy as np
import pygame

from evolve import Individual, init_base_env, load_game_to_env
from play_env import PlayEnv, Rule, TileType
from games import *


@hydra.main(config_path="configs", config_name="human")
def main(cfg):
    game, height, width = cfg.game, cfg.height, cfg.width
    cfg.game = cfg.game
    if isinstance(game, str):
        if '/' in game:
            # Then it's a filepath, so we are loading up an evolved game, saved to a yaml.
            fname = game
            env = init_base_env(cfg)
            individual = Individual.load(fname)
            load_game_to_env(env, individual)
        else:
            game = globals()[game]
            env: PlayEnv = game.make_env(height, width)
    else:
        env: PlayEnv = game.make_env(height, width)

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