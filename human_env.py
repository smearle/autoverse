import argparse
from enum import Enum
from fire import Fire

import numpy as np
import pygame
from evolve import Individual, init_base_env, load_game_to_env

from gen_env import GenEnv, Rule, TileType
from games import *


def main(game=maze, height: int = 10, width: int = 10):
    if isinstance(game, str):
        if '/' in game:
            # Then it's a filepath, so we are loading up an evolved game, saved to a yaml.
            fname = game
            env = init_base_env()
            individual = Individual.load(fname)
            load_game_to_env(env, individual)
        else:
            game = globals()[game]
            env: GenEnv = game.make_env(height, width)
    else:
        env: GenEnv = game.make_env(height, width)

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
    Fire(main)