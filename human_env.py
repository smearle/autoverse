import argparse
from enum import Enum
from fire import Fire

import numpy as np
import pygame

from gen_env import GenEnv, Rule, colors, TileType
from games import (hamilton, maze, maze_pcg, maze_npc, power_line, sokoban)


def main(game=maze):
    if isinstance(game, str):
        game = globals()[game]

    env: GenEnv = game.make_env()
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