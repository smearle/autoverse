from enum import Enum

import numpy as np
import pygame

from gen_env import GenEnv, Rule, colors, TileType
from games import hamilton, maze, sokoban
from games.common import GamesEnum


# GAME = TestGame.MAZE
# GAME = TestGame.SOKOBAN
GAME = GamesEnum.HAMILTON

if __name__ == "__main__":
    if GAME == GamesEnum.MAZE:
        env = maze.make_env()
    elif GAME == GamesEnum.SOKOBAN:
        env = sokoban.make_env()
    elif GAME == GamesEnum.HAMILTON:
        env = hamilton.make_env()
    env.reset()
    env.render(mode='pygame')
    done = False

    running = True

    while running:
        env.tick_human()

    # Done! Time to quit.
    pygame.quit()