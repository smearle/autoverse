from functools import partial
from pdb import set_trace as TT

import numpy as np

from games.common import colors, force, floor, player, player_move, wall
from gen_env import GenEnv
from rules import Rule, RuleSet
from tiles import TileSet, TileType


def make_env(env_cfg={}):
    goal = TileType('goal', num=1, color=colors['green'], cooccurs=[floor])
    tiles = TileSet([force, floor, goal, player, wall])

    player_consume_goal = Rule(
        'player_consume_goal',
        in_out=np.array([
            [
                [[player]],  # Player and goal tile overlap.
                [[goal]],
            ],
            [
                [[player]],  # Player remains.
                [[None]],  # Goal is removed.
            ]
        ]),
        rotate=False,
        reward=1,
        done=True,
    )
    rules = RuleSet([player_move, player_consume_goal])
    env = GenEnv(10, 10, tiles=tiles, rules=rules, player_placeable_tiles=[force])
    # env = partial(GenEnv, h=10, w=10, tiles=tiles, rules=rules, player_placeable_tiles=[force])

    # env.queue_maps([np.array([
    # ])])
    return env