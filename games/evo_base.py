from functools import partial
from pdb import set_trace as TT

import numpy as np

from play_env import PlayEnv
from rules import Rule, RuleSet
from tiles import TilePlacement, TileSet, TileType


def make_env(height, width):
    force = TileType(name='force', prob=0, color='purple')
    wall = TileType('wall', prob=1/4, color='black')
    floor = TileType('floor', prob=1/4, color='grey')
    player = TileType('player', prob=0, num=1, color='blue')
    goal = TileType('goal', prob=1/4, color='green')
    spike = TileType('spike', prob=1/4, color='red')
    tiles = TileSet([floor, goal, wall, force, spike, player])

    player_move = Rule(
        'A', 
        in_out=np.array(  [
            [[[None]],  
            ]  
            ,
            [[[None]], 
            ],
        ]),
        )

    player_consume_goal = Rule(
        'B',
        in_out=np.array([
            [
                [[None]],
            ],
            [
                [[None]],
            ]
        ]),
    )
    wall_kill_force = Rule(
        'C',
        in_out=np.array([
            [
                [[None]],
            ],
            [
                [[None]],
            ]
        ]),
    )
    spike_kill_player = Rule(
        'D',
        in_out=np.array([
            [
                [[None]],
            ],
            [
                [[None]], 
            ]
        ]),
    )
    rules = RuleSet([wall_kill_force, player_consume_goal, spike_kill_player, player_move])
    env = PlayEnv(height, width, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)])
    return env