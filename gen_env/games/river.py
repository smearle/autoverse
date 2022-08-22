from functools import partial
from math import inf
from pdb import set_trace as TT
from turtle import back

import numpy as np
from events import Event, activate_rules, on_start

from gen_env.envs.gen_env import GenEnv
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TileNot, TilePlacement, TileSet, TileType
from gen_env.variables import Variable


def make_env(height, width):
    b = TileType(name='b', color='black', prob=1.0)
    g = TileType(name='a', color='grey', num=1)
    r = TileType(name='r', color='red', num=1)
    grass = TileType(name='grass', color='green')
    forest = TileType(name='forest', color='dark_green')
    water = TileType(name='water', color='blue')

    # Maybe we want separate tilesets for maze generation and gameplay?
    tiles = TileSet([b, g, r, grass, forest, water])

    grow_g = Rule(
        'grow_g',
        in_out=np.array([
            [
                [[g, b]],
            ],
            [
                [[g, g]],
            ],
        ]),
        rotate=True,
        random=True,
    )
    grow_r = Rule(
        'grow_r',
        in_out=np.array([
            [
                [[r, b]],
            ],
            [
                [[r, r]],
            ],
        ]),
        rotate=True,
        random=True,
    )
    place_river = Rule(
        'place_river',
        in_out=np.array([
            [
                [[r, b]],
            ],
            [
                [[water, water]],
            ],
        ]),
        rotate=True,
        random=True,
    )
    remove_g = Rule(
        'remove_g',
        in_out=np.array([
            [
                [[g]],
            ],
            [
                [[b]],
            ],
        ]),
    )
    remove_r = Rule(
        'remove_r',
        in_out=np.array([
            [
                [[r]],
            ],
            [
                [[b]],
            ],
        ]),
    )


    player_move = Rule(
        'player_move', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [[[player, floor]],  # Player next to a passable/floor tile.
            [[None, force]], # A force is active on said passable tile.
                ]  
            ,
            # Both changes are applied to the relevant channels, given by the respective input subpatterns.
            [[[None, player]],  # Player moves to target. No change at source.
            [[None, None]],  # Force is removed from target tile.
            ],
        ]),
        rotate=True,)

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
    rule_sets = [
        
    ]

    env = GenEnv(height, width, tiles=tiles, rules=gamepley_rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)],
                 events=[grow_maze], variables=[maze_is_generated, placed_floors])
    return env