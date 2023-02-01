from functools import partial
from pdb import set_trace as TT

import numpy as np

from gen_env import GenEnv
from rules import Rule, RuleSet
from tiles import TilePlacement, TileSet, TileType


def make_env(height, width):
    force = TileType(name='force', num=0, color='purple')
    wall = TileType('wall', prob=0.1, color='black')
    floor = TileType('floor', prob=0.9, color='grey')
    player = TileType('player', num=1, color='blue', cooccurs=[floor])
    goal = TileType('goal', num=1, color='green', cooccurs=[floor])
    tile_a = TileType('tile_a', prob=0, color='red')
    tiles = TileSet([floor, goal, player, wall, force, tile_a])
    search_tiles = [floor, goal, player, wall, tile_a]

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
                [[player, force]],  # Player and goal tile overlap.
                [[None, goal]],
            ],
            [
                [[None, player]],  # Player remains.
                [[None, None]],  # Goal is removed.
            ]
        ]),
        rotate=True,
        reward=1,
        done=True,
    )
    player_consume_goal = Rule(
        'A',
        in_out=np.array([
            [
                [[None, None, None]],
                [[None, None, None]],
            ],
            [
                [[None, None, None]],
                [[None, None, None]],
            ]
        ]),
        rotate=True,
        reward=0,
        done=False,
    )
    rules = RuleSet([player_move, player_consume_goal])
    env = GenEnv(height, width, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)],
        search_tiles=search_tiles)
    return env