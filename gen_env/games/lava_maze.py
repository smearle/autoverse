from functools import partial
from pdb import set_trace as TT

import numpy as np

from gen_env.envs.play_env import PlayEnv, GameDef
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TilePlacement, TileSet, TileType


def make_env():
    force = TileType(name='force', num=0, color='purple')
    lava = TileType('lava', prob=0.1, color='red')
    floor = TileType('floor', prob=0.9, color='grey')
    player = TileType('player', num=1, color='blue', cooccurs=[floor])
    goal = TileType('goal', num=1, color='green', cooccurs=[floor])
    tiles = TileSet([force, lava, player, goal, floor])
    search_tiles = [floor, goal, player, lava]

    player_move = Rule(
        'player_move', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [[[player, force]],  # Player next to a passable/floor tile.
            [[None, floor]], # A force is active on said passable tile.
                ]  
            ,
            # Both changes are applied to the relevant channels, given by the respective input subpatterns.
            [[[None, player]],  # Player moves to target. No change at source.
            [[None, floor]],  # Force is removed from target tile.
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
    player_fall_lava = Rule(
        'player_die',
        in_out=np.array([
            [
                [[player, force]],  # Player and goal tile overlap.
                [[None, lava]],
            ],
            [
                [[None, None]],  # Player remains.
                [[None, lava]],  # Goal is removed.
            ]
        ]),
        rotate=True,
        reward=-1,
        done=False,
    )
    rules = RuleSet([player_move, player_consume_goal, player_fall_lava])
    player_passable_tiles = [(force, TilePlacement.ADJACENT)]
    game_def = GameDef(
        tiles=tiles,
        rules=rules,
        player_placeable_tiles=player_passable_tiles,
        search_tiles=search_tiles,
    )
    return game_def