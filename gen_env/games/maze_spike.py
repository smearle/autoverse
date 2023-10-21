from functools import partial
from pdb import set_trace as TT

import numpy as np

from gen_env.envs.play_env import PlayEnv
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TilePlacement, TileSet, TileType


def make_env():
    force = TileType(name='force', prob=0, color='purple')
    wall = TileType('wall', prob=0.1, color='black')
    floor = TileType('floor', prob=0.9, color='grey')
    player = TileType('player', prob=0, color='blue', num=1, cooccurs=[floor])
    goal = TileType('goal', num=1, color='green', cooccurs=[floor])
    spike = TileType('spike', num=1, color='red', cooccurs=[floor])
    tiles = TileSet([floor, goal, player, wall, force, spike])

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
                [[player, force]],  # Player moving toward goal.
                [[None, goal]],
            ],
            [
                [[None, player]],  # Player moves.
                [[None, None]],  # Goal is removed.
            ]
        ]),
        inhibits=[player_move],
        rotate=True,
        reward=1,
        done=True,
    )
    wall_kill_force = Rule(
        'wall_kill_force',
        in_out=np.array([
            [
                [[force]],
                [[wall]],
            ],
            [
                [[None]],
                [[wall]],
            ]
        ]),
        rotate=False,
        reward=0,
        done=False,
    )
    spike_kill_player = Rule(
        'spike_kill_player',
        in_out=np.array([
            [
                [[player, force]],
                [[None, spike]],
            ],
            [
                [[None, None]],  
                [[None, spike]], 
            ]
        ]),
        inhibits=[player_move],
        rotate=True,
        reward=-1,
        done=True,
    )
    rules = RuleSet([wall_kill_force, player_consume_goal, spike_kill_player, player_move])
    game_def = dict(
        tiles=tiles,
        rules=rules,
        player_placeable_tiles=[(force, TilePlacement.ADJACENT)],
    )
    return game_def