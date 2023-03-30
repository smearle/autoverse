from functools import partial
from pdb import set_trace as TT

import numpy as np

from envs.play_env import PlayEnv
from rules import Rule, RuleSet
from tiles import TilePlacement, TileSet, TileType


def make_env(height, width):
    force = TileType(name='force', prob=0, color='purple')
    wall = TileType('wall', prob=0.1, color='black')
    floor = TileType('floor', prob=0.9, color='white')
    player = TileType('player', prob=0, color='blue', num=1, cooccurs=[floor])
    wire = TileType('wire', color='grey', cooccurs=[wall], inhibits=[floor])
    powered_wire = TileType('powered_wire', color='yellow', cooccurs=[floor])
    source = TileType('source', num=1, color='green', cooccurs=[powered_wire])
    target = TileType('target', num=1, color='red', cooccurs=[wire])
    tiles = TileSet([force, floor, wall, wire, powered_wire, source, target, player])  # Overlapping tiles are rendered in this order.

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
        rotate=True,
    )
    wire_conduct_power = Rule(
        'wire_conduct_power',
        in_out=np.array([
            [
                [[powered_wire, wire]],
            ],
            [
                [[powered_wire, powered_wire]],
            ]
        ]),
        rotate=True,
    )
    reward_powered_target = Rule(
        'reward_powered_target',
        in_out=np.array([
            [
                [[target]],
                [[powered_wire]],
            ],
            [
                [[target]],
                [[powered_wire]],
            ]
        ]),
        rotate=False,
        reward=1,
    )
    rules = RuleSet([player_move, wire_conduct_power, reward_powered_target])
    env = PlayEnv(height, width, tiles=tiles, rules=rules, 
        player_placeable_tiles=[(force, TilePlacement.ADJACENT), (wire, TilePlacement.CURRENT)])
    # env = partial(GenEnv, h=10, w=10, tiles=tiles, rules=rules, player_placeable_tiles=[force])

    # env.queue_games([np.array([
    # ])])
    return env