from functools import partial
from pdb import set_trace as TT

import numpy as np

from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TilePlacement, TileSet, TileType


def make_env():
    floor = TileType('floor', prob=0.9, color='grey')
    player = TileType('player', num=1, color='blue', cooccurs=[])
    tile_a = TileType('tile_a', prob=0, color='red')
    tile_b = TileType('tile_b', prob=0, color='yellow')
    tile_c = TileType('tile_c', prob=0, color='orange')
    # tiles = TileSet([floor, goal, player, wall, force, tile_a])
    tiles = TileSet([player, tile_a, tile_b, tile_c])
    # search_tiles = [floor, goal, player, wall, tile_a]
    search_tiles = [floor, player, tile_a, tile_b, tile_c]

    rule_a = Rule(
        'A',
        in_out=np.array([
            [
                # [[None, player, force]],
                # [[None, None, floor]],
                [[None, None, None]],
                [[None, None, None]],
            ],
            [
                # [[None, None, player]],
                # [[None, None, floor]],
                [[None, None, None]],
                [[None, None, None]],
            ]
        ]),
        rotate=True,
        reward=0,
        done=False,
        # max_applications=1,
    )
    rule_b = Rule(
        'B',
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
        # max_applications=1,
    )
    rule_c = Rule(
        'C',
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
        # max_applications=1,
    )
    rule_d = Rule(
        'D',
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
        # max_applications=1,
    )
    rule_e = Rule(
        'E',
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
        # max_applications=1,
    )
    rules = RuleSet([rule_a, rule_b, rule_c, rule_d, rule_e])
    # rules = RuleSet([rule_a])
    # rules = RuleSet([player_move, player_consume_goal, rule_a, rule_b])
    # env = PlayEnv(height, width, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)],
        # search_tiles=search_tiles, cfg=cfg)
    game_def = dict(
        tiles=tiles,
        rules=rules,
        player_placeable_tiles=[(tile_a, TilePlacement.ADJACENT)],
    )
    return game_def