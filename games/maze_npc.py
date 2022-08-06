from functools import partial
from pdb import set_trace as TT

import numpy as np

from events import Event, EventGraph, on_start
from games.common import colors
from gen_env import GenEnv
from pathfinding import draw_shortest_path
from rules import Rule, RuleSet
from tiles import TilePlacement, TileSet, TileType


def make_env(env_cfg={}):
    npc_path = TileType('npc_path', prob=0, color=colors['grey'])
    force = TileType(name='force', prob=0, color=None)
    wall = TileType('wall', prob=0.1, color=colors['black'])
    floor = TileType('floor', prob=0.9, color=colors['white'])
    npc = TileType('npc', prob=0, color=colors['red'], num=1, cooccurs=[floor])
    player = TileType('player', prob=0, color=colors['blue'], num=1, cooccurs=[floor])
    goal = TileType('goal', num=1, color=colors['green'], cooccurs=[floor])
    tiles = TileSet([force, floor, npc_path, goal, npc, player, wall])

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

    npc_move = Rule(
        'npc_move',
        in_out=np.array([
            [
                [[npc, npc_path]],
            ],
            [
                [[None, npc]],
            ]
        ]),
        rotate=True,
    )

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
    npc_consume_goal = Rule(
        'npc_consume_goal',
        in_out=np.array([
            [
                [[npc]],  # Player and goal tile overlap.
                [[goal]],
            ],
            [
                [[npc]],  # Player remains.
                [[None]],  # Goal is removed.
            ]
        ]),
        rotate=False,
        reward=-1,
        done=True,
    )
    npc_search_goal = Event(
        'npc_search_goal',
        tick_func=partial(draw_shortest_path, traversable_tiles=[floor], src_tile=npc, trg_tile=goal, out_tile=npc_path),
    )
    events = [npc_search_goal]
    rules = RuleSet([player_move, npc_move, player_consume_goal, npc_consume_goal])
    env = GenEnv(10, 10, tiles=tiles, rules=rules, events=events, player_placeable_tiles=[(force, TilePlacement.ADJACENT)])
    return env