from functools import partial
from pdb import set_trace as TT

import numpy as np
from events import Event, activate_rules

from play_env import PlayEnv
from rules import Rule, RuleSet
from tiles import TilePlacement, TileSet, TileType
from variables import Variable


def make_env(height, width):

    force = TileType(name='force', prob=0, color='purple')
    wall = TileType('wall', prob=0.3, color='black')
    floor = TileType('floor', prob=0.7, color='grey')
    player = TileType('player', num=1, color='blue', cooccurs=[floor])
    key = TileType('key', num=1, color='gold', cooccurs=[floor])
    door = TileType('door', num=1, color='brown')
    enemy = TileType('enemy', num=1, color='red', cooccurs=[floor])

    tiles = TileSet([floor, key, door, enemy, player, wall, force])

    player_keys = Variable(
        'player_keys',
        initial_value=0,
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
        rotate=True,
    )
    enemy_move = Rule(
        'enemy_move', 
        in_out=np.array(  [
            [
                [[enemy, floor]],
            ],
            [
                [[None, enemy]],
            ],
        ]),
        rotate=True,
        random=True,
        max_applications=1,
    )
    enemy_hunt_player = Rule(
        'enemy_hunt_player', 
        in_out=np.array(  [
            [
                [[enemy, player]],
            ],
            [
                [[None, enemy]],
            ],
        ]),
        rotate=True,
        random=True,
        max_applications=1,
        reward=-1,
        done=True,
        inhibits=[enemy_move, player_move]
    )
    player_get_key = Rule(
        'player_get_key',
        in_out=np.array([
            [
                [[player, force]],  # Player moving toward key.
                [[None, key]],
            ],
            [
                [[None, player]],  # Player moves.
                [[None, None]],  # Key is removed.
            ]
        ]),
        application_funcs=[player_keys.increment],
        rotate=True,
        reward=1,
        inhibits=[player_move],
    )
    player_unlock_door = Rule(
        'player_unlock_door',
        in_out=np.array([
            [
                [[player, force]],  # Player moving toward door.
                [[None, door]],
            ],
            [
                [[None, player]],  # Player moves.
                [[None, None]],  # Door is removed.
            ]
        ]),
        inhibits=[player_move],
        rotate=True,
        reward=1,
        done=True,
    )
    player_unlock_door.compile()
    kill_force = Rule(
        'kill_force',
        in_out=np.array([
            [
                [[force]],
            ],
            [
                [[None]],
            ]
        ]),
        rotate=False,
        reward=0,
        done=False,
    )
    enemy_kill_player = Rule(
        'enemy_kill_player',
        in_out=np.array([
            [
                [[player]],
                [[enemy]],
            ],
            [
                [[None]],  
                [[enemy]], 
            ]
        ]),
        inhibits=[player_move],
        rotate=True,
        reward=-1,
        done=True,
    )

    rules = RuleSet([enemy_hunt_player, enemy_kill_player, 
        enemy_move, player_get_key, player_move, kill_force])

    player_has_key = Event(
        name='player_has_key',
        tick_func=partial(activate_rules, rules=rules + [player_unlock_door]),
        init_cond=lambda: player_keys.value >= 1,
    )

    env = PlayEnv(height, width, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)], 
        events=[player_has_key], variables=[player_keys])
    return env