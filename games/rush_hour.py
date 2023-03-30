from functools import partial
from pdb import set_trace as TT

import numpy as np
from events import Event, activate_rules

from envs.play_env import PlayEnv, apply_rules
from objects import ObjectType
from rules import ObjectRule, Rule, RuleSet
from tiles import TilePlacement, TileSet, TileType, tiles_to_multihot
from variables import Variable


def make_env(height, width):
    anchor = TileType(name='anchor', prob=0, color='pink')
    force = TileType(name='force', prob=0, color='purple')
    wall = TileType('wall', prob=0.1, color='black')
    floor = TileType('floor', prob=0.9, color='grey')
    player = TileType('player', prob=0, color='blue', num=1, cooccurs=[floor])
    car = TileType('car', prob=0, color='red', num=1, cooccurs=[floor])
    goal = TileType('goal', num=1, color='green', cooccurs=[floor])
    tiles = TileSet([force, anchor, floor, goal, car, player, wall])

    car_obj = ObjectType(name='car', 
        patterns=[
            [[car, car]], 
            [[car, car, car]], 
        ],
        rules=[
            ObjectRule(
                name='move_car',
                in_out=[
                    [
                        [[car]],
                    ],
                    []
                ]
            )
        ],
    )

    car_0 = partial(car_obj.GameObject.__init__, pattern_idx=0, rot=0)

    map_0 = np.array([
        [wall, wall, wall, wall, wall, wall, wall],
        [wall, floor, floor, floor, floor, floor, wall],
        [wall, floor, floor, floor, car_0, floor, wall],
        [wall, floor, floor, floor, floor, floor, wall],
        [wall, floor, floor, player, floor, floor, wall],
        [wall, floor, floor, floor, floor, floor, wall],
        [wall, wall, wall, wall, goal, wall, wall],
    ])
    maps = [map_0]
    maps_objsets = [tiles_to_multihot(tiles, map_i) for map_i in maps]

    cars_placed = Variable(
        name='cars_placed',
    )

    grow_cars = Rule(
        name='grow_cars',
        in_out=np.array([
            [
                [[car, floor]],
            ],
            [
                [[car, car]],
            ]
        ]),
        rotate=True,
        max_applications=1,
        application_funcs=[cars_placed.increment],
    )
    player_move = Rule(
        'player_move', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [
                [[player, floor]],  # Player next to a passable/floor tile.
                [[None, force]], # A force is active on said passable tile.
            ],
            # Both changes are applied to the relevant channels, given by the respective input subpatterns.
            [
                [[None, player]],  # Player moves to target. No change at source.
                [[None, None]],  # Force is removed from target tile.
            ],
        ]),
        rotate=True,
    )
    move_car = Rule(
        'move_car', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [
                [[None, player, floor]],  
                [[None, anchor, force]], 
                [[car, car, None]], 
            ],
            # Both changes are applied to the relevant channels, given by the respective input subpatterns.
            [
                [[None, None, player]],  
                [[None, None, None]],  
                [[None, car, car]]
            ],
        ]),
        rotate=True,
    )
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
    kill_anchor = Rule(
        'kill_anchor',
        in_out=np.array([
            [
                [[anchor]],
            ],
            [
                [[None]],
            ]
        ]),
        rotate=False,
        reward=0,
        done=False,
    )
    gameplay_rules = RuleSet([kill_force, move_car, kill_anchor, player_move])
    gameplay = Event(
        name='gameplay',
        tick_func=partial(activate_rules, rules=gameplay_rules),
    )
    generate_board = Event(
        name='generate_board',
        tick_func=partial(activate_rules, rules=[grow_cars]),
        done_cond=lambda: cars_placed.value > 0,
        children=[gameplay]
    )

    env = PlayEnv(height, width, tiles=tiles, rules=gameplay_rules, 
        player_placeable_tiles=[(force, TilePlacement.ADJACENT), (anchor, TilePlacement.CURRENT)],
        # events=[generate_board]
       )
    env.queue_games(maps_objsets)
    return env