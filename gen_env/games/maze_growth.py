from functools import partial
from pdb import set_trace as TT
from turtle import back

import numpy as np

from gen_env.events import Event, activate_rules, on_start
from gen_env.envs.play_env import PlayEnv
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TileNot, TilePlacement, TileSet, TileType
from gen_env.variables import Variable


def make_env(height, width):
    force = TileType(name='force', prob=0, color='grey')
    wall = TileType('wall', prob=1.0, color='black')
    floor = TileType('floor', prob=0.0, color='white', num=1)
    player = TileType('player', prob=0, color='blue', num=0, cooccurs=[floor])
    goal = TileType('goal', num=0, color='green', cooccurs=[floor])

    # Tiles for maze-generation along.
    # TODO: Exclude these from the observation / onehot-encoded map after maze generation?
    corridor = TileType('corridor', prob=0, color='grey')

    # Maybe we want separate tilesets for maze generation and gameplay?
    tiles = TileSet([force, floor, goal, player, wall, corridor])

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
    gamepley_rules = RuleSet([player_move, player_consume_goal])

    maze_is_generated = Variable(
        'maze_is_generated',
        initial_value=0,
    )

    # We'll define these rules essentially in reverse order since they ``inhibit'' one another. Maybe the logic should
    # flow forward instead?
    place_goal = Rule(
        name='place_goal',
        in_out=np.array([
            [
                [[floor]],
                [[TileNot(player)]]
            ],
            [
                [[goal]],
                [[None]],
            ]
        ]),
        rotate=False,
        random=True,
        max_applications=1,
        application_funcs=[
            maze_is_generated.increment
        ],
    )
    place_goal.compile()
    place_player = Rule(
        name='builder_to_player',
        in_out=np.array([
            [
                [[floor]],
            ],
            [
                [[player]],
            ]
        ]),
        rotate=False,
        random=True,
        children=[place_goal],
    )
    place_floor = Rule(
        name='build_floor',
        in_out=np.array([
            [
                [[corridor]],
            ],
            [
                [[floor]],
            ]
        ]),
        rotate=False,
        random=True,
        inhibits=[place_player],
    )
    place_floor.compile()
    grow = Rule(
        name='grow',
        in_out=np.array([
            [
                [[floor, wall, wall]],
            ],
            [
                [[floor, corridor, floor]],
            ]
        ]),
        max_applications=1,
        rotate=True,
        random=True,
        inhibits=[place_floor, place_player],
    )
    # The order of the list containing rules should determine their order of application.
    maze_gen_rules = RuleSet([grow, place_floor, place_player])
    place_goal.compile()
    maze_gameplay = Event(
        name='maze_gameplay',
        tick_func=partial(activate_rules, rules=gamepley_rules),
    )
    generate_maze = Event(
        name='generate_maze',
        tick_func=partial(activate_rules, rules=maze_gen_rules),
        done_cond=lambda: maze_is_generated.value > 0,
        children=[maze_gameplay]
    )
    env = PlayEnv(height, width, tiles=tiles, rules=gamepley_rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)],
                 events=[generate_maze], variables=[maze_is_generated])
    return env