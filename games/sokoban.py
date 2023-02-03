from math import inf
import numpy as np

from play_env import PlayEnv
from rules import Rule, RuleSet
from tiles import TileNot, TilePlacement, TileSet, TileType

def make_env(height, width):
    n_crates = 3
    force = TileType(name='force', prob=0, color=None)
    # passable = TileType(name='passable', prob=0, color=None)
    # floor = TileType('floor', prob=0.8, color='white', parents=[passable])
    wall = TileType('wall', prob=0.2, color='black')
    floor = TileType('floor', prob=0.8, color='grey')
    player = TileType('player', prob=0, color='blue', num=1, cooccurs=[floor])
    crate = TileType('crate', num=n_crates, color='brown')  # Not passable.
    target = TileType('target', num=n_crates, color='green', cooccurs=[floor])

    tiles = TileSet([force, floor, wall, target, crate, player])
    search_tiles = [floor, wall, target, crate, player]

    done_at_reward = n_crates
    player_move = Rule(
        'player_move', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [[[player, floor]],  # Player next to a passable/floor tile.
            [[None, force]], # A force is active on said passable tile.
            ],
            # Both changes are applied to the relevant channels, given by the respective input subpatterns.
            [[[None, player]],  # Player moves to target. No change at source.
            [[None, None]],  # Force is removed from target tile.
            ],
        ]),
        rotate=True,
        max_applications=inf,
    )
    crate_on_target = Rule(
        'crate_on_target',
        in_out=np.array([
            [
                [[crate]],
                [[target]],
                [[TileNot(force)]]  # Otherwise we can puch a clone-crate off a target.
            ],
            [  # Kill target.
                [[crate]],
                [[None]],
                [[None]],
            ]
        ]),
        # max_applications=inf,
        rotate=False,
        reward=1,
        max_applications=inf,
    )
    # This prevents player from passing through crates when they are on targets (and against a wall).
    crate_kill_force = Rule(
        'crate_kill_force',
        in_out=np.array([
            [
                [[crate,]],
                [[force]],
            ],
            [
                [[crate]],
                [[None]],
            ]
        ]),
        max_applications=inf,
    )
    player_push_crate = Rule(
        'player_push_crate',
        in_out=np.array([
            [
                [[player, crate, floor]],  # Player next to force, then floor
                [[None, force, None]],  # Crate on same tile as force.
            ],
            [
                [[None, player, crate]],  # Player moves, force removed, crate overwrites floor.
                [[None, None, None]],  # No effect.
            ]
        ]),
        max_applications=inf,
        # max_applications=1,
        rotate=True,
        reward=0,
        inhibits=[player_move, crate_kill_force],
    )
    # Order is important for movement/physics.
    rules = RuleSet([player_push_crate, crate_kill_force, player_move, crate_on_target])

    return PlayEnv(10, 10, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)], 
        done_at_reward=n_crates, search_tiles=search_tiles)