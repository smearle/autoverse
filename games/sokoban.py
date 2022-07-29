import numpy as np

from games.common import colors, player_move
from gen_env import GenEnv
from rules import Rule, RuleSet
from tiles import TilePlacement, TileSet, TileType

def make_env():
    n_crates = 3
    force = TileType(name='force', prob=0, color=None)
    # passable = TileType(name='passable', prob=0, color=None)
    # floor = TileType('floor', prob=0.8, color=colors['white'], parents=[passable])
    wall = TileType('wall', prob=0.2, color=colors['black'])
    floor = TileType('floor', prob=0.8, color=colors['white'])
    player = TileType('player', prob=0, color=colors['blue'], num=1, cooccurs=[floor])
    crate = TileType('crate', num=n_crates, color=colors['brown'])  # Not passable.
    target = TileType('target', num=n_crates, color=colors['green'], cooccurs=[floor])

    tiles = TileSet([force, floor, wall, target, crate, player])

    done_at_reward = n_crates
    player_push_crate = Rule(
        'player_push_crate',
        in_out=np.array([
            [
                [[player, force, floor]],  # Player next to force, then floor
                [[None, crate, None]],  # Crate on same tile as force.
            ],
            [
                [[None, player, crate]],  # Player moves, force removed, crate overwrites floor.
                [[None, None, None]],  # No effect.
            ]
        ]),
        rotate=True,
        reward=-0
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
    )
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
    )
    crate_on_target = Rule(
        'crate_on_target',
        in_out=np.array([
            [
                [[crate]],
                [[target]],
            ],
            [  # No change.
                [[crate]],
                [[target]],
            ]
        ]),
        rotate=False,
        reward=1
    )
    # Order is important for movement/physics.
    rules = RuleSet([player_push_crate, crate_kill_force, player_move, crate_on_target])

    return GenEnv(10, 10, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)], 
        done_at_reward=n_crates)