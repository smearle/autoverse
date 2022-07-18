import numpy as np

from games.common import colors, force, floor, player, player_move, wall
from gen_env import GenEnv
from rules import Rule, RuleSet
from tiles import TileSet, TileType

def make_env():
    n_crates = 3
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

    return GenEnv(10, 10, tiles=tiles, rules=rules, player_placeable_tiles=[force], done_at_reward=n_crates)