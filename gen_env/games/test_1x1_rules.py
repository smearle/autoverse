
from math import inf
import numpy as np

from gen_env.envs.play_env import GameDef, PlayEnv
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TileNot, TilePlacement, TileSet, TileType

def make_env():
    n_crates = 3
    force = TileType(name='force', prob=0, color='purple')
    # passable = TileType(name='passable', prob=0, color=None)
    # floor = TileType('floor', prob=0.8, color='white', parents=[passable])
    wall = TileType('wall', prob=1.0, color='black')
    floor = TileType('floor', prob=0, color='grey')
    player = TileType('player', prob=0, color='blue', num=1, cooccurs=[floor])
    crate = TileType('crate', num=n_crates, color='brown')  # Not passable.
    target = TileType('target', num=n_crates, color='green', cooccurs=[floor])

    tiles = TileSet([player, force, crate, target, wall, floor])
    search_tiles = [floor, wall, target, crate, player]

    done_at_reward = n_crates
    # player_move = Rule(
    #     'player_move', 
    #     in_out=np.array(  [# Both input patterns must be present to activate the rule.
    #         [[[None, player, floor]],  # Player next to a passable/floor tile.
    #         [[None, None, force]], # A force is active on said passable tile.
    #         ],
    #         # Both changes are applied to the relevant channels, given by the respective input subpatterns.
    #         [[[None, None, player]],  # Player moves to target. No change at source.
    #         [[None, None, floor]],  # Force is removed from target tile.
    #         ],
    #     ]),
    #     rotate=True,
    #     max_applications=inf,
    # )
    crate_on_target = Rule(
        'none_wall',
        in_out=np.array([
            [
                [[wall, None, None]],
                # [[None, None, None]],
                # [[wall]],
            ],
            [  # Kill target.
                [[wall, None, None]],
                # [[None, None, None]],
                # [[wall]],
            ]
        ]),
        # max_applications=inf,
        # FIXME: SHould be able to make this false
        rotate=True,
        reward=1,
        max_applications=inf,
    )
    # Order is important for movement/physics.
    rules = RuleSet([crate_on_target])

    # return PlayEnv(10, 10, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)], 
    #     done_at_reward=n_crates, search_tiles=search_tiles)

    game_def = GameDef(
        tiles=tiles,
        rules=rules,
        player_placeable_tiles=[(force, TilePlacement.ADJACENT)],
        done_at_reward=n_crates,
        search_tiles=search_tiles,
    )
    return game_def
