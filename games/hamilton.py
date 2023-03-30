import numpy as np

from envs.play_env import PlayEnv
from rules import Rule, RuleSet
from tiles import TilePlacement, TileSet, TileType


def make_env(height, width):
    force = TileType(name='force', prob=0, color=None)
    floor = TileType('floor', prob=0.8, color='white')
    player = TileType('player', prob=0, color='blue', num=1, cooccurs=[floor])
    lava = TileType('lava', prob=0.2, color='black')
    slime = TileType('slime', color='purple', cooccurs=[lava])
    tiles = TileSet([force, floor, lava, slime, player])

    player_move = Rule(
        'player_move', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [
                [[player, floor]],  # Player next to a passable/floor tile.
                [[floor, force]], # A force is active on said passable tile.
            ]  
            ,
        # Both changes are applied to the relevant channels, given by the respective input subpatterns.
            [
                [[slime, player]],  # Player moves to target. No change at source.
                [[None, None]],  # Floor removed from source. Force is removed from target tile.
            ],
        ]),
        rotate=True,
    )

    lava_kill_player = Rule(
        'lava_kill_player', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [
                [[force]],  # Force is active on lava tile.
                [[lava]],
            ],
            [
                [[None]],  
                [[lava]],  
            ],
        ]),
        rotate=False,
        reward=-1,
        done=True
    )
    reward_slime = Rule(
        'reward_slime',
        in_out=np.array([
            [
                [[slime]],
            ],
            [
                [[slime]],
            ],
        ]),
        rotate=False,
        reward=1,
    )
    rules = RuleSet([player_move, lava_kill_player, reward_slime])

    return PlayEnv(height, width, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)])