import numpy as np

from games.common import colors, force, floor, player
from gen_env import GenEnv
from rules import Rule, RuleSet
from tiles import TileSet, TileType


def make_env():
    lava = TileType('lava', prob=0.2, color=colors['black'])
    slime = TileType('slime', color=colors['purple'], cooccurs=[lava])
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
        reward=1,)

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
        done=True)
    rules = RuleSet([player_move, lava_kill_player])

    return GenEnv(10, 10, tiles=tiles, rules=rules, player_placeable_tiles=[force])