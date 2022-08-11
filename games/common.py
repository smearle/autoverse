from enum import Enum

import numpy as np

from rules import Rule
from tiles import TileType


class GamesEnum(Enum):
    MAZE = 1
    HAMILTON = 2
    SOKOBAN = 3

force = TileType(name='force', prob=0, color='purple')
# passable = TileType(name='passable', prob=0, color=None)
# floor = TileType('floor', prob=0.8, color='white', parents=[passable])
wall = TileType('wall', prob=0.2, color='black')
floor = TileType('floor', prob=0.8, color='white')
player = TileType('player', prob=0, color='blue', num=1, cooccurs=[floor])


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
    rotate=True,)
