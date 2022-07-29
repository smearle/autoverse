from enum import Enum

import numpy as np

from rules import Rule
from tiles import TileType


class GamesEnum(Enum):
    MAZE = 1
    HAMILTON = 2
    SOKOBAN = 3

colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'grey': (128, 128, 128),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'brown': (165, 42, 42),
    'dark_red': (128, 0, 0),
    'error': (255, 192, 203),  # pink
}
colors = {k: np.array(v) for k, v in colors.items()}

force = TileType(name='force', prob=0, color=None)
# passable = TileType(name='passable', prob=0, color=None)
# floor = TileType('floor', prob=0.8, color=colors['white'], parents=[passable])
wall = TileType('wall', prob=0.2, color=colors['black'])
floor = TileType('floor', prob=0.8, color=colors['white'])
player = TileType('player', prob=0, color=colors['blue'], num=1, cooccurs=[floor])


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
