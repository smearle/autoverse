from enum import Enum
from pdb import set_trace as TT
import random
from typing import Iterable, List, Tuple

import numpy as np


class TilePlacement(Enum):
    """Where can the player place tiles?"""
    # Tiles can be placed at the tile currently occupied by the player.
    CURRENT = 0
    # Tiles can be placed at any of the 4 tiles adjacent to the player.
    ADJACENT = 1

colors = {
    'aqua': (0, 255, 255),
    'black': (0, 0, 0),
    'blue': (0, 0, 255),
    'brown': (165, 42, 42),
    'cyan': (0, 255, 255),
    'dark_red': (128, 0, 0),
    'error': (255, 192, 203),  # pink
    'gold': (255, 215, 0),
    'green': (0, 255, 0),
    'grey': (128, 128, 128),
    'light_grey': (211, 211, 211),
    'magenta': (255, 0, 255),
    'orange': (255, 165, 0),
    'pink': (255, 192, 203),
    'purple': (128, 0, 128),
    'red': (255, 0, 0),
    'white': (255, 255, 255),
    'yellow': (255, 255, 0),
    'yellow_green': (154, 205, 50),
}
colors = {k: np.array(v) for k, v in colors.items()}

class TileType():
    def __init__(self, name: str, color: str, prob: float = 0,  passable: bool = False, num: int = None,
            parents: List = [], cooccurs: List = [], inhibits: List = []):
        """Initialize a new tile type.

        Args:
            name (str): Name of the tile
            prob (float): Probability of the tile spawning during random map generation, if applicable. If 0, then `num`
                must be specified to spawn an exact number of this tile.
            color (Tuple[int]): The color of the tile during rendering.
            passable (bool, optional): Whether this tile is passable to player. Defaults to False.
            num (int, optional): Exactly how many instances of this tile should spawn during random map generation. 
                Defaults to None.
            parents (List[TileType]): A list of tile types from which this tile can be considered a sub-type. Useful 
                for rules that consider parent types. (Deprecated)
            cooccurs (List[TileType]): A list of tile types that co-occur with this tile. At each tick, these tiles will
                spawn/overlap with any instances of this tile.
            inhibits (List[TileType]): A list of tile types that inhibit this tile. At each tick, these tiles will be
                removed wherever this tile is present.
        """
        self.name = name
        self.prob = prob
        self.color_name = color
        if color is None:
            color = 'purple'
        self.color = colors[color]
        self.passable = passable
        self.num = num
        self.idx = None  # This needs to be set externally, given some consistent ordering of the tiles.
        self.parents = parents
        self.cooccurs = cooccurs
        self.inhibits = inhibits
        # When this tile is ``active'' in the multihot encoding. Overwritten by TileNot wrapper for pattern matching.
        self.trg_val = 1  

    def mutate(self, other_tiles):
        x = random.random()
        n_mut_types = 6
        if x < 1 / n_mut_types:
            self.prob = max(0, self.prob - 0.1)
            self.num = None
        elif x < 2 / n_mut_types:
            self.prob = min(1, self.prob + 0.1)
            self.num = None
        elif x < 3 / n_mut_types:
            self.prob = random.random()
        elif x < 4 / n_mut_types:
            self.num = random.randint(0, 10) if self.num is None else None
            self.prob = 0
        elif x < 5 / n_mut_types:
            self.inhibits = random.sample(other_tiles, random.randint(0, len(other_tiles)))
        else:
            self.cooccurs = random.sample(other_tiles, random.randint(0, len(other_tiles)))

    def to_dict(self):
        return {
            self.name: {
                'color': self.color_name,
                'prob': self.prob,
                'num': self.num,
                'cooccurs': [t.name for t in self.cooccurs],
                'inhibits': [t.name for t in self.inhibits],
            }
        }

    def from_dict(d):
        return TileType(**d)

    def get_idx(self):
        if self is None:
            return -1
        return self.idx

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"TileType: {self.name}"


class TileSet(list):
    def __init__(self, tiles: Iterable[TileType]):
        [setattr(tile, "idx", i) for i, tile in enumerate(tiles)]
        super().__init__(tiles)


class TileNot():
    def __init__(self, tile: TileType):
        self.tile = tile
        self.get_idx = tile.get_idx
        self.trg_val = 0

    def __str__(self):
        return f"TileNot {self.tile}"
