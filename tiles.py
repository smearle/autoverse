from enum import Enum
from typing import Iterable, List, Tuple


class TilePlacement(Enum):
    """Where can the player place tiles?"""
    # Tiles can be placed at the tile currently occupied by the player.
    CURRENT = 0
    # Tiles can be placed at any of the 4 tiles adjacent to the player.
    ADJACENT = 1


class TileType():
    def __init__(self, name: str, color: Tuple[int], prob: float = 0,  passable: bool = False, num: int = None,
            parents: List = [], cooccurs: List = [], inhibits: List = [],):
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
        self.color = color
        self.passable = passable
        self.num = num
        self.idx = None  # This needs to be set externally, given some consistent ordering of the tiles.
        self.parents = parents
        self.cooccurs = cooccurs
        self.inhibits = inhibits

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

