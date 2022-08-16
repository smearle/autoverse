from typing import Iterable, List, Tuple

import numpy as np

from tiles import TileType


class ObjectType():
    def __init__(self, name: str, patterns: List[List[TileType]]):
        self.name = name
        self.patterns = patterns

        class GameObject():
            def __init__(self, pattern_idx: int, rot: int, pos: Tuple[int, int], map_arr: np.ndarray):
                """
                Args:
                    pattern_idx (int): Index of the pattern to use.
                    pos (Tuple[int]): Position of the object on the map.
                    map_arr (np.ndarray): The multihot (c, h, w) map array.
                """
                self.pattern = patterns[pattern_idx]
                # Always top-left of the pattern(?)
                self.pos = pos

            def move_to(self, pos: Tuple[int, int], map_arr: np.ndarray):
                self.remove_from(self.pos, map_arr)
                self.add_to(pos, map_arr)

            def remove_from(pos, map_arr):
                x, y = pos
                for i in self.pattern.shape[0]:
                    for j in self.pattern.shape[1]:
                        tile = self.pattern[i, j]
                        map_arr[tile.get_idx(), x + i, y + j] = 0

            def add_to(pos, map_arr):
                x, y = pos
                for i in self.pattern.shape[0]:
                    for j in self.pattern.shape[1]:
                        map_arr[tile.get_idx(), x + i, y + j] = 1

            def __str__(self):
                return f"{name}: {self.pattern}"

        self.GameObject = GameObject


def tiles_to_multihot(tiles: Iterable[TileType], map_arr: np.ndarray) -> np.ndarray:
    """
    Convert a 2D array of tiles to a 3D multi-hot encoding of the map.

    Args:
        tiles (Iterable[TileType]): The tiles to encode.
        map_arr (np.ndarray): The map to encode.

    Returns:
        np.ndarray: The multi-hot encoding of the map.
    """
    multihot = np.zeros((len(tiles), map_arr.shape[0], map_arr.shape[1]))
    objects = []
    for x in range(map_arr.shape[0]):
        for y in range(map_arr.shape[1]):
            tile = map_arr[x, y]
            multihot[tile.get_idx(), x, y] = 1
            # el = map_arr[x, y]
            if isinstance(el, TileType):
                el = [el]
            elif isinstance(el, dict):
                obj_type = el.pop('type')
                obj = obj_type(x=x, y=y, **el)
                objects.append(obj)
            # for tile in el:
                # multihot[tile.get_idx(), x, y] = 1
    for obj in objects:
        obj.add_to(obj.pos, multihot)

    def tick(self, env):
        pass

    return multihot, objects
