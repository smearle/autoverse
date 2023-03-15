"""Rules determine a game's mechanics. They map input tile patterns to output patterns, rewards, done state (... and 
changes in global variables?).

During each step of the game engine, rules are applied in order, and the application of each rule involves scanning over
the tiles in the board in search of the relevant input pattern.

TODO: 
  - Add a `random_order` boolean value to rules that will result in scanning the board in a random order. THis will 
  allow, e.g., NPCs that move randomly on the board.
  - Re-implement some of MarkovJunior's level generation algos. Each game can optionally have a separate set of 
    map-generation rules so that we get, e.g., reasonable-looking mazes.


Implementation goal / design principle: compile rules as differential convolutional neural networks.

Assumptions to start:
- each rule will be applied in sequence. Whenever a rule is applied, we expect everything else on the map
to remain fixed. 
- two applications of the same rule will not overlap.
- don't worry about differentiability *too much* yet. E.g., can use step functions for now then scale things later to make it
  smooth/differentiable (?)

- weight of 2 from channel to itself at same (center) tile
- bias of -1 everywhere
- step function activations

  (This would any instances of A by 1 to the right at each tick.)
- [A, -] -> [-, A]: weight of 1 from channel to itself at adjacent tile
                    weight of -1 from channel to itself at center tile

  (i.e. player (A) moving toward user-input force (B))
- [A, B] -> [-, A]: weight of 1 from channel to itself at center tile (by default)
                    weight of 1/2 from A center to B right, 1/2 from B right to B right; and the -1/2 from both to A left  

- [A, -] -> [B, -]: weight of 2 from A center to B center
                    nothing else since empty tile in output does not replace anything
"""
from functools import partial
import math
from pdb import set_trace as TT
import random
from typing import Dict, Iterable

import numpy as np
from objects import ObjectType

from tiles import TileType


def tile_to_str(tile: TileType) -> str:
    return tile.name if tile is not None else tile

def tile_from_str(name: str, names_to_tiles: Dict[str, TileType]) -> TileType:
    return names_to_tiles[name] if not (name is None or name == 'None') else None


class Rule():
    """Rules match input to output patterns. When the environment ticks, the input patters are identified and replaced 
    by the output patterns. Each rule may consist of multiple sub-rules, any one of which can be applied. Each rule may
    consist of multiple subpatterns, which may all match in order for the subrule to be applied.
    Input and output are 2D patches of tiles.
    """
    def __init__(self, name: str, in_out: Iterable[TileType], rotate: bool = True, reward: int = 0, done: bool = False,
            random: bool = False, max_applications: int = np.inf, inhibits: Iterable = [], children: Iterable = [],
            application_funcs: Iterable = []):
        """Process the main subrule `in_out`, potentially rotating it to produce a set of subrules. Also convert 
        subrules from containing TileType objects to their indices for fast matching during simulation.

        Args:
            name (str): The rule's name. For human-readable purposes.
            in_out (Iterable[TileType]): A sub-rule with shape (2, n_subpatterns, h, w)
            rotate (bool): Whether subpatterns .
        """
        self.name = name
        self._in_out = in_out
        self._rotate = rotate
        self.application_funcs = application_funcs
        self.children = children
        self.done = done
        self.inhibits = inhibits
        self.max_applications = max_applications
        self.random = random
        self.reward = reward
        self.compile()

    def from_dict(d, names_to_tiles):
        assert len(d) == 1
        name, d = list(d.items())[0]
        d['name'] = name
        in_out = np.stack((d['in_out']['in'], d['in_out']['out']), axis=0)
        _tile_from_str = partial(tile_from_str, names_to_tiles=names_to_tiles)
        in_out = np.vectorize(_tile_from_str)(in_out)
        d['in_out'] = in_out
        return Rule(**d)

    def to_dict(self):
        in_out = np.vectorize(tile_to_str)(self._in_out)
        inp, outp = in_out.tolist()
        # TODO: record application_funcs?
        return {
            self.name: {
                'in_out': {
                    'in': inp,
                    'out': outp,
                },
                'max_applications': self.max_applications,
                'rotate': self._rotate,
                'random': self.random,
                'reward': self.reward,
                'done': self.done,
                'inhibits': [t.name for t in self.inhibits],
                'children': [t.name for t in self.children],
            }
        }

    def compile(self):
        # List of subrules resulting from rule (i.e. if applying rotation).
        # in_out = np.vectorize(TileType.get_idx)(self._in_out)
        # subrules = [in_out]
        # (in_out, subpatterns, height, width)
        in_out = self._in_out
        subrules = [in_out]
        # Only rotate if necessary.
        if self._rotate and self._in_out[0].shape != (1, 1):
            subrules += [np.rot90(in_out, k=1, axes=(2, 3)), np.rot90(in_out, k=2, axes=(2,3)), 
                np.rot90(in_out, k=3, axes=(2,3))]
        self.subrules = subrules

    def observe(self, n_tiles):
        # return np.array([self.done, self.reward / 3, self.max_applications / 11, self.random, self._rotate,])
        in_out_disc = np.vectorize(TileType.get_idx)(self._in_out)

        # if in_out_disc.min() < 0:
        #     print('WARNING: negative tile index in rule observation. `TileNot` observations not supported. Hope you\'re not training on this!')
        in_out_disc = np.clip(in_out_disc, 0, n_tiles) 

        in_out_onehot = np.eye(n_tiles + 1)[in_out_disc]
        return in_out_onehot.flatten()

    def mutate_new(self, tiles, other_rules):
        # TODO: User GenEnv actions to modify map and rules (need to factor this out of GenEnv to be a static method)
        pass

    def mutate(self, tiles, other_rules):
        n_muts = 1
        x = random.random()
        if False:
            pass
        # if x < 1 / n_muts:
        #     return
        # elif x < 2 / n_muts:
        #     self.random = not self.random
        # elif x < 3 / n_muts:
        #     self.done = not self.done
        # elif x < 1 / n_muts:
        #     self.reward = random.randint(0, 3)
        # elif x < 5 / n_muts:
        #     self.max_applications = random.randint(0, 11)
        #     self.max_applications = math.inf if self.max_applications == 0 else self.max_applications
        # elif x < 6 / n_muts:
        #     self._rotate = not self._rotate
        # elif x < 7 / n_muts:
        #     self.inhibits = random.sample(other_rules, random.randint(0, len(other_rules)))
        # elif x < 8 / n_muts:
        #     self.children = random.sample(other_rules, random.randint(0, len(other_rules)))
        else:
            # if 1 < 1 / n_muts:
            if True:
                # Flip something in the in-out pattern.
                io_idx = random.randint(0, 1)
                subp_idx = random.randint(0, self._in_out.shape[1] - 1)
                if self._in_out.shape[2] == 0:
                    raise Exception('Cannot mutate rule with no subpatterns')
                i = random.randint(0, self._in_out.shape[2] - 1)
                j = random.randint(0, self._in_out.shape[3] - 1)
                tile = self._in_out[io_idx, subp_idx, i, j]
                tile_idx = tile.get_idx() if tile is not None else len(tiles)
                tiles_none = tiles + [None]
                new_in_out = self._in_out.copy()
                new_in_out[io_idx, subp_idx, i, j] = tiles_none[(tile_idx + random.randint(1, len(tiles) - 1)) % (len(tiles) + 1)]
            else:
                # Add something to the in-out pattern. Either a new subpattern, new rows, or new columns
                axis = random.randint(1, 3)
                diff = random.randint(0, 1)
                if diff == 0 and self._in_out.shape[axis] > 1 or self._in_out.shape[axis] == 3:
                    # Index of subpattern/row/column to be removed
                    i = random.randint(0, self._in_out.shape[axis] - 1)
                    new_in_out = np.delete(self._in_out, i, axis=axis)
                else:
                    new_shape = list(self._in_out.shape)
                    new_shape[axis] = 1
                    new = np.random.randint(0, len(tiles) + 1, new_shape)
                    # new = np.vectorize(lambda x: tiles[x] if x < len(tiles) else None)(new)
                    new = np.array(tiles + [None])[new]
                    new_in_out = np.concatenate((self._in_out, new), axis=axis)

            if is_valid(new_in_out):
                self._in_out = new_in_out
        self.compile()

    def hashable(self):
        int_in_out = np.vectorize(lambda x: x.get_idx() if x is not None else -1)(self._in_out)
        return hash(int_in_out.tobytes())


def is_valid(in_out):
    # Accept new in-out rule only if it does not result in invalid player transformation. 
    in_out_players = np.vectorize(lambda x: x is not None and x.is_player)(in_out)
    return in_out_players.sum() == 0 or in_out_players[0].sum() == 1 and in_out_players[1].sum() == 1

class ObjectRule(Rule):
    def __init__(self, *args, offset=(0, 0), **kwargs):
        super().__init__(*args, **kwargs)
        # Expect an object in the first input subpattern at the offset coordinates.
        assert isinstance(self._in_out[0, 0, offset[0], offset[0]], ObjectType.GameObject)

    def compile(self):
        super().compile()

class RuleSet(list):
    def __init__(self, rules: Iterable[Rule]):
        super().__init__(rules)
        [rule.compile() for rule in rules]

