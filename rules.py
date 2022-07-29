"""Rules determine a game's mechanics. They map input tile patterns to output patterns, rewards, done state (... and 
changes in global variables?).

During each step of the game engine, rules are applied in order, and the application of each rule involves scanning over
the tiles in the board in search of the relevant input pattern.


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
from typing import Iterable

import numpy as np

from tiles import TileType


class Rule():
    """Rules match input to output patterns. When the environment ticks, the input patters are identified and replaced 
    by the output patterns. Each rule may consist of multiple sub-rules, any one of which can be applied. Each rule may
    consist of multiple subpatterns, which may all match in order for the subrule to be applied.
    Input and output are 2D patches of tiles.
    """
    def __init__(self, name: str, in_out: Iterable[TileType], rotate: bool = True, reward: int = 0, done: bool = False):
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
        self.reward = reward
        self.done = done

    def compile(self):
        # List of subrules resulting from rule (i.e. if applying rotation).
        in_out_int = np.vectorize(TileType.get_idx)(self._in_out)
        subrules = [in_out_int]
        if self._rotate:
            subrules += [np.rot90(in_out_int, k=1, axes=(2, 3)), np.rot90(in_out_int, k=2, axes=(2,3)), 
                np.rot90(in_out_int, k=3, axes=(2,3))]
        self.subrules = subrules


class RuleSet(list):
    def __init__(self, rules: Iterable[Rule]):
        super().__init__(rules)
        [rule.compile() for rule in rules]