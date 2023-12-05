import random
from typing import Dict, Iterable
import yaml

import chex
from einops import rearrange
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np

from gen_env.envs.play_env import PlayEnv
from gen_env.configs.config import GenEnvConfig
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TileType, TileSet


class Individual():
    def __init__(self, cfg: GenEnvConfig, tiles: Iterable[TileType], rules: Iterable[Rule], map: np.ndarray):
        self.cfg = cfg
        self.tiles = tiles
        self.rules = rules
        for rule in self.rules:
            rule.n_tile_types = len(self.tiles)
            rule.compile()
        self.map = map
        self.fitness = None

        self.obs_seq = None
        self.action_seq = None
        self.reward_seq = None

    def mutate(self, key, fixed_tile_nums):
        if self.cfg.mutate_rules:
            # Mutate between 1 and 3 random rules
            i_arr = np.random.randint(0, len(self.rules), random.randint(1, 3))
            for i in i_arr:
                rule: Rule = self.rules[i]
                rule.mutate(self.tiles, self.rules[:i] + self.rules[i+1:])
                self.rules[i] = rule

        # if not hasattr(self.cfg, 'fix_map') or not self.cfg.fix_map:
        # if not self.cfg.fix_map:
            # Mutate between 0 and 3 random tiles
            # j_arr = np.random.randint(0, len(self.tiles) - 1, random.randint(0, 3))
            # for j in j_arr:
            #     tile: TileType = self.tiles[j]
            #     if tile.is_player:
            #         continue
            #     other_tiles = [t for t in self.tiles[:j] + self.tiles[j+1:] if not t.is_player]
            #     tile.mutate(other_tiles)


            # TODO: This should be multi-hot (repairing impossible co-occurrences after). 
            # Currently evolving a onehot initial map, adding co-occurrences later.

        # Mutate onehot map by randomly changing some tile types
        # Pick number of tiles to sample from gaussian
        n_mut_tiles = abs(int(np.random.normal(0, 10)))
        # disc_map = self.map.argmax(axis=0)
        # k_arr = np.random.randint(0, disc_map.size - 1, n_mut_tiles)
        flip_pct = jax.random.uniform(key, shape=(), minval=0.0, maxval=0.5)
        bit_flips = jax.random.bernoulli(key, p=flip_pct, shape=self.map.shape)
        map = jnp.bitwise_xor(self.map, bit_flips)
        # for k in k_arr:
            # breakpoint()
            # disc_map.flat[k] = np.random.randint(0, len(self.tiles))
        
        # self.map = PlayEnv.repair_map(disc_map, self.tiles)
        self.map = PlayEnv.repair_map(key, map, fixed_tile_nums=fixed_tile_nums)

        if self.map is None:
            breakpoint()


    def save(self, filename):
        # Save dictionary to yaml
        with open(filename, 'w') as f:
            d = {'tiles': [t.to_dict() for t in self.tiles], 'rules': [r.to_dict() for r in self.rules],
                 'map': self.map.tolist()}
            yaml.safe_dump(d, f, indent=4, allow_unicode=False)

    def load(filename, cfg):
        # Load dictionary from yaml
        with open(filename, 'r') as f:
            d = yaml.safe_load(f)
            tiles = []
            for t_dict in d['tiles']:
                assert len(t_dict) == 1
                name = list(t_dict.keys())[0]
                t_dict = t_dict[name]
                t_dict.update({'name': name})
                tiles.append(TileType.from_dict(t_dict))
            tiles = TileSet(tiles)
            names_to_tiles = {t.name: t for t in tiles}
            rules = [Rule.from_dict(r, names_to_tiles=names_to_tiles) for r in d['rules']]
            for t in tiles:
                t.cooccurs = [names_to_tiles[c] for c in t.cooccurs]
                t.inhibits = [names_to_tiles[i] for i in t.inhibits]
            names_to_rules = {r.name: r for r in rules}
            for r in rules:
                r.children = [names_to_rules[c] for c in r.children]
                r.inhibits = [names_to_rules[i] for i in r.inhibits]
            rules = RuleSet(rules)
            map = np.array(d['map'])
        return Individual(tiles=tiles, rules=rules, cfg=cfg, map=map)

    def hashable(self):
        rule_hashes = [r.hashable() for r in self.rules]
        rules_hash = hash((tuple(rule_hashes)))
        map_hash = hash(self.map.tobytes())
        return hash((rules_hash, map_hash))
