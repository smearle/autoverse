import random
from typing import Dict, Iterable, List
import yaml

import chex
from einops import rearrange
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np

from gen_env.envs.play_env import GenEnvParams, PlayEnv
from gen_env.configs.config import GenEnvConfig
from gen_env.rules import Rule, RuleData, RuleSet, mutate_rules
from gen_env.tiles import TileType, TileSet


@struct.dataclass
class IndividualData:
    env_params: GenEnvParams
    fitnesses: Iterable[float]
    # bc_0: float
    # bc_1: float
    action_seqs: Iterable[chex.Array]
    # obs_seq: chex.Array


# Will contain only the best playtrace, for offline learning
@struct.dataclass
class IndividualPlaytraceData:
    env_params: GenEnvParams
    fitness: float
    action_seq: chex.Array
    obs_seq: chex.Array
    rew_seq: chex.Array

    
def hash_individual(individual: IndividualData) -> int:
    hashable_elite = jax.tree_map(lambda x: np.array(x).tobytes(), individual)
    # Return flattened leaf nodes of pytree
    flat_hashable_elite: List = jax.tree_leaves(hashable_elite)
    return hash(frozenset(flat_hashable_elite))

class Individual():
    def __init__(self, cfg: GenEnvConfig, tiles: Iterable[TileType]):
        self.cfg = cfg
        # self.tiles = tiles
        # self.rules = rules
        # for rule in self.rules:
        #     rule.n_tile_types = len(self.tiles)
        #     rule.compile()
        # self.map = map
        self.fitness = None

        self.obs_seq = None
        self.action_seq = None
        self.reward_seq = None

        self.player_idx = None
        for t in tiles:
            if t.is_player:
                self.player_idx = t.idx
        assert self.player_idx == 0

    def mutate(self, key, map, rules, tiles):

        # def mutate_rules(key, rules, tiles) -> RuleData:

        #     def _mutate_rule(key, rule, rule_reward, tiles):
        #         rule_int, rule_reward = mutate_rule(key, rule, rule_reward, tiles)
        #         rules = RuleData(rule=rule_int, reward=rule_reward)
        #         return rules

        #     mutate_keys = jax.random.split(key, rules.rule.shape[0])

        #     mutated_rules = jax.vmap(_mutate_rule, in_axes=(0, 0, 0, None))\
        #         (mutate_keys, rules.rule, rules.reward, tiles)

        #     # Uniform sample probability of masking out rule mutations
        #     mask_pct = jax.random.uniform(key, shape=(1,), minval=0.0, maxval=0.5)
        #     mask = jax.random.bernoulli(key, p=mask_pct, shape=rules.reward.shape)
        #     mutated_rules = mutated_rules.replace(
        #         rule=jax.tree_map(lambda x: jnp.where(mask[...,None,None,None,None,None], x, 0), mutated_rules.rule),
        #         reward=jax.tree_map(lambda x: jnp.where(mask, x, 0), mutated_rules.reward),
        #     )

        #     return mutated_rules

        rules = jax.lax.cond(self.cfg.mutate_rules, lambda key, rules: mutate_rules(key, rules), lambda _, __: rules, key, rules)

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
        # n_mut_tiles = abs(int(np.random.normal(0, 10)))
        # disc_map = self.map.argmax(axis=0)
        # k_arr = np.random.randint(0, disc_map.size - 1, n_mut_tiles)

        flip_pct = jax.random.uniform(key, shape=(), minval=0.0, maxval=0.5)
        bit_flips = jax.random.bernoulli(key, p=flip_pct, shape=map.shape)

        # Mask out bit flips at `player_idx`        
        bit_flips = bit_flips.at[self.player_idx].set(0)

        map = map.astype(jnp.int16)
        map = jnp.bitwise_xor(map, bit_flips)

        # for k in k_arr:
            # breakpoint()
            # disc_map.flat[k] = np.random.randint(0, len(self.tiles))
        
        # self.map = PlayEnv.repair_map(disc_map, self.tiles)
        # self.map = PlayEnv.repair_map(key, map, fixed_tile_nums=fixed_tile_nums)

        return map, rules


    # TODO: bring this up to speed
    # def save(self, filename):
    #     # Save dictionary to yaml
    #     with open(filename, 'w') as f:
    #         d = {'tiles': [t.to_dict() for t in self.tiles], 'rules': [r.to_dict() for r in self.rules],
    #              'map': self.map.tolist()}
    #         yaml.safe_dump(d, f, indent=4, allow_unicode=False)

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
        return IndividualData(tiles=tiles, rules=rules, cfg=cfg, map=map)

    # TODO: bring this up to speed
    # def hashable(self):
    #     rule_hashes = [r.hashable() for r in self.rules]
    #     rules_hash = hash((tuple(rule_hashes)))
    #     map_hash = hash(self.map.tobytes())
    #     return hash((rules_hash, map_hash))
