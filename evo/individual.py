import random
from typing import Dict, Iterable
import yaml

from einops import rearrange
import numpy as np

from rules import Rule, RuleSet
from tiles import TileType, TileSet


class Individual():
    def __init__(self, cfg, tiles: Iterable[TileType], rules: Iterable[Rule], map: np.ndarray):
        self.cfg = cfg
        self.tiles = tiles
        self.rules = rules
        self.map = map
        self.fitness = None
        self.action_seq = None

    def mutate(self):
        if self.cfg.mutate_rules:
            # Mutate between 1 and 3 random rules
            # Assume we're evolving rules, leave 2 base rules intact
            i_arr = np.random.randint(2, len(self.rules), random.randint(1, 3))
            for i in i_arr:
                rule: Rule = self.rules[i]
                rule.mutate(self.tiles, self.rules[:i] + self.rules[i+1:])
                self.rules[i] = rule
        # Mutate between 0 and 3 random tiles
        # j_arr = np.random.randint(0, len(self.tiles) - 1, random.randint(0, 3))
        # for j in j_arr:
        #     tile: TileType = self.tiles[j]
        #     if tile.is_player:
        #         continue
        #     other_tiles = [t for t in self.tiles[:j] + self.tiles[j+1:] if not t.is_player]
        #     tile.mutate(other_tiles)

        # Mutate onehot map by randomly changing some tile types
        # Pick number of tiles to sample from gaussian
        n_mut_tiles = abs(int(np.random.normal(0, 10)))
        disc_map = self.map.argmax(axis=0)
        k_arr = np.random.randint(0, disc_map.size - 1, n_mut_tiles)
        for k in k_arr:
            disc_map.flat[k] = np.random.randint(0, len(self.tiles))

        fixed_num_tiles = [t for t in self.tiles if t.num is not None]
        free_num_tile_idxs = [t.idx for t in self.tiles if t.num is None]
        # For tile types with fixed numbers, make sure this many occur
        for tile in fixed_num_tiles:
            # If there are too many, remove some
            # print(f"Checking {tile.name} tiles")
            idxs = np.where(disc_map.flat == tile.idx)[0]
            # print(f"Found {len(idxs)} {tile.name} tiles")
            if len(idxs) > tile.num:
                # print(f'Found too many {tile.name} tiles, removing some')
                for idx in idxs[tile.num:]:
                    disc_map.flat[idx] = np.random.choice(free_num_tile_idxs)
                # print(f'Removed {len(idxs) - tile.num} tiles')
                assert len(np.where(disc_map == tile.idx)[0]) == tile.num
            elif len(idxs) < tile.num:
                # FIXME: Not sure if this is working
                net_idxs = []
                chs_i = 0
                np.random.shuffle(free_num_tile_idxs)
                while len(net_idxs) < tile.num:
                    # Replace only 1 type of tile (weird)
                    idxs = np.where(disc_map.flat == free_num_tile_idxs[chs_i])[0]
                    net_idxs += idxs.tolist()
                    chs_i += 1
                    if chs_i >= len(free_num_tile_idxs):
                        print(f"Warning: Not enough tiles to mutate into {tile.name} tiles")
                        break
                idxs = np.array(net_idxs[:tile.num])
                for idx in idxs:
                    disc_map.flat[idx] = tile.idx
                assert len(np.where(disc_map == tile.idx)[0]) == tile.num
        for tile in fixed_num_tiles:
            assert len(np.where(disc_map == tile.idx)[0]) == tile.num
        self.map = rearrange(np.eye(len(self.tiles))[disc_map], 'h w c -> c h w')

    def save(self, filename):
        # Save dictionary to yaml
        with open(filename, 'w') as f:
            d = {'tiles': [t.to_dict() for t in self.tiles], 'rules': [r.to_dict() for r in self.rules]}
            yaml.safe_dump(d, f, indent=4, allow_unicode=False)

    def load(filename):
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
        return Individual(tiles=tiles, rules=rules)

    def hashable(self):
        rule_hashes = [r.hashable() for r in self.rules]
        rules_hash = hash((tuple(rule_hashes)))
        map_hash = hash(self.map.tobytes())
        return hash((rules_hash, map_hash))
