import copy
import glob
import os
from pdb import set_trace as TT
import random
import shutil
from typing import Iterable

from fire import Fire
from einops import rearrange
import hydra
import imageio
import numpy as np
# import pool from ray
from ray.util.multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
import yaml

from games import *
from gen_env import GenEnv
from rules import Rule, RuleSet
from search_agent import solve
from tiles import TileSet, TileType


def init_base_env():
    # env = evo_base.make_env(10, 10)
    env = maze.make_env(10, 10)
    # env = maze_spike.make_env(10, 10)
    # env = sokoban.make_env(10, 10)
    # env.search_tiles = [t for t in env.tiles]
    return env


class Individual():
    def __init__(self, tiles: Iterable[TileType], rules: Iterable[Rule], map: np.ndarray):
        self.tiles = tiles
        self.rules = rules
        self.map = map
        self.fitness = None
        self.action_seq = None

    def mutate(self):
        # Mutate between 1 and 3 random rules
        # i_arr = np.random.randint(0, len(self.rules) - 1, random.randint(1, 3))
        # for i in i_arr:
        #     rule: Rule = self.rules[i]
        #     rule.mutate(self.tiles, self.rules[:i] + self.rules[i+1:])
        #     self.rules[i] = rule
        # Mutate between 0 and 3 random tiles
        # j_arr = np.random.randint(0, len(self.tiles) - 1, random.randint(0, 3))
        # for j in j_arr:
        #     tile: TileType = self.tiles[j]
        #     if tile.is_player:
        #         continue
        #     other_tiles = [t for t in self.tiles[:j] + self.tiles[j+1:] if not t.is_player]
        #     tile.mutate(other_tiles)

        # Mutate onehot map by randomly changing some tile types
        disc_map = self.map.argmax(axis=0)
        k_arr = np.random.randint(0, disc_map.size - 1, random.randint(0, self.map.size // 2))
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
                # assert len(np.where(disc_map == tile.idx)[0]) == tile.num
            elif len(idxs) < tile.num:
                # print(f'Found too few {tile.name} tiles, adding some')
                for idx in idxs[:tile.num - len(idxs)]:
                    disc_map.flat[idx] = tile.idx
                assert len(np.where(disc_map == tile.idx)[0]) == tile.num
        # for tile in fixed_num_tiles:
            # assert len(np.where(disc_map == tile.idx)[0]) == tile.num
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


# @dataclass
# class Playtrace:
#     obs_seq: List[np.ndarray]
#     action_seq: List[int]
#     reward_seq: List[float]


def load_game_to_env(env: GenEnv, individual: Individual):
    env.tiles = individual.tiles
    env.rules = individual.rules
    return env


def evaluate_multi(args):
    return evaluate(*args)


def evaluate(env: GenEnv, individual: Individual, render: bool, trg_n_iter: bool):
    load_game_to_env(env, individual)
    env.queue_maps([individual.map.copy()])
    env.reset()
    init_state = env.get_state()
    best_state_actions, best_reward, n_iter_best, n_iter = solve(env, max_steps=trg_n_iter)
    action_seq = None
    if best_state_actions is not None:
        (final_state, action_seq) = best_state_actions
        if render:
            env.set_state(init_state)
            env.render()
            for action in action_seq:
                env.step(action)
                env.render()
    # TODO: dummy
    # fitness = best_reward
    # fitness = n_iter_best
    # if fitness == 1:
    #     fitness += n_iter / (trg_n_iter + 2)
    fitness = len(action_seq) if action_seq is not None else 0
    individual.fitness = fitness
    individual.action_seq = action_seq
    print(f"Achieved fitness {fitness} at {n_iter_best} iterations with {best_reward} reward. Searched for {n_iter} iterations total.")
    return individual


# def main(exp_name='0', overwrite=False, load=False, multi_proc=False, render=False):
@hydra.main(config_path="configs", config_name="evo")
def main(cfg):
    exp_name, overwrite, num_proc, render = cfg.exp_name, cfg.overwrite, cfg.num_proc, cfg.render
    load = not overwrite
    log_dir = os.path.join(cfg.log_dir, str(exp_name))
    loaded = False
    if os.path.isdir(log_dir):
        if load:
            if cfg.load_gen is not None:
                save_file = os.path.join(log_dir, f'gen-{int(cfg.load_gen)}.npz')
            else:
                # Get `gen-xxx.npz` with largest `xxx`
                save_files = glob.glob(os.path.join(log_dir, 'gen-*.npz'))
                save_file = max(save_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            save_dict = np.load(save_file, allow_pickle=True)['arr_0'].item()
            n_gen = save_dict['n_gen']
            elites = save_dict['elites']
            trg_n_iter = save_dict['trg_n_iter']
            pop_size = len(elites)
            loaded = True
        elif not overwrite:
            print(f"Directory {log_dir} already exists. Use `--overwrite=True` to overwrite.")
            return
        else:
            shutil.rmtree(log_dir)
    if not loaded:
        pop_size = 10
        trg_n_iter = 100_000
        os.makedirs(log_dir)

    env = init_base_env()
    env.reset()
    if num_proc > 1:
        envs = [init_base_env() for _ in range(num_proc)]

    if cfg.evaluate:
        print(f"Elites at generation {n_gen}:")
        for e_idx, e in enumerate(elites):
            # Re-play the episode, recording observations and rewards (for imitation learning)
            print(f"Fitness: {e.fitness}")
            print(f"Action sequence: {e.action_seq}")
            obs_seq = []
            rew_seq = []
            env.queue_maps([e.map])
            obs = env.reset()
            # Debug: interact after episode completes (investigate why episode ends early)
            # env.render(mode='pygame')
            # while True:
            #     env.tick_human()
            obs_seq.append(obs)
            if cfg.record:
                frames = [env.render(mode='rgb_array')]
            if cfg.render:
                env.render(mode='human')
            done = False
            i = 0
            while not done:
                if i >= len(e.action_seq):
                    print('Warning: action sequence too short. Ending episode before env is done.')
                    break
                obs, reward, done, info = env.step(e.action_seq[i])
                obs_seq.append(obs)
                rew_seq = rew_seq + [reward]
                if cfg.record:
                    frames.append(env.render(mode='rgb_array'))
                if cfg.render:
                    env.render(mode='human')
                i += 1
            if cfg.record:
                # imageio.mimsave(os.path.join(log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.gif"), frames, fps=10)
                # Save as mp4
                imageio.mimsave(os.path.join(log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.mp4"), frames, fps=10)
            e.obs_seq = obs_seq
            e.rew_seq = rew_seq
            env.queue_maps([e.map])
            obs = env.reset()
        return

    def multiproc_eval_offspring(offspring):
        eval_offspring = []
        while len(offspring) > 0:
            eval_offspring += pool.map(evaluate_multi, [(env, ind, render, trg_n_iter) for env, ind in zip(envs, offspring)])
            offspring = offspring[cfg.num_proc:]
        return eval_offspring

    # Initial population
    if not loaded:
        n_gen = 0
        tiles = env.tiles
        rules = env.rules
        map = env.map
        ind = Individual(tiles, rules, map)
        offspring = []
        for _ in range(pop_size):
            o = copy.deepcopy(ind)
            o.mutate()
            offspring.append(o)
        if num_proc == 1:
            for o in offspring:
                o = evaluate(env, o, render, trg_n_iter)
        else:
            with Pool(processes=num_proc) as pool:
                offspring = multiproc_eval_offspring(offspring)
        elites = offspring

    # Training loop
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    for n_gen in range(n_gen, 10000):
        parents = np.random.choice(elites, size=cfg.batch_size, replace=True)
        offspring = []
        for p in parents:
            o: Individual = copy.deepcopy(p)
            o.mutate()
            offspring.append(o)
        if num_proc == 1:
            for o in offspring:
                o = evaluate(env, o, render, trg_n_iter)
        else:
            with Pool(processes=num_proc) as pool:
                offspring = multiproc_eval_offspring(offspring)

        elites = np.concatenate((elites, offspring))
        # Discard the weakest.
        for e in elites:
            if o.fitness is None:
                raise ValueError("Fitness is None.")
        elite_idxs = np.argpartition(np.array([o.fitness for o in elites]), cfg.batch_size)[:cfg.batch_size]
        elites = np.delete(elites, elite_idxs)
        fits = [e.fitness for e in elites]
        max_fit = max(fits)
        mean_fit = np.mean(fits)
        min_fit = min(fits) 
        # Log stats to tensorboard.
        writer.add_scalar('fitness/best', max_fit, n_gen)
        writer.add_scalar('fitness/mean', mean_fit, n_gen)
        writer.add_scalar('fitness/min', min_fit, n_gen)
        # Print stats about elites.
        print(f"Generation {n_gen}")
        print(f"Best fitness: {max_fit}")
        print(f"Average fitness: {mean_fit}")
        print(f"Median fitness: {np.median(fits)}")
        print(f"Worst fitness: {min_fit}")
        print(f"Standard deviation: {np.std(fits)}")
        print()
        # Increment trg_n_iter if the best fitness is within 10 of it.
        if max_fit > trg_n_iter - 10:
            trg_n_iter *= 2
        if n_gen % 10 == 0: 
            # Save the elites.
            np.savez(os.path.join(log_dir, f"gen-{n_gen}"),
            # np.savez(os.path.join(log_dir, "elites"), 
                {
                    'n_gen': n_gen,
                    'elites': elites,
                    'trg_n_iter': trg_n_iter
                })
            # Save the elite's game mechanics to a yaml
            elite_games_dir = os.path.join(log_dir, "elite_games")
            if not os.path.isdir(elite_games_dir):
                os.mkdir(os.path.join(log_dir, "elite_games"))
            for i, e in enumerate(elites):
                e.save(os.path.join(elite_games_dir, f"{i}.yaml"))

if __name__ == '__main__':
    main()
