import copy
import os
from pdb import set_trace as TT
import random
import shutil
from typing import Iterable

from fire import Fire
import numpy as np
# import pool from ray
from ray.util.multiprocessing import Pool
import yaml

from games import *
from gen_env import GenEnv
from rules import Rule, RuleSet
from search_agent import solve
from tiles import TileSet, TileType


LOG_DIR = 'runs_evo'


def init_base_env():
    # env = evo_base.make_env(10, 10)
    env = maze_spike.make_env(10, 10)
    # env.search_tiles = [t for t in env.tiles]
    return env


class Individual():
    def __init__(self, tiles: Iterable[TileType], rules: Iterable[Rule]):
        self.tiles = tiles
        self.rules = rules
        self.fitness = None

    def mutate(self):
        i_arr = np.random.randint(0, len(self.rules) - 1, random.randint(1, 3))
        for i in i_arr:
            rule: Rule = self.rules[i]
            rule.mutate(self.tiles, self.rules[:i] + self.rules[i+1:])
            self.rules[i] = rule
        j_arr = np.random.randint(0, len(self.tiles) - 1, random.randint(0, 3))
        for j in j_arr:
            tile: TileType = self.tiles[j]
            tile.mutate(self.tiles[:j] + self.tiles[j+1:])

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


def load_game_to_env(env: GenEnv, individual: Individual):
    env.tiles = individual.tiles
    env.rules = individual.rules
    return env


def evaluate_multi(args):
    return evaluate(*args)


def evaluate(env: GenEnv, individual: Individual, render: bool, trg_n_iter: bool):
    load_game_to_env(env, individual)
    env.reset()
    init_state = env.get_state()
    best_state_actions, best_reward, n_iter_best, n_iter = solve(env, max_steps=trg_n_iter)
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
    fitness = n_iter_best
    if fitness == 1:
        fitness += n_iter / (trg_n_iter + 2)
    print(f"Achieved fitness {fitness} at {n_iter_best} iterations with {best_reward} reward. Searched for {n_iter} iterations total.")
    return fitness


def main(exp_name='0', overwrite=False, load=False, multi_proc=False, render=False):
    log_dir = os.path.join(LOG_DIR, exp_name)
    loaded = False
    if os.path.isdir(log_dir):
        if load:
            save_dict = np.load(os.path.join(log_dir, 'elites.npz'), allow_pickle=True)['arr_0'].item()
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
        trg_n_iter = 10
        os.makedirs(log_dir)

    batch_size = 10
    env = init_base_env()
    if multi_proc:
        envs = [init_base_env() for _ in range(batch_size)]

    if not loaded:
        n_gen = 0
        tiles = env.tiles
        rules = env.rules
        ind = Individual(tiles, rules)
        # ind.fitness = evaluate(env, ind)
        elites = []
        for _ in range(pop_size):
            o = copy.deepcopy(ind)
            o.mutate()
            elites.append(o)
        if not multi_proc:
            for e in elites:
                e.fitness = evaluate(env, e, render, trg_n_iter)
        else:
            with Pool(processes=10) as pool:
                # fits = pool.map(evaluate_multi, [(env, elite, render) for env, elite in zip(envs, elites)])
                fits = pool.map(evaluate_multi, [(env, elite, render, trg_n_iter) for elite in elites])
                for elite, fit in zip(elites, fits):
                    elite.fitness = fit

    for n_gen in range(n_gen, 10000):
        parents = np.random.choice(elites, size=batch_size, replace=True)
        offspring = []
        for p in parents:
            o: Individual = copy.deepcopy(p)
            o.mutate()
            offspring.append(o)
        if not multi_proc:
            for o in offspring:
                o.fitness = evaluate(env, o, render, trg_n_iter)
        else:
            with Pool(processes=10) as pool:
                fits = pool.map(evaluate_multi, [(env, ind, render, trg_n_iter) for env, ind in zip(envs, offspring)])
                for o, fit in zip(offspring, fits):
                    o.fitness = fit
        elites = np.concatenate((elites, offspring))
        # Discard the weakest.
        elite_idxs = np.argpartition(np.array([o.fitness for o in elites]), batch_size)[:batch_size]
        elites = np.delete(elites, elite_idxs)
        fits = [e.fitness for e in elites]
        # Print stats about elites.
        max_fit = max(fits)
        print(f"Generation {n_gen}")
        print(f"Best fitness: {max_fit}")
        print(f"Average fitness: {np.mean(fits)}")
        print(f"Median fitness: {np.median(fits)}")
        print(f"Worst fitness: {min(fits)}")
        print(f"Standard deviation: {np.std(fits)}")
        print()
        # Increment trg_n_iter if the best fitness is within 10 of it.
        if max_fit > trg_n_iter - 10:
            trg_n_iter *= 2
        # Save the elites.
        np.savez(os.path.join(log_dir, "elites"), 
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
    Fire(main)
