import copy
from enum import unique
import glob
import os
from pdb import set_trace as TT
import random
import shutil
from typing import Iterable

# from fire import Fire
from einops import rearrange
import hydra
import imageio
import numpy as np
# import pool from ray
# from ray.util.multiprocessing import Pool
from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
import yaml

from games import GAMES
from play_env import PlayEnv
from evo.eval import load_game_to_env, evaluate_multi, evaluate
from evo.individual import Individual
from rules import Rule, RuleSet
from search_agent import solve
from tiles import TileSet, TileType


def init_base_env(cfg):
    env = GAMES[cfg.game].make_env(10, 10)
    # env = evo_base.make_env(10, 10)
    # env = maze.make_env(10, 10)
    # env = maze_for_evo.make_env(10, 10)
    # env = maze_spike.make_env(10, 10)
    # env = sokoban.make_env(10, 10)
    # env.search_tiles = [t for t in env.tiles]
    return env

# @dataclass
# class Playtrace:
#     obs_seq: List[np.ndarray]
#     action_seq: List[int]
#     reward_seq: List[float]

def aggregate_playtraces(cfg):
    unique_elites_path = os.path.join(cfg.log_dir, 'unique_elites.npz')

    # If overwriting, or elites have not previously been aggregated, then collect all unique games.
    if cfg.overwrite or not os.path.isfile(unique_elites_path):
        # Aggregate all playtraces into one file
        elite_files = glob.glob(os.path.join(cfg.log_dir, 'gen-*.npz'))
        # An elite is a set of game rules, a game map, and a solution/playtrace
        # elite_hashes = set()
        elites = {}
        n_evaluated = 0
        for f in elite_files:
            save_dict = np.load(f, allow_pickle=True)['arr_0'].item()
            elites_i = save_dict['elites']
            for elite in elites_i:
                n_evaluated += 1
                e_hash = elite.hashable()
                if e_hash not in elites or elites[e_hash].fitness < elite.fitness:
                    elites[e_hash] = elite
        print(f"Aggregated {len(elites)} unique elites from {n_evaluated} evaluated individuals.")
        # Replay episodes, recording obs and rewards and attaching to individuals
        env = init_base_env(cfg)
        elites = list(elites.values())
        for elite in elites:
            # assert elite.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
            frames = replay_episode(cfg, env, elite)
        # Save unique elites to npz file
        np.savez(unique_elites_path, elites)
    # If not overwriting, load existing elites
    else:
        # Load elites from file
        elites = np.load(unique_elites_path, allow_pickle=True)['arr_0']
    # Additionally save elites to workspace directory for easy access for imitation learning
    np.savez(os.path.join(cfg.workspace, 'unique_elites.npz'), elites)

def replay_episode(cfg, env, elite):
    # Re-play the episode, recording observations and rewards (for imitation learning)
    # print(f"Fitness: {elite.fitness}")
    # print(f"Action sequence: {elite.action_seq}")
    load_game_to_env(env, elite)
    obs_seq = []
    rew_seq = []
    env.queue_maps([elite.map.copy()])
    obs = env.reset()
    # assert env.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
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
        if i >= len(elite.action_seq):
            # print('Warning: action sequence too short. Ending episode before env is done. This must mean no solution was found.')
            # breakpoint()
            break
        obs, reward, done, info = env.step(elite.action_seq[i])
        obs_seq.append(obs)
        rew_seq = rew_seq + [reward]
        if cfg.record:
            frames.append(env.render(mode='rgb_array'))
        if cfg.render:
            env.render(mode='human')
        i += 1
    if i < len(elite.action_seq):
        print('Warning: action sequence too long.')
        # breakpoint()
    elite.obs_seq = obs_seq
    elite.rew_seq = rew_seq
    if cfg.record:
        return frames

def get_log_dir(cfg):
    return os.path.join(cfg.workspace, cfg.game, f"{'mutRule_' if cfg.mutate_rules else ''}exp-{cfg.exp_id}")


# def main(exp_id='0', overwrite=False, load=False, multi_proc=False, render=False):
@hydra.main(version_base='1.3', config_path="configs", config_name="evo")
def main(cfg):
    overwrite, num_proc, render = cfg.overwrite, cfg.num_proc, cfg.render
    if cfg.record:
        cfg.evaluate=True
    cfg.log_dir = get_log_dir(cfg)
    load = not overwrite
    if cfg.aggregate_playtraces:
        aggregate_playtraces(cfg)
        return
    loaded = False
    if os.path.isdir(cfg.log_dir):
        if load:
            if cfg.load_gen is not None:
                save_file = os.path.join(cfg.log_dir, f'gen-{int(cfg.load_gen)}.npz')
            else:
                # Get `gen-xxx.npz` with largest `xxx`
                save_files = glob.glob(os.path.join(cfg.log_dir, 'gen-*.npz'))
                save_file = max(save_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

            # HACK to load trained run after refactor
            import evo
            import sys
            individual = evo.individual
            sys.modules['individual'] = individual
            # end HACK

            save_dict = np.load(save_file, allow_pickle=True)['arr_0'].item()
            n_gen = save_dict['n_gen']
            elites = save_dict['elites']
            trg_n_iter = save_dict['trg_n_iter']
            pop_size = len(elites)
            loaded = True
            print(f"Loaded {len(elites)} elites from {save_file} at generation {n_gen}.")
        elif not overwrite:
            print(f"Directory {cfg.log_dir} already exists. Use `--overwrite=True` to overwrite.")
            return
        else:
            shutil.rmtree(cfg.log_dir, ignore_errors=True)
    if not loaded:
        pop_size = cfg.batch_size
        trg_n_iter = 1000 # Max number of iterations while searching for solution. Will increase during evolution
        os.makedirs(cfg.log_dir)

    env = init_base_env(cfg)
    env.reset()
    if num_proc > 1:
        envs = [init_base_env(cfg) for _ in range(num_proc)]

    if cfg.evaluate:
        print(f"Elites at generation {n_gen}:")
        for e_idx, e in enumerate(elites[:10]):
            frames = replay_episode(cfg, env, e)
            if cfg.record:
                # imageio.mimsave(os.path.join(log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.gif"), frames, fps=10)
                # Save as mp4
                imageio.mimsave(os.path.join(cfg.log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.mp4"), frames, fps=10)
                # Save elite as yaml
                e.save(os.path.join(cfg.log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.yaml"))
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
        ind = Individual(cfg, tiles, rules, map)
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
    writer = SummaryWriter(log_dir=cfg.log_dir)
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
            # trg_n_iter *= 2
            trg_n_iter += 1000
        if n_gen % cfg.save_freq == 0: 
            # Save the elites.
            np.savez(os.path.join(cfg.log_dir, f"gen-{n_gen}"),
            # np.savez(os.path.join(log_dir, "elites"), 
                {
                    'n_gen': n_gen,
                    'elites': elites,
                    'trg_n_iter': trg_n_iter
                })
            # Save the elite's game mechanics to a yaml
            elite_games_dir = os.path.join(cfg.log_dir, "elite_games")
            if not os.path.isdir(elite_games_dir):
                os.mkdir(os.path.join(cfg.log_dir, "elite_games"))
            for i, e in enumerate(elites):
                e.save(os.path.join(elite_games_dir, f"{i}.yaml"))

if __name__ == '__main__':
    main()
