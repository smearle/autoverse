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
import jax
import numpy as np
# import pool from ray
# from ray.util.multiprocessing import Pool
from multiprocessing import Pool
from tensorboardX import SummaryWriter

from gen_env.configs.config import Config
from gen_env.games import GAMES
from gen_env.envs.play_env import EnvParams, PlayEnv
from gen_env.evo.eval import evaluate_multi, evaluate
from gen_env.evo.individual import Individual
from gen_env.rules import is_valid
from gen_env.utils import get_params_from_individual, init_base_env, load_game_to_env, validate_config



# @dataclass
# class Playtrace:
#     obs_seq: List[np.ndarray]
#     action_seq: List[int]
#     reward_seq: List[float]

def collect_elites(cfg: Config):

    # If overwriting, or elites have not previously been aggregated, then collect all unique games.
    # if cfg.overwrite or not os.path.isfile(unique_elites_path):
    # Aggregate all playtraces into one file
    elite_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
    # Get the highest generation number
    gen_nums = [int(f.split('-')[-1].split('.')[0]) for f in elite_files]
    latest_gen = max(gen_nums)
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

    vid_dir = os.path.join(cfg._log_dir_evo, 'debug_videos')
    os.makedirs(vid_dir, exist_ok=True)
    # Replay the episode, storing the obs and action sequences to the elite.
    for e_idx, elite in enumerate(elites):
    #     # assert elite.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
        frames = replay_episode(cfg, env, elite, record=False)

        # Will only have returned frames in case of funky error, for debugging
        if frames is not None:
            imageio.mimsave(os.path.join(vid_dir, f"elite-{e_idx}_fitness-{elite.fitness}.mp4"), frames, fps=10)
            frames_2 = replay_episode(cfg, env, elite, record=False)
            imageio.mimsave(os.path.join(vid_dir, f"elite-{e_idx}_fitness-{elite.fitness}_take2.mp4"), frames_2, fps=10)
            breakpoint()


    # Sort elites by increasing fitness

    if not os.path.isdir(cfg._log_dir_player_common):
        os.mkdir(cfg._log_dir_player_common)

    train_elites, val_elites, test_elites = split_elites(cfg, elites)
    # Save elites to file
    np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_train_elites.npz'), train_elites)
    np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_val_elites.npz'), val_elites)
    np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_test_elites.npz'), test_elites)

    # Save unique elites to npz file
    # If not overwriting, load existing elites
    # else:
    #     # Load elites from file
    #     elites = np.load(unique_elites_path, allow_pickle=True)['arr_0']

    # if not os.path.isdir(cfg._log_dir_player_common):
    #     os.mkdir(cfg._log_dir_playecommonv)

    # Additionally save elites to workspace directory for easy access for imitation learning
    # np.savez(unique_elites_path, elites)

def split_elites(cfg: Config, elites: Iterable[Individual]):
    """ Split elites into train, val and test sets."""
    elites.sort(key=lambda x: x.fitness, reverse=True)

    n_elites = len(elites)
    # n_train = int(n_elites * .8)
    # n_val = int(n_elites * .1)
    # n_test = n_elites - n_train - n_val

    # Sample train/val/test sets from elites with a range of fitness values. Every `n`th elite is sampled.
    # This ensures that the train/val/test sets are diverse. No elites can be in multiple sets.
    train_elites = []
    val_elites = []
    test_elites = []
    for i in range(n_elites):
        if i % 10 == 0:
            val_elites.append(elites[i])
        elif (i + 1) % 10 == 0:
            test_elites.append(elites[i])
        else:
            train_elites.append(elites[i])

    n_train = len(train_elites)
    n_val = len(val_elites)
    n_test = len(test_elites)

    # train_elites = elites[:n_train]
    # val_elites = elites[n_train:n_train+n_val]
    # test_elites = elites[n_train+n_val:]
    print(f"Split {n_elites} elites into {n_train} train, {n_val} val, {n_test} test.")
    return train_elites, val_elites, test_elites


def replay_episode(cfg: Config, env: PlayEnv, elite: Individual, 
                   record: bool = False):
    """Re-play the episode, recording observations and rewards (for imitation learning)."""
    # print(f"Fitness: {elite.fitness}")
    params = get_params_from_individual(env, elite)
    load_game_to_env(env, elite)
    obs_seq = []
    rew_seq = []
    # env.queue_games([elite.map.copy()], [elite.rules.copy()])
    key = jax.random.PRNGKey(0)
    state, obs = env.reset(key=key, params=params)
    # print(f"Initial state reward: {state.ep_rew}")
    # assert env.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    # Debug: interact after episode completes (investigate why episode ends early)
    # env.render(mode='pygame')
    # while True:
    #     env.tick_human()
    obs_seq.append(obs)
    if record:
        frames = [env.render(mode='rgb_array', state=state, params=params)]
    if cfg.render:
        env.render(mode='human', state=state)
    done = False
    i = 0
    while not done:
        if i >= len(elite.action_seq):
            # FIXME: Problem with player death...?
            print('Warning: action sequence too short. Ending episode before env is done. Probably because of cap on '
                  'search iterations.')
            # breakpoint()
            # if not record:
            #     print('Replaying again, rendering this time')
            #     return replay_episode(cfg, env, elite, record=True)
            break
        state, obs, reward, done, info = env.step(key, state=state, action=elite.action_seq[i], params=params)
        # print(state.ep_rew)
        obs_seq.append(obs)
        rew_seq = rew_seq + [reward]
        if record:
            frames.append(env.render(mode='rgb_array', state=state, params=params))
        if cfg.render:
            env.render(mode='human', state=state, params=params)
        i += 1
    if i < len(elite.action_seq):
        # FIXME: Problem with player death...?
        # raise Exception("Action sequence too long.")
        print('Warning: action sequence too long.')
        if not record:
            print('Replaying again, rendering this time')
            return replay_episode(cfg, env, elite, record=True)
        # breakpoint()
    elite.obs_seq = obs_seq
    elite.rew_seq = rew_seq
    if record:
        return frames


# def main(exp_id='0', overwrite=False, load=False, multi_proc=False, render=False):
@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def main(cfg: Config):
    validate_config(cfg)
    vid_dir = os.path.join(cfg._log_dir_evo, 'videos')
    
    overwrite, n_proc, render = cfg.overwrite, cfg.n_proc, cfg.render

    if overwrite:
        # Use input to overwrite
        # ovr_bool = input(f"Directory {cfg._log_dir_evo} already exists. Overwrite? (y/n)")
        # if ovr_bool == 'y':
        shutil.rmtree(cfg._log_dir_evo, ignore_errors=True)
        # else:
            # return

    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    # if cfg.record:
    #     cfg.evaluate=True
    load = not overwrite
    if cfg.collect_elites:
        collect_elites(cfg)
        return
    loaded = False
    if os.path.isdir(cfg._log_dir_evo):
        ckpt_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
        if len(ckpt_files) == 0:
            print(f'No checkpoints found in {cfg._log_dir_evo}. Starting from scratch')
        elif load:
            if cfg.load_gen is not None:
                save_file = os.path.join(cfg._log_dir_evo, f'gen-{int(cfg.load_gen)}.npz')
            else:
                # Get `gen-xxx.npz` with largest `xxx`
                save_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
                save_file = max(save_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

            # HACK to load trained run after refactor
            # from gen_env import evo
            # from gen_env import configs
            # from gen_env import tiles, rules
            # import sys
            # individual = evo.individual
            # sys.modules['individual'] = individual
            # sys.modules['evo'] = evo
            # sys.modules['configs'] = configs
            # sys.modules['tiles'] = tiles
            # sys.modules['rules'] = rules
            # end HACK

            save_dict = np.load(save_file, allow_pickle=True)['arr_0'].item()
            n_gen = save_dict['n_gen']
            elites = save_dict['elites']
            trg_n_iter = save_dict['trg_n_iter']
            pop_size = len(elites)
            loaded = True
            print(f"Loaded {len(elites)} elites from {save_file} at generation {n_gen}.")
        elif not overwrite:
            print(f"Directory {cfg._log_dir_evo} already exists. Use `--overwrite=True` to overwrite.")
            return
        else:
            shutil.rmtree(cfg._log_dir_il, ignore_errors=True)
    if not loaded:
        pop_size = cfg.batch_size
        trg_n_iter = 200 # Max number of iterations while searching for solution. Will increase during evolution
        os.makedirs(cfg._log_dir_evo, exist_ok=True)

    env, params = init_base_env(cfg)
    key = jax.random.PRNGKey(0)
    env_state, obs = env.reset(key=key, params=params)
    # if num_proc > 1:
    #     envs, params = zip(*[init_base_env(cfg) for _ in range(num_proc)])
    #     breakpoint()
        # envs = [init_base_env(cfg) for _ in range(num_proc)]

    if cfg.evaluate:
        # breakpoint()
        print(f"Elites at generation {n_gen}:")
        eval_elites(cfg, env, elites, n_gen=n_gen, vid_dir=vid_dir)
        return

    def multiproc_eval_offspring(offspring):
        eval_offspring = []
        while len(offspring) > 0:
            envs = [env for _ in range(len(offspring))]
            eval_offspring += pool.map(evaluate_multi, [(env, ind, render, trg_n_iter) for env, ind in zip(envs, offspring)])
            offspring = offspring[cfg.n_proc:]
        return eval_offspring

    fixed_tile_nums = np.array([t.num if t.num is not None else -1 for t in env.tiles])

    # Initial population
    if not loaded:
        n_gen = 0
        tiles = env.tiles
        rules = env.rules
        map = env_state.map
        ind = Individual(cfg, tiles, rules, map)
        offspring = []
        for _ in range(pop_size):
            o = copy.deepcopy(ind)
            key, subkey = jax.random.split(key)
            o.mutate(key=key, fixed_tile_nums=fixed_tile_nums)
            offspring.append(o)
        if n_proc == 1:
            for o in offspring:
                o = evaluate(key, env, o, render, trg_n_iter)
        else:
            with Pool(processes=n_proc) as pool:
                offspring = multiproc_eval_offspring(offspring)
        elites = offspring

    # Training loop
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=cfg._log_dir_evo)
    for n_gen in range(n_gen, 10000):
        parents = np.random.choice(elites, size=cfg.batch_size, replace=True)
        offspring = []
        for p in parents:
            o: Individual = copy.deepcopy(p)
            offspring.append(o)
            for rule in o.rules:
                if not is_valid(rule._in_out):
                    breakpoint()
        # if n_proc == 1:
        for o in offspring:
            o = evaluate(key, env, o, render, trg_n_iter)
        # else:
        #     with Pool(processes=n_proc) as pool:
        #         offspring = multiproc_eval_offspring(offspring)

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
        # if max_fit > trg_n_iter - 10:
        if max_fit > trg_n_iter * 0.5:
            # trg_n_iter *= 2
            trg_n_iter += 200
        if n_gen % cfg.save_freq == 0: 
            # Save the elites.
            np.savez(os.path.join(cfg._log_dir_evo, f"gen-{n_gen}"),
            # np.savez(os.path.join(log_dir, "elites"), 
                {
                    'n_gen': n_gen,
                    'elites': elites,
                    'trg_n_iter': trg_n_iter
                })
            # Save the elite's game mechanics to a yaml
            elite_games_dir = os.path.join(cfg._log_dir_evo, "elite_games")
            if not os.path.isdir(elite_games_dir):
                os.mkdir(os.path.join(cfg._log_dir_evo, "elite_games"))
            for i, e in enumerate(elites):
                e.save(os.path.join(elite_games_dir, f"{i}.yaml"))

        if n_gen % cfg.eval_freq == 0:
            eval_elites(cfg, env, elites, n_gen=n_gen, vid_dir=vid_dir)

def eval_elites(cfg: Config, env: PlayEnv, elites: Iterable[Individual], n_gen: int, vid_dir: str):
    """ Evaluate elites."""
    # Sort elites by fitness.
    elites = sorted(elites, key=lambda e: e.fitness, reverse=True)
    for e_idx, e in enumerate(elites[:10]):
        frames = replay_episode(cfg, env, e, record=cfg.record)
        if cfg.record:
            # imageio.mimsave(os.path.join(log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.gif"), frames, fps=10)
            # Save as mp4
            # imageio.mimsave(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.mp4"), frames, fps=10)
            imageio.mimsave(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.mp4"), frames, fps=10)
            # Save elite as yaml
            e.save(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.yaml"))


if __name__ == '__main__':
    main()
