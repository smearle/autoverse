"""This is a simple example demonstrating how to clone the behavior of an expert.
Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
from functools import partial
import os

import gym
import hydra
import imageio
import numpy as np
import pandas as pd
import pickle
import torch as th

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

import imitation
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import TransitionsMinimal
from imitation.data.wrappers import RolloutInfoWrapper

from configs.config import Config
from evolve import init_base_env, load_game_to_env
from games import maze
from play_env import PlayEnv
from utils import validate_config


def evaluate_policy_on_elites(cfg: Config, policy, env, elites, name):
    vid_dir = os.path.join(cfg.log_dir_il, "videos")
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    net_rew = 0
    for e_idx, elite in enumerate(elites):
        rew, frames = evaluate_policy(cfg, policy, env, elite)
        net_rew += rew
        if cfg.record:
            # Save frames to video
            imageio.mimsave(os.path.join(vid_dir, f"{name}_elite-{e_idx}.mp4"), frames, fps=10)
    return net_rew / len(elites)

def evaluate_policy(cfg, policy, env: PlayEnv, individual):
    load_game_to_env(env, individual)
    # assert individual.map[4].sum() == 0, "Force tile should not be present in map"
    env.queue_games([individual.map.copy()], [individual.rules.copy()])
    obs = env.reset()
    frames = None
    if cfg.record:
        frames = [env.render(mode="rgb_array")]
    done = False
    total_reward = 0
    while not done:
        action, _ = policy.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        if cfg.record:
            frames.append(env.render(mode="rgb_array"))
        total_reward += reward
    # print(f"Reward: {total_reward}")
    return total_reward, frames

# TODO: Should maybe just be save/loading policy instead of entire trainer(?)
def save(cfg, bc_trainer: bc.BC, curr_epoch):
    # Save logger separately
    # pickle.dump(bc_trainer._bc_logger._logger, open(os.path.join(cfg.log_dir, "logger.pkl"), "wb"))
    # Save transitions and bc_trainer with pickle
    # pickle.dump(bc_trainer, open(os.path.join(cfg.log_dir, "bc_trainer.pkl"), "wb"))
    bc_trainer.save_policy(os.path.join(cfg.log_dir_il, "policy"))
    # Save current epoch number
    with open(os.path.join(cfg.log_dir_il, "epoch.txt"), "w") as f:
        f.write(str(curr_epoch))

def load(cfg):
    # Load transitions and bc_trainer
    # bc_trainer = pickle.load(open(os.path.join(cfg.log_dir, "bc_trainer.pkl"), "rb"))
    # bc_trainer._bc_logger._logger = pickle.load(open(os.path.join(cfg.log_dir, "logger.pkl"), "rb"))
    policy = bc.reconstruct_policy(os.path.join(cfg.log_dir_il, "policy"))
    # Load current epoch number
    with open(os.path.join(cfg.log_dir, "epoch.txt"), "r") as f:
        # curr_epoch = int(f.read())
        curr_batch = int(f.read())
    return policy, curr_batch

@hydra.main(version_base="1.3", config_path="configs", config_name="il")
def main(cfg: Config):
    validate_config(cfg)
    # Environment class doesn't matter and will be overwritten when we load in an individual.
    # env = maze.make_env(10, 10)
    env = init_base_env(cfg)
    rng = np.random.default_rng(cfg.exp_id)
    if not os.path.exists(cfg.log_dir_il):
        os.makedirs(cfg.log_dir_il)

    # Initialize tensorboard logger
    logger = th.utils.tensorboard.SummaryWriter(cfg.log_dir_il)

    # HACK to load trained run after refactor
    import evo
    import sys
    individual = evo.individual
    sys.modules['individual'] = individual
    # end HACK

    elites = np.load(os.path.join(cfg.log_dir_common, "unique_elites.npz"), allow_pickle=True)['arr_0']
    policy_kwargs = dict(net_arch=[64, 64], observation_space=env.observation_space, action_space=env.action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: th.finfo(th.float32).max,)
    if cfg.overwrite or not os.path.exists(os.path.join(cfg.log_dir_common, "transitions.npz")):
        # progress_df = pd.DataFrame()
        cfg.overwrite = True
        obs = []
        acts = []
        for elite in elites:
            # obs_seq = [ob for ob in elite.obs_seq[:-1]]
            obs_seq = elite.obs_seq[:-1]
            # if len(obs_seq) != len(elite.action_seq):
            #     breakpoint()
            assert len(obs_seq) == len(elite.action_seq)
            obs.extend(obs_seq)
            acts.extend(elite.action_seq)
        obs = np.array(obs)
        acts = np.array(acts)
        infos = np.array([{} for _ in range(obs.shape[0])])
        assert obs.shape[0] == acts.shape[0]
        transitions = TransitionsMinimal(obs=obs, acts=acts, infos=infos)
        print(f"Loaded {transitions.obs.shape[0]} transitions from {len(elites)} playtraces.")
        # Save the transitions with pickle
        np.savez(os.path.join(cfg.log_dir_common, "transitions.npz"), {
            'obs': transitions.obs,
            'acts': transitions.acts,
            'infos': transitions.infos,
            })
        # curr_epoch = 0
        curr_batch = 0

        # policy=None
        policy = MlpPolicy(**policy_kwargs)


    else:
        # progress_df = pd.read_csv(os.path.join(cfg.log_dir, "progress.csv"))
        transitions = np.load(os.path.join(cfg.log_dir_common, "transitions.npz"), allow_pickle=True)['arr_0'].item()
        transitions = TransitionsMinimal(
            obs=transitions['obs'],
            acts=transitions['acts'],
            infos=transitions['infos'],
        )

        policy, curr_batch = load(cfg)

    custom_logger = imitation.util.logger.configure(
        os.path.join(cfg.log_dir_il, "logs"),
        ["stdout", "tensorboard"],
    )

    bc_trainer = bc.BC(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        custom_logger=custom_logger,
    )

    # HACK
    assert not hasattr(bc_trainer._bc_logger, '_current_batch')
    bc_trainer._bc_logger._current_batch = curr_batch

    def on_epoch_end(bc_trainer, cfg, base_n_epoch):
        curr_epoch = base_n_epoch + bc_trainer._bc_logger._current_epoch
        save(cfg, bc_trainer, curr_epoch)

    def on_batch_end(bc_trainer, cfg, base_n_batch):
        curr_batch = base_n_batch + bc_trainer._bc_logger._current_batch
        if curr_batch % cfg.save_freq == 0:
            save(cfg, bc_trainer, curr_batch)

    # on_epoch_end = partial(on_epoch_end, bc_trainer=bc_trainer, cfg=cfg, base_n_epoch=curr_epoch)
    on_batch_end = partial(on_batch_end, bc_trainer=bc_trainer, cfg=cfg, base_n_batch=curr_batch)

    if cfg.overwrite:
        # save(cfg, bc_trainer, curr_epoch)
        save(cfg, bc_trainer, curr_batch)
    
    else:
        # bc_trainer._bc_logger._tensorboard_step = curr_epoch
        bc_trainer._bc_logger._tensorboard_step = curr_batch

    # reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[-10:], name=f"epoch-{curr_epoch}")
    reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[-10:], name=f"batch-{curr_batch}")
    # print(f"Reloaded epoch {curr_epoch}.\nReward before imitation learning: {reward}")
    print(f"Reloaded batch {curr_batch}.\nReward before imitation learning: {reward}")
    # with open(os.path.join(cfg.log_dir, f"epoch-{curr_epoch}_reward.txt"), "w") as f:
    with open(os.path.join(cfg.log_dir_il, f"epoch-{curr_batch}_reward.txt"), "w") as f:
        f.write(str(reward))

    print("Training a policy using Behavior Cloning")
    # n_train_epochs = cfg.n_epochs - curr_epoch
    n_train_batches = cfg.n_il_batches - curr_batch
    if n_train_batches > 0:
        bc_trainer.train(
            n_epochs=None, 
            n_batches=n_train_batches, 
            # on_epoch_end=on_epoch_end, 
            on_batch_end=on_batch_end
        )

        save(cfg, bc_trainer, cfg.n_il_batches)

        reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[-10:], name=f"batch-{cfg.n_il_batches}")
        print(f"Reward after imitation learning: {reward}")
        with open(os.path.join(cfg.log_dir_il, f"batch-{cfg.n_il_batches}_reward.txt"), "w") as f:
            f.write(str(reward))
        
        # Access the logged data
        # logger = bc_trainer.logger
        # bc_log_dir = logger.dir
        # Load progress.csv
        # progress_df_new = pd.read_csv(os.path.join(bc_log_dir, "progress.csv"), index_col=False)
        # Add to existing dataframe
        # breakpoint()
        # progress_df = pd.concat([progress_df, progress_df_new], ignore_index=True)

        # Add number of rows in saved df to row indices of new df
        # progress_df_new.index += progress_df.shape[0]

        # Save dataframe
        # progress_df.to_csv(os.path.join(cfg.log_dir, "progress.csv"))

    policy = bc_trainer.policy

    # Save the state dict
    th.save(policy.state_dict(), os.path.join(cfg.log_dir_il, "policy_il.pt"))



if __name__ == "__main__":
    main()