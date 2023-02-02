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

from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import MlpPolicy

import imitation
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import TransitionsMinimal
from imitation.data.wrappers import RolloutInfoWrapper

from evolve import init_base_env, load_game_to_env
from games import maze


def evaluate_policy_on_elites(cfg, policy, env, elites, name):
    net_rew = 0
    for e_idx, elite in enumerate(elites):
        rew, frames = evaluate_policy(cfg, policy, env, elite)
        net_rew += rew
        if cfg.record:
            # Save frames to video
            imageio.mimsave(os.path.join(cfg.log_dir, f"{name}_elite-{e_idx}.mp4"), frames, fps=10)
    return net_rew / len(elites)

def evaluate_policy(cfg, policy, env, individual):
    load_game_to_env(env, individual)
    # assert individual.map[4].sum() == 0, "Force tile should not be present in map"
    env.queue_maps([individual.map.copy()])
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
    bc_trainer.save_policy(os.path.join(cfg.log_dir, "policy"))
    # Save current epoch number
    with open(os.path.join(cfg.log_dir, "epoch.txt"), "w") as f:
        f.write(str(curr_epoch))

def load(cfg):
    # Load transitions and bc_trainer
    # bc_trainer = pickle.load(open(os.path.join(cfg.log_dir, "bc_trainer.pkl"), "rb"))
    # bc_trainer._bc_logger._logger = pickle.load(open(os.path.join(cfg.log_dir, "logger.pkl"), "rb"))
    policy = bc.reconstruct_policy(os.path.join(cfg.log_dir, "policy"))
    # Load current epoch number
    with open(os.path.join(cfg.log_dir, "epoch.txt"), "r") as f:
        curr_epoch = int(f.read())
    return policy, curr_epoch
    # return bc_trainer, curr_epoch

@hydra.main(config_path="configs", config_name="il")
def main(cfg):
    cfg.log_dir = os.path.join(cfg.workspace, f"exp-{cfg.exp_id}")
    # Environment class doesn't matter and will be overwritten when we load in an individual.
    # env = maze.make_env(10, 10)
    env = init_base_env(cfg)
    rng = np.random.default_rng(cfg.exp_id)
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    # HACK to load trained run after refactor
    import evo
    import sys
    individual = evo.individual
    sys.modules['individual'] = individual
    # end HACK

    elites = np.load("runs_evo/unique_elites.npz", allow_pickle=True)['arr_0']
    policy_kwargs = dict(net_arch=[64, 64], observation_space=env.observation_space, action_space=env.action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: th.finfo(th.float32).max,)
    if cfg.overwrite or not os.path.exists(os.path.join(cfg.log_dir, "transitions.npz")):
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
        np.savez(os.path.join(cfg.log_dir, "transitions.npz"), {
            'obs': transitions.obs,
            'acts': transitions.acts,
            'infos': transitions.infos,
            })
        curr_epoch = 0

        # policy=None
        policy = MlpPolicy(**policy_kwargs)


    else:
        # progress_df = pd.read_csv(os.path.join(cfg.log_dir, "progress.csv"))
        transitions = np.load(os.path.join(cfg.log_dir, "transitions.npz"), allow_pickle=True)['arr_0'].item()
        transitions = TransitionsMinimal(
            obs=transitions['obs'],
            acts=transitions['acts'],
            infos=transitions['infos'],
        )

        policy, curr_epoch = load(cfg)
        # bc_trainer, curr_epoch = load(cfg)

    custom_logger = imitation.util.logger.configure(
        os.path.join(cfg.log_dir, "logs"),
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

    def on_epoch_end(bc_trainer, cfg, base_n_epoch):
        curr_epoch = base_n_epoch + bc_trainer._bc_logger._current_epoch
        save(cfg, bc_trainer, curr_epoch)

    on_epoch_end = partial(on_epoch_end, bc_trainer=bc_trainer, cfg=cfg, base_n_epoch=curr_epoch)

    if cfg.overwrite:
        save(cfg, bc_trainer, curr_epoch)
        reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[-10:], name=f"epoch-{curr_epoch}")
        print(f"Reward before imitation learning: {reward}")
        with open(os.path.join(cfg.log_dir, f"epoch-{curr_epoch}_reward.txt"), "w") as f:
            f.write(str(reward))
    
    else:
        bc_trainer._bc_logger._tensorboard_step = curr_epoch

    print("Training a policy using Behavior Cloning")
    n_train_epochs = cfg.n_epochs - curr_epoch
    if n_train_epochs > 0:
        bc_trainer.train(n_epochs=n_train_epochs, on_epoch_end=on_epoch_end)

        save(cfg, bc_trainer, cfg.n_epochs)

        reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[-10:], name=f"epoch-{cfg.n_epochs}")
        print(f"Reward after imitation learning: {reward}")
        with open(os.path.join(cfg.log_dir, f"epoch-{cfg.n_epochs}_reward.txt"), "w") as f:
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

    # Now take the imitation-learned policy and do RL with it using sb3 PPO
    from stable_baselines3.common.env_util import make_vec_env
    policy = bc_trainer.policy
    make_env = partial(init_base_env, cfg)
    env = make_vec_env(make_env, n_envs=100, vec_env_cls=SubprocVecEnv)
    # model = PPO.load(os.path.join(cfg.log_dir, 'policy'))
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=cfg.log_dir, policy_kwargs=policy_kwargs)
    model.set_parameters({'policy': policy.state_dict(), 'policy.optimizer': policy.optimizer.state_dict()})
    model.learn(total_timesteps=1000000, tb_log_name="ppo")
    model.save(os.path.join(cfg.log_dir, "ppo"))


if __name__ == "__main__":
    main()