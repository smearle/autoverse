"""This is a simple example demonstrating how to clone the behavior of an expert.
Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import os

import gym
import hydra
import imageio
import numpy as np
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import TransitionsMinimal
from imitation.data.wrappers import RolloutInfoWrapper

from evolve import init_base_env, load_game_to_env


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
    env.queue_maps([individual.map.copy()])
    obs = env.reset()
    frames = None
    if cfg.record:
        frames = [env.render(mode="rgb_array")]
    done = False
    total_reward = 0
    while not done:
        obs = flatten_obs(obs)
        action, _ = policy.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        if cfg.record:
            frames.append(env.render(mode="rgb_array"))
        total_reward += reward
    print(f"Reward: {total_reward}")
    return total_reward, frames

# TODO: move this inside env??
def flatten_obs(obs):
    return np.concatenate((obs['map'].flatten(), obs['player_rot'].flatten()))

# TODO: Should maybe just be save/loading policy instead of entire trainer(?)
def save(cfg, bc_trainer, curr_epoch):
    # Save transitions and bc_trainer with pickle
    pickle.dump(bc_trainer, open(os.path.join(cfg.log_dir, "bc_trainer.pkl"), "wb"))
    # Save current epoch number
    with open(os.path.join(cfg.log_dir, "epoch.txt"), "w") as f:
        f.write(str(curr_epoch))

def load(cfg):
    # Load transitions and bc_trainer
    bc_trainer = pickle.load(open(os.path.join(cfg.log_dir, "bc_trainer.pkl"), "rb"))
    # Load current epoch number
    with open(os.path.join(cfg.log_dir, "epoch.txt"), "r") as f:
        n_epoch = int(f.read())
    return bc_trainer, curr_epoch

@hydra.main(config_path="configs", config_name="il")
def main(cfg):
    cfg.log_dir = os.path.join(cfg.workspace, f"exp-{cfg.exp_id}")
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    env = init_base_env()
    elites = np.load("runs_evo/unique_elites.npz", allow_pickle=True)['arr_0']
    if cfg.overwrite:
        rng = np.random.default_rng(cfg.exp_id)
        obs = []
        acts = []
        for elite in elites:
            obs_seq = [flatten_obs(ob) for ob in elite.obs_seq[:-1]]
            obs.extend(obs_seq)
            acts.extend(elite.action_seq)
        obs = np.array(obs)
        acts = np.array(acts)
        infos = np.array([{} for _ in range(obs.shape[0])])
        assert obs.shape[0] == acts.shape[0]
        transitions = TransitionsMinimal(obs=obs, acts=acts, infos=infos)
        print(f"Loaded {transitions.obs.shape[0]} transitions from {len(elites)} playtraces.")

        # transitions = sample_expert_transitions()
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=rng,
            # custom_logger=cfg.log_dir,
        )
        curr_epoch = 0
        save(cfg, bc_trainer, curr_epoch)
    else:
        bc_trainer, curr_epoch = load(cfg)

    reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[0:10], name=f"epoch-{curr_epoch}")
    print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=cfg.n_epochs)

    reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[0:10], name=f"epoch-{cfg.n_epochs}")
    print(f"Reward after training: {reward}")


if __name__ == "__main__":
    main()