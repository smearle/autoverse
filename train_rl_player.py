from functools import partial
import os

import hydra
import numpy as np
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import torch as th
from configs.config import Config

from evolve import init_base_env
from utils import validate_config


# Don't need to inherit yet...
class PPO(SB3PPO):
    # def __init__(self, env_archive, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        pass


# Don't need to callback. (But would prefer, to allow swapping in agents)
def callback(_locals, _globals):
    pass


@hydra.main(version_base="1.3", config_path="configs", config_name="rl")
def main(cfg: Config):
    validate_config(cfg)
    n_iters = cfg.n_iters
    env_archive = np.load(os.path.join(cfg.log_dir_common, "unique_elites.npz"), allow_pickle=True)['arr_0']
    maps = [ind.map for ind in env_archive]
    rules = [ind.rules for ind in env_archive]

    model_state_dict = None

    if cfg.load_rl:
    # if os.path.exists(os.path.join(cfg.log_dir, "policy_rl.pt")):
        model_state_dict = th.load(os.path.join(cfg.log_dir_rl, "policy_rl.pt"))

    if cfg.load_il:
        with open(os.path.join(cfg.log_dir_il, "policy_il.pt"), "rb") as f:
            model_state_dict = th.load(f)

    # Now take the imitation-learned policy and do RL with it using sb3 PPO
    make_env = partial(init_base_env, cfg)
    env: SubprocVecEnv = make_vec_env(make_env, n_envs=cfg.n_proc, vec_env_cls=SubprocVecEnv)
    n_envs = len(env.remotes)

    # Divide the maps and rules between the environments
    maps_batched = np.array_split(maps, n_envs)
    rules_batched = np.array_split(rules, n_envs)

    for i, r in enumerate(env.remotes):
        maps_i, rules_i = maps_batched[i], rules_batched[i]
        r.send(('env_method', (('queue_games', (maps_i, rules_i), {}))))
        ret = r.recv()

    # model = PPO.load(os.path.join(cfg.log_dir_rl, 'policy'))
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=cfg.log_dir_rl, policy_kwargs=policy_kwargs)
    optimizer = th.optim.Adam(model.policy.parameters(), lr=1e-4)

    if model_state_dict is not None:
        model.set_parameters({'policy': model_state_dict, 'policy.optimizer': optimizer.state_dict()})

    model.learn(total_timesteps=cfg.n_iters, tb_log_name="ppo", callback=callback)
    model.save(os.path.join(cfg.log_dir_rl, f"policy_rl_epoch-{cfg.n_iters}.pt"))


if __name__ == "__main__":
    main()
