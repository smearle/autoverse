
from functools import partial
import json
import os
import shutil

import hydra
import jax
import numpy as np

from evaluate import eval_elite_nn, eval_nn
from gen_env.configs.config import ILConfig, RLConfig
from gen_env.evo.individual import IndividualPlaytraceData
from gen_env.utils import init_base_env, init_config
from il_player_jax import init_bc_agent
from pcgrl_utils import get_network
from utils import init_il_config, init_rl_config, load_elite_envs


@hydra.main(config_path="gen_env/configs", config_name="il")
def eval_rl(cfg: RLConfig):
    init_config(cfg)
    latest_gen = init_il_config(cfg)
    init_rl_config(cfg)
    assert latest_gen is not None, \
            "Must select an evo-gen from which to load playtraces for imitation learning when running IL script." +\
            "Set `load_gen=-1` to load latest generation for which playtraces have been aggregated."

    rng = jax.random.PRNGKey(cfg.seed)
    # Environment class doesn't matter and will be overwritten when we load in an individual.
    # env = maze.make_env(10, 10)
    env, env_params = init_base_env(cfg)

    network = get_network(env, env_params, cfg)
    init_x = env.gen_dummy_obs(env_params)
    # init_x = env.observation_space(env_params).sample(_rng)[None, ]
    network_params = network.init(rng, init_x)
    apply_fn = network.apply
    eval_nn(cfg, latest_gen=latest_gen, env=env, apply_fn=apply_fn, network_params=network_params, n_eps=100)
        
if __name__ == '__main__':
    eval_rl()