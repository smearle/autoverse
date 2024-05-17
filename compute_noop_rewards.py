

import os
import pickle
import jax
from jax import numpy as jnp

from gen_env.configs.config import GenEnvConfig
from gen_env.envs.play_env import PlayEnv
from gen_env.evo.individual import IndividualPlaytraceData
from gen_env.utils import init_base_env, init_config
from utils import load_elite_envs


def main(cfg: GenEnvConfig):
    init_config(cfg)
    env, env_params = init_base_env(cfg)
    env: PlayEnv
    rng = jax.random.PRNGKey(cfg.seed)

    train_elites, val_elites, test_elites = load_elite_envs(cfg, cfg.load_gen)

    def step_env_noop(carry, _):
        # Hardcoded to select a rotation action
        action = env.ROTATE_LEFT_ACTION
        obs, state = carry
        obs, state, reward, done, info, params = env.step(rng, state, action, env_params, env_params) 
        return (obs, state), reward
    
    def eval_elite_noop(elite, _):
        params = e.env_params
        obs, state = env.reset(rng, params) 
        rewards = jax.lax.scan(step_env_noop, (obs, state), env.max_episode_steps)
        ep_reward = rewards.sum()
        return ep_reward

    for elites in [train_elites, val_elites, test_elites]:
        for e in elites:
            e: IndividualPlaytraceData
            n_elites = e.env_params.rule_dones.shape[0]

            _, ep_rewards = jax.lax.scan(eval_elite_noop, e, jnp.arange(n_elites))
            e.noop_ep_rew = ep_rewards

    # Save elite files under names above
    with open(os.path.join(cfg._log_dir_common, f"gen-{cfg.load_gen}_filtered_train_elites.pkl"), 'wb') as f:
        pickle.dump(train_elites, f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{cfg.load_gen}_filtered_val_elites.pkl"), 'wb') as f:
        pickle.dump(val_elites, f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{cfg.load_gen}_filtered_test_elites.pkl"), 'wb') as f:
        pickle.dump(test_elites, f)


if __name__ == '__main__':
    main()