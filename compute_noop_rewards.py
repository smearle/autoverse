from functools import partial
import os
import pickle

import hydra
import jax
from jax import numpy as jnp

from gen_env.configs.config import GenEnvConfig
from gen_env.envs.play_env import PlayEnv
from gen_env.evo.individual import IndividualPlaytraceData
from gen_env.utils import init_base_env, init_config
from utils import load_elite_envs


def step_env_noop(carry, _, env):
    rng = jax.random.PRNGKey(0)  # inconsequential
    # Hardcoded to select a rotation action
    action = env.ROTATE_LEFT_ACTION
    obs, state, env_params = carry
    obs, state, reward, done, info, env_params_idx = env.step(rng, state, action, env_params, env_params) 
    return (obs, state, env_params), reward


def step_env_random(carry, _, env):
    rng = jax.random.PRNGKey(0)  # inconsequential
    # Hardcoded to select a rotation action
    action = env.action_space.sample()
    obs, state, env_params = carry
    obs, state, reward, done, info, env_params_idx = env.step(rng, state, action, env_params, env_params) 
    return (obs, state, env_params), reward


def eval_elite_noop(params, env):
    rng = jax.random.PRNGKey(0)  # inconsequential
    _step_env_noop = partial(step_env_noop, env=env)
    obs, state = env.reset(rng, params) 
    _, rewards = jax.lax.scan(_step_env_noop, (obs, state, params), None, env.max_episode_steps)
    ep_reward = rewards.sum()
    return ep_reward


def eval_elite_random(params, env, n_eps=100):
    rng = jax.random.PRNGKey(0)  # inconsequential
    _step_env_random = partial(step_env_random, env=env)
    obs, state = env.reset(rng, params) 
    _, rewards = jax.lax.scan(_step_env_random, (obs, state, params), None, env.max_episode_steps * n_eps)
    ep_reward_mean = rewards.mean()
    ep_reward_std = rewards.std()
    ep_reward_max = rewards.max()
    return ep_reward_mean, ep_reward_std, ep_reward_max


@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def main(cfg: GenEnvConfig):
    init_config(cfg)
    env, dummy_params = init_base_env(cfg)
    env: PlayEnv
    _eval_elite_noop = partial(eval_elite_noop, env=env)
    _eval_elite_random = partial(eval_elite_random, env=env)
    rng = jax.random.PRNGKey(cfg.seed)

    train_elites, val_elites, test_elites = load_elite_envs(cfg, cfg.load_gen)

    new_elite_sets = []
    for e in [train_elites, val_elites, test_elites]:
        e: IndividualPlaytraceData
        n_elites = e.env_params.rule_dones.shape[0]
        e_params = e.env_params

        # Backward compatibility HACK
        e_params = e_params.replace(rew_scale=jnp.ones((n_elites,)), rew_bias=jnp.zeros((n_elites,)))
        # TODO: May need to batch this vmapping to prevent OOM
        noop_ep_rewards = jax.vmap(_eval_elite_noop, in_axes=(0))(e_params)
        e = e.replace(env_params=e.env_params.replace(noop_ep_rew=noop_ep_rewards))
        random_ep_reward_means, random_ep_reward_stds, random_ep_reward_maxs = \
            jax.vmap(_eval_elite_random, in_axes=(0))(e_params)
        e = e.replace(env_params=e.env_params.replace(random_ep_rew=random_ep_reward_means))
        e = e.replace(env_params=e.env_params.replace(search_ep_rew=e.rew_seq.sum(1)))

        # stacked_ep_rews = jnp.stack([e.env_params.noop_ep_rew, e.env_params.random_ep_rew, e.env_params.search_ep_rew[:, 0]], axis=1)
        stacked_ep_rews = jnp.stack([e.env_params.noop_ep_rew, random_ep_reward_maxs], axis=1)

        best_ep_rew = jnp.max(stacked_ep_rews, axis=1)
        worst_ep_rew = jnp.min(stacked_ep_rews, axis=1)

        reward_bias = -random_ep_reward_means
        e = e.replace(env_params=e.env_params.replace(rew_bias=reward_bias))
        # Note that since we have integer rewards, this will always be a decimal
        # reward_scale = 1 / random_ep_reward_stds
        reward_scale = 1 / (best_ep_rew - worst_ep_rew)
        reward_scale = jnp.where(jnp.isinf(reward_scale), 1, reward_scale)
        e = e.replace(env_params=e.env_params.replace(rew_scale=reward_scale))
        
        new_elite_sets.append(e)

    train_elites, val_elites, test_elites = new_elite_sets
    # Save elite files under names above
    with open(os.path.join(cfg._log_dir_common, f"gen-{cfg.load_gen}_filtered_train_elites.pkl"), 'wb') as f:
        pickle.dump(train_elites, f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{cfg.load_gen}_filtered_val_elites.pkl"), 'wb') as f:
        pickle.dump(val_elites, f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{cfg.load_gen}_filtered_test_elites.pkl"), 'wb') as f:
        pickle.dump(test_elites, f)


if __name__ == '__main__':
    main()