import copy
import dataclasses
from math import inf
from pdb import set_trace as TT
import random
from typing import Iterable

from fire import Fire
import gym
import jax
from jax import numpy as jnp
import numpy as np

from gen_env.games import (hamilton, maze, maze_backtracker, maze_npc, power_line, sokoban)
from gen_env.envs.play_env import EnvParams, EnvState, PlayEnv, Rule, TileType


# For debugging. Render every environment state that is visited during search.
RENDER = False


def solve_parallel(env: PlayEnv, state: EnvState, params: EnvParams,
                   max_steps: int = inf, render: bool = RENDER):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    key = jax.random.PRNGKey(0)
    frontier = [(state, [], 0)]
    visited = {}
    best_state_actions = None
    best_reward = -inf
    n_iter = 0
    n_iter_best = 0

    possible_actions = jnp.array(range(env.action_space.n))

    # Define a parallel version of the env.step_env function
    parallel_step_env = vmap(env.step_env, in_axes=(None, 0, None, None, None))

    while len(frontier) > 0:
        n_iter += 1
        if n_iter > max_steps:
            break

        # Find the idx of the best state in the frontier
        best_idx = np.argmax([f[2] for f in frontier])

        parent_state, parent_action_seq, parent_rew = frontier.pop(best_idx)
        visited[hash(env, parent_state)] = parent_rew

        # Generate new keys for each action to ensure randomness
        keys = jax.random.split(key, len(possible_actions))

        # Take steps in parallel
        states, obss, rews, dones, infos = parallel_step_env(keys, possible_actions, parent_state, params, render)

        for action, (state, obs, rew, done, info) in zip(possible_actions, zip(states, obss, rews, dones, infos)):
            child_rew = state.ep_rew
            action_seq = parent_action_seq + [action]
            hashed_state = hash(env, state)
            if hashed_state in visited and child_rew <= visited[hashed_state]:
                continue
            visited[hashed_state] = child_rew
            if child_rew > best_reward:
                best_state_actions = (state, action_seq)
                best_reward = child_rew
                n_iter_best = n_iter
            if not jnp.all(done):
                frontier.append((state, action_seq, child_rew))

    return best_state_actions, best_reward, n_iter_best, n_iter


def solve(env: PlayEnv, state: EnvState, params: EnvParams,
          max_steps: int = inf, render: bool = RENDER):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    key = jax.random.PRNGKey(0)
    frontier = [(state, [], 0)]
    visited = {}
    best_state_actions = None
    best_reward = -inf
    n_iter = 0
    n_iter_best = 0

    if isinstance(env.action_space, gym.spaces.Discrete):
        possible_actions = jnp.array(list(range(env.action_space.n)), dtype=jnp.int32)
    else:
        raise NotImplementedError

    while len(frontier) > 0:
        n_iter += 1
        if n_iter > max_steps:
            break

        # Find the idx of the best state in the frontier
        best_idx = np.argmax([f[2] for f in frontier])

        parent_state, parent_action_seq, parent_rew = frontier.pop(best_idx)
        visited[hash(env, parent_state)] = parent_rew
        # random.shuffle(possible_actions)

        step_key = jax.random.split(key, possible_actions.shape[0])
        v_state, v_obs, v_rew, v_done, v_info = \
            jax.vmap(env.step_env, in_axes=(None, None, 0, None))(
                key,
                parent_state,
                possible_actions,
                params)

        for i, action in enumerate(possible_actions):
            # state, obs, rew, done, info = \
            #     env.step_env(key=key, action=action, state=parent_state, params=params)
            child_rew = v_state.ep_rew[i]
            if render:
                env.render()
            action_seq = parent_action_seq + [action]
            state = jax.tree_map(lambda x: x[i], v_state)
            hashed_state = hash(env, state)
            if hashed_state in visited and child_rew <= visited[hashed_state]:
                continue
            visited[hashed_state] = child_rew
            if child_rew > best_reward:
                best_state_actions = (state, action_seq)
                best_reward = child_rew
                n_iter_best = n_iter
                # print(f'found new best: {best_reward} at {n_iter_best} iterations step {state.n_step} action sequence length {len(action_seq)}')
            if not jnp.all(v_done[i]):
                # Add this state to the frontier so can we can continue searching from it later
                frontier.append((state, action_seq, child_rew))

    return best_state_actions, best_reward, n_iter_best, n_iter


def hash(env: PlayEnv, state):
    return env.hashable(state)


def main(game=maze, height=10, width=10, render=False):
    if isinstance(game, str):
        game = globals()[game]

    env: PlayEnv = game.make_env(height=height, width=width)
    while True:
        sol = solve(env, render=render)

if __name__ == "__main__":
    Fire(main)
