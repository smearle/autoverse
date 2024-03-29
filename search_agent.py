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


def solve(env: PlayEnv, state: EnvState, params: EnvParams,
          max_steps: int = inf, render: bool = RENDER):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    # height, width = env.height, env.width
    # state = env.get_state()
    key = jax.random.PRNGKey(0)
    frontier = [(state, [], 0)]
    visited = {}
    # visited = {env.player_pos: state}
    # visited = {type(env).hashable(state): state}
    # visited_0 = [state]
    best_state_actions = None
    best_reward = -inf
    n_iter = 0
    n_iter_best = 0

    if isinstance(env.action_space, gym.spaces.Discrete):
        possible_actions = list(range(env.action_space.n))
    else:
        raise NotImplementedError

    while len(frontier) > 0:
        n_iter += 1
        if n_iter > max_steps:
            break
        # parent_state, parent_action_seq, parent_rew = frontier.pop(0)
        # Find the idx of the best state in the frontier
        best_idx = np.argmax([f[2] for f in frontier])
        parent_state, parent_action_seq, parent_rew = frontier.pop(best_idx)
        
        # FIXME: Redundant, remove me
        # env.set_state(parent_state)

        # visited[env.player_pos] = env.get_state()
        # if type(env).hashable(parent_state) in visited:
            # continue
        visited[hash(env, parent_state)] = parent_rew
        # print(visited.keys())
        random.shuffle(possible_actions)
        for action in possible_actions:
            # env.set_state(parent_state)
            # print('set frontier state')
            state, obs, rew, done, info = \
                env.step(key=key, action=action, state=parent_state, params=params)
            child_rew = state.ep_rew
            if render:
                env.render()
            # print(f'action: {action}')

            # FIXME: Redundant, remove me

            # map_arr = state['map_arr']
            action_seq = parent_action_seq + [action]
            # if env.player_pos in visited:
            hashed_state = hash(env, state)
            # if hashed_state in visited and child_rew > visited[hashed_state]:
            #     breakpoint()
            if hashed_state in visited and child_rew <= visited[hashed_state]:
                # print(f'already visited {hashed_state}')
                continue
            # visited[env.player_pos] = state
            # visited[tuple(action_seq)] = state
            visited[hashed_state] = child_rew
            # print(len(visited))
            # visited_0.append(state)
            # print([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()])
            # if not np.all([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()]):
                # TT()
            if child_rew > best_reward:
                best_state_actions = (state, action_seq)
                best_reward = child_rew
                n_iter_best = n_iter
                print(f'found new best: {best_reward} at {n_iter_best} iterations step {state.n_step} action sequence length {len(action_seq)}')
            if not jnp.all(done):
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
