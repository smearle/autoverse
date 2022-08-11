import copy
from math import inf
from pdb import set_trace as TT
from typing import Iterable

from fire import Fire
import gym
import numpy as np

from games import (hamilton, maze, maze_backtracker, maze_npc, power_line, sokoban)
from gen_env import GenEnv, Rule, TileType


RENDER = False


def solve(env: GenEnv, max_steps: int = inf, render: bool = RENDER):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    # height, width = env.height, env.width
    state = env.get_state()
    frontier = [(state, [])]
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
        parent_state, action_seq = frontier.pop(0)
        env.set_state(parent_state)
        # visited[env.player_pos] = env.get_state()
        # if type(env).hashable(parent_state) in visited:
            # continue
        visited[hash(env, parent_state)] = parent_state
        # print(visited.keys())
        for action in possible_actions:
            env.set_state(parent_state)
            # print('set frontier state')
            obs, rew, done, info = env.step(action)
            if render:
                env.render()
            # print(f'action: {action}')
            state = env.get_state()
            # map_arr = state['map_arr']
            action_seq = action_seq + [action]
            # if env.player_pos in visited:
            hashed_state = hash(env, state)
            if hashed_state in visited:
                # print('already visited', hash(type(env).hashable(state)))
                continue
            # env.render()
            # visited[env.player_pos] = state
            # visited[tuple(action_seq)] = state
            visited[hashed_state] = state
            # print(len(visited))
            # visited_0.append(state)
            # print([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()])
            # if not np.all([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()]):
                # TT()
            if rew > best_reward:
                best_state_actions = (state, action_seq)
                best_reward = rew
                n_iter_best = n_iter
                # print('found new best')
            if not done:
                frontier.append((state, action_seq))
            # print(n_iter)

    return best_state_actions, best_reward, n_iter_best, n_iter


def hash(env: GenEnv, state):
    return env.hashable(state)


def main(game=maze, height=10, width=10, render=False):
    if isinstance(game, str):
        game = globals()[game]

    env: GenEnv = game.make_env(height=height, width=width)
    while True:
        env.reset()
        env.render()
        sol = solve(env, render=render)

if __name__ == "__main__":
    Fire(main)
