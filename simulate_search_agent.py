from math import inf
from pdb import set_trace as TT

from fire import Fire
import gym
import numpy as np
import pygame

from games import (hamilton, maze, maze_pcg, maze_npc, power_line, sokoban)
from gen_env import GenEnv, Rule, colors, TileType


def solve(env: GenEnv):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    # height, width = env.height, env.width
    state = env.get_state()
    frontier = [(state, [])]
    visited = {env.player_pos: state}
    # visited = {type(env).hashable(state): state}
    # visited_0 = [state]
    best_state_actions = None
    best_reward = -inf

    if isinstance(env.action_space, gym.spaces.Discrete):
        possible_actions = list(range(env.action_space.n))
    else:
        raise NotImplementedError

    while len(frontier) > 0:
        parent_state, action_seq = frontier.pop(0)
        # visited[type(env).hashable(state)] = parent_state
        # print(visited.keys())
        for action in possible_actions:
            env.set_state(parent_state)
            print('set frontier state')
            env.render()
            obs, rew, done, info = env.step(action)
            print(f'action: {action}')
            env.render()
            state = env.get_state()
            map_arr = state['map_arr']
            action_seq = action_seq + [action]
            if env.player_pos in visited:
            # if type(env).hashable(state) in visited:
            # if np.any([np.all(state['map_arr'] == s['map_arr']) for s in visited_0]):
                continue
            visited[env.player_pos] = state
            # visited[tuple(action_seq)] = state
            # visited.add(type(env).hashable(state))
            # visited_0.append(state)
            print(len(visited))
            # print([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()])
            # if not np.all([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()]):
                # TT()
            if rew > best_reward:
                best_state_actions = (state, action_seq)
                best_reward = rew
                print('found new best')
            if not done:
                frontier.append((state, action_seq))

    return best_state_actions, best_reward


def main(game=maze):
    if isinstance(game, str):
        game = globals()[game]

    env: GenEnv = game.make_env()
    while True:
        env.reset()
        env.render()
        sol = solve(env)

if __name__ == "__main__":
    Fire(main)
