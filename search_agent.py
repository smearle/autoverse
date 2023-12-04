import copy
import dataclasses
from functools import partial
from math import inf
from pdb import set_trace as TT
import random
from typing import Iterable

from fire import Fire
from flax import struct
import gym
import jax
from jax import numpy as jnp
import numpy as np

from gen_env.games import (hamilton, maze, maze_backtracker, maze_npc, power_line, sokoban)
from gen_env.envs.play_env import EnvParams, EnvState, PlayEnv, Rule, TileType


# For debugging. Render every environment state that is visited during search.
RENDER = False

FRONTIER_VMAP_SIZE = 1_000

@struct.dataclass
class SearchNode:
    state: EnvState
    action_seq: Iterable[int]
    done: bool


@partial(jax.jit, static_argnums=(0,))
def expand_frontier(env: PlayEnv, next_frontier: SearchNode, params: EnvParams,
                    possible_actions: jnp.ndarray):

    def apply_action(key, state, action, params, action_seq):
        state, obs, s_rew, done, info = env.step_env(key=key, action=action, state=state, params=params)
        action_seq = action_seq.at[state.n_step-1].set(action)
        return SearchNode(state, action_seq, done)

    def expand_frontier_node(key, state, action_seq):
        step_key = jax.random.split(key, possible_actions.shape[0])
        nodes = \
            jax.vmap(apply_action, in_axes=(None, None, 0, None, None))(
                step_key,
                state,
                possible_actions,
                params,
                action_seq,
            )
        return nodes

    frontier = jax.vmap(expand_frontier_node, in_axes=(None, 0, 0))(
        jax.random.PRNGKey(0), next_frontier.state, next_frontier.action_seq)

    # Flatten the frontier/possible-actions dimensions so the first dimension corresponds to newly generated states
    frontier = jax.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), frontier)

    return frontier


def solve(env: PlayEnv, state: EnvState, params: EnvParams,
          max_steps: int = inf, render: bool = RENDER, max_episode_steps: int = 100):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""

    key = jax.random.PRNGKey(0)
    action_seq = jnp.empty(max_episode_steps, dtype=jnp.int32)
    frontier = SearchNode(state, action_seq, 0)
    # Add a batch dimension to the frontier
    frontier = jax.tree_map(lambda x: jnp.expand_dims(x, 0), frontier)

    queued_frontier = frontier # HACK to give it some initial value. We will end up searching this state twice.
                               # How can we initialize it with dimension 0 of size 0?
    # queued_frontier = 

    frontier_size = 1
    next_frontier = frontier

    visited = {}
    best_state_actions = None
    best_reward = -inf
    n_iter = 0
    n_iter_best = 0

    possible_actions = jnp.array(list(range(env.action_space.n)), dtype=jnp.int32)
    n_actions = possible_actions.shape[0]

    while n_iter == 0 or next_frontier.state.ep_rew.shape[0] > 0:

        # Expand the frontier
        # print(f'pre-expand frontier')
        frontier = expand_frontier(env, next_frontier, params, possible_actions)
        # print(f'post-expand frontier')

        # Filter out any states that are already in the visited dict
        idxs_to_keep = []
        idxs_to_delete = []

        for i in range(frontier.done.shape[0]):
            # Can't have expanded more states than this
            if i == frontier_size * n_actions:
                idxs_to_delete += list(range(i, frontier.done.shape[0]))
                break
            state = jax.tree_map(lambda x: x[i], frontier.state)
            hashed_state = hash(env, state)
            # print(f'hashed state {hashed_state}')
            if hashed_state not in visited or frontier.state.ep_rew[i] > visited[hashed_state]:
                if not frontier.done[i]:
                    idxs_to_keep.append(i)
                else:
                    idxs_to_delete.append(i)
                    # print(f'Deleting state because done')
                visited[hashed_state] = frontier.state.ep_rew[i]
                if frontier.state.ep_rew[i] > best_reward:
                    print(f'New best reward {frontier.state.ep_rew[i]}')
                    best_state_actions = (state, frontier.action_seq[i])
                    best_reward = frontier.state.ep_rew[i]
                    n_iter_best = n_iter
            else:
                # print('deleting state because already visited')
                idxs_to_delete.append(i)


        # FIXME:Remove all things from the frontier that we've already visited
        # frontier = jax.tree_map(lambda x: x[jnp.array(idxs_to_keep, dtype=jnp.int32)], frontier) 
        frontier = jax.tree_map(
            lambda x: jnp.delete(x, 
                                 jnp.array(idxs_to_delete, dtype=jnp.int32),
                                 axis=0),
            frontier)

        # Combine with the queued frontier
        total_frontier = jax.tree_map(lambda x, y: jnp.concatenate((x, y)), frontier, queued_frontier)

        # Break if we have nothing left
        if total_frontier.done.shape[0] == 0:
            break

        best_idxs = jnp.argpartition(total_frontier.state.ep_rew, 
                                     -min(total_frontier.done.shape[0], FRONTIER_VMAP_SIZE))
        # Flip the order so the best idxs are first
        best_idxs = best_idxs[::-1]

        frontier_size = min(len(best_idxs), FRONTIER_VMAP_SIZE)

        n_iter += frontier_size
        if n_iter > max_steps:
            break

        # Pad this out with the first idx to be size of FRONTIER_VMAP_SIZE (if 
        # necessary)
        best_idxs = jnp.pad(best_idxs, 
                            (0, max(0, FRONTIER_VMAP_SIZE - frontier_size)),
                            mode='constant', constant_values=best_idxs[0])

        # Exclude idxs beyond FRONTIER_VMAP_SIZE (if necessary)
        best_idxs = best_idxs[:FRONTIER_VMAP_SIZE]

        next_frontier = jax.tree_map(lambda x: x[best_idxs], total_frontier)

        # The new queued_frontier is everything other than the best_idxs
        queued_frontier = jax.tree_map(lambda x: jnp.delete(x, best_idxs, axis=0), total_frontier)
        
        print(f'iter {n_iter}, frontier size {frontier.done.shape} queued frontier size {queued_frontier.done.shape}')


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
