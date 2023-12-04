import copy
import dataclasses
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
    reward: float
    done: bool


def solve(env: PlayEnv, state: EnvState, params: EnvParams,
          max_steps: int = inf, render: bool = RENDER, max_episode_steps: int = 100):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""

    def apply_action(key, state, action, params, action_seq, rew):
        state, obs, s_rew, done, info = env.step_env(key=key, action=action, state=state, params=params)
        action_seq = action_seq.at[state.n_step].set(action)
        rew += s_rew
        return SearchNode(state, action_seq, rew, done)

    def expand_frontier_node(state, action_seq, rew):
        step_key = jax.random.split(key, possible_actions.shape[0])
        nodes = \
            jax.vmap(apply_action, in_axes=(None, None, 0, None, None, None))(
                step_key,
                state,
                possible_actions,
                params,
                action_seq,
                rew,
            )
        return nodes

    key = jax.random.PRNGKey(0)
    action_seq = jnp.empty(max_episode_steps, dtype=jnp.int32)
    queued_frontier = []
    frontier = SearchNode(state, action_seq, 0, False)
    frontier_size = 1
    # Add a batch dimension to the frontier
    frontier = jax.tree_map(lambda x: jnp.expand_dims(x, 0), frontier)

    visited = {}
    best_state_actions = None
    best_reward = -inf
    n_iter = 0
    n_iter_best = 0

    possible_actions = jnp.array(list(range(env.action_space.n)), dtype=jnp.int32)
    n_actions = possible_actions.shape[0]

    while n_iter == 0 or frontier.reward.shape[0] > 0:

        # if n_iter > 0:
        #     best_n_idxs = jnp.argpartition(frontier.reward, -min(frontier.reward.shape[0], FRONTIER_VMAP_SIZE))

        #     # Pad this out with the first idx to be size of FRONTIER_VMAP_SIZE
        #     best_n_idxs = jnp.pad(best_n_idxs, (0, FRONTIER_VMAP_SIZE - len(best_n_idxs)), mode='constant', constant_values=best_n_idxs[0])

        #     frontier = jax.tree_map(lambda x: x[best_n_idxs], frontier)

        frontier = jax.vmap(expand_frontier_node, in_axes=(0, 0, 0))(frontier.state, frontier.action_seq, frontier.reward)

        # Now flatten the frontier/possible-actions dimensions so the first dimension corresponds to newly generated states
        frontier = jax.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), frontier)

        # Filter out any states that are already in the visited dict
        idxs_to_keep = []
        for i in range(frontier.reward.shape[0]):
            # Can't have expanded more states than this
            if i == frontier_size * n_actions:
                break
            state = jax.tree_map(lambda x: x[i], frontier.state)
            hashed_state = hash(env, state)
            if hashed_state not in visited or frontier.reward[i] > visited[hashed_state]:
                if not frontier.done[i]:
                    idxs_to_keep.append(i)
                visited[hashed_state] = frontier.reward[i]
                if frontier.reward[i] > best_reward:
                    best_state_actions = (state, frontier.action_seq[i])
                    best_reward = frontier.reward[i]
                    n_iter_best = n_iter

        n_iter += 1
        if n_iter > max_steps:
            break

        # Remove all things from the frontier that we've already visited
        frontier = jax.tree_map(lambda x: x[jnp.array(idxs_to_keep, dtype=jnp.int32)], frontier) 

        jax.debug.print('frontier size {frontier_size}', frontier_size=frontier.reward.shape)

        # More efficient to do this in a single vmap
        # hashed_states = jax.vmap(hash, in_axes=(None, 0))(env, frontier.state)

        # TODO: Put extra frontier states on a queue
        # TODO: Pad out the frontier if we have too few new states, or bump things from the queue

        # for i, action in enumerate(possible_actions):
        #     # state, obs, rew, done, info = \
        #     #     env.step_env(key=key, action=action, state=parent_state, params=params)
        #     child_rew = v_state.ep_rew[i]
        #     if render:
        #         env.render()
        #     action_seq = parent_action_seq + [action]
        #     state = jax.tree_map(lambda x: x[i], v_state)
        #     hashed_state = hash(env, state)
        #     if hashed_state in visited and child_rew <= visited[hashed_state]:
        #         continue
        #     visited[hashed_state] = child_rew
        #     if child_rew > best_reward:
        #         best_state_actions = (state, action_seq)
        #         best_reward = child_rew
        #         n_iter_best = n_iter
        #         # print(f'found new best: {best_reward} at {n_iter_best} iterations step {state.n_step} action sequence length {len(action_seq)}')
        #     if not jnp.all(v_done[i]):
        #         # Add this state to the frontier so can we can continue searching from it later
        #         frontier.append((state, action_seq, child_rew))

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
