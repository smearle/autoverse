import jax
import numpy as np

from gen_env.envs.play_env import GenEnvParams, PlayEnv
from gen_env.evo.individual import IndividualData
from gen_env.utils import load_game_to_env
from search_agent import batched_bfs, bfs

def evaluate_multi(args):
    return evaluate(*args)

def evaluate_DUMMY(env: PlayEnv, individual: IndividualData, render: bool, trg_n_iter: bool):
    n_players_in_rules = 0
    for rule in individual.rules:
        n_players_in_rules += np.sum(np.vectorize(lambda t: t is not None and t.is_player)(rule._in_out[1])) - \
            np.sum(np.vectorize(lambda t: t is not None and t.is_player)(rule._in_out[0]))
    individual.fitness = n_players_in_rules
    individual.action_seq = []
    return individual

def evaluate(key: jax.random.PRNGKey,
             env: PlayEnv, env_params: GenEnvParams, render: bool, trg_n_iter: bool):
    # load_game_to_env(env, env_params)
    # env.queue_games([env_params.map.copy()], [env_params.rules.copy()])
    # params = get_params_from_individual(env, individual)
    obs, init_state = env.reset(key=key, params=env_params)
    # Save the map after it having been cleaned up by the environment
    # individual.map = env.map.copy()
    # assert individual.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    best_state_actions, best_reward, n_iter_best, n_iter = \
        bfs(env, init_state, env_params, max_steps=trg_n_iter)
    action_seq = None
    if best_state_actions is not None:
        (final_state, action_seq) = best_state_actions
        if render:
            env.render(state=state)
            for action in action_seq:
                state, obs, reward, done, info = env.step(action, state)
                env.render()
    # TODO: dummy
    # fitness = best_reward
    fitness = n_iter_best
    if fitness == 1:
        fitness += n_iter / (trg_n_iter + 2)
    # fitness = len(action_seq) if action_seq is not None else 0
    # env_params.fitness = fitness
    # env_params.action_seq = action_seq
    print(f"Achieved fitness {fitness} at {n_iter_best} iterations with {best_reward} reward. Searched for {n_iter} iterations total.")
    return fitness