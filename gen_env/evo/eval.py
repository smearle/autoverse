import numpy as np

from gen_env.envs.play_env import PlayEnv
from gen_env.evo.individual import Individual
from gen_env.utils import load_game_to_env
from search_agent import solve

def evaluate_multi(args):
    return evaluate(*args)

def evaluate_DUMMY(env: PlayEnv, individual: Individual, render: bool, trg_n_iter: bool):
    n_players_in_rules = 0
    for rule in individual.rules:
        n_players_in_rules += np.sum(np.vectorize(lambda t: t is not None and t.is_player)(rule._in_out[1])) - \
            np.sum(np.vectorize(lambda t: t is not None and t.is_player)(rule._in_out[0]))
    individual.fitness = n_players_in_rules
    individual.action_seq = []
    return individual

def evaluate(env: PlayEnv, individual: Individual, render: bool, trg_n_iter: bool):
    load_game_to_env(env, individual)
    env.queue_games([individual.map.copy()], [individual.rules.copy()])
    init_state, obs = env.reset()
    # Save the map after it having been cleaned up by the environment
    individual.map = env.map.copy()
    # assert individual.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    best_state_actions, best_reward, n_iter_best, n_iter = solve(env, max_steps=trg_n_iter)
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
    individual.fitness = fitness
    individual.action_seq = action_seq
    print(f"Achieved fitness {fitness} at {n_iter_best} iterations with {best_reward} reward. Searched for {n_iter} iterations total.")
    return individual