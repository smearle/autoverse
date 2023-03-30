
from envs.play_env import PlayEnv
from evo.individual import Individual
from search_agent import solve

def load_game_to_env(env: PlayEnv, individual: Individual):
    env.tiles = individual.tiles
    env._init_rules = individual.rules
    env.init_obs_space()
    return env

def evaluate_multi(args):
    return evaluate(*args)

def evaluate(env: PlayEnv, individual: Individual, render: bool, trg_n_iter: bool):
    load_game_to_env(env, individual)
    env.queue_games([individual.map.copy()], [individual.rules.copy()])
    env.reset()
    init_state = env.get_state()
    # Save the map after it having been cleaned up by the environment
    individual.map = env.map.copy()
    # assert individual.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    best_state_actions, best_reward, n_iter_best, n_iter = solve(env, max_steps=trg_n_iter)
    action_seq = None
    if best_state_actions is not None:
        (final_state, action_seq) = best_state_actions
        if render:
            env.set_state(init_state)
            env.render()
            for action in action_seq:
                env.step(action)
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