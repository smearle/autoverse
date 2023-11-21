from timeit import default_timer as timer

import hydra
import jax

from gen_env.utils import init_base_env, validate_config
from gen_env.configs.config import Config
from search_agent import solve

n_episodes = 10
n_steps = 100

# def main(exp_id='0', overwrite=False, load=False, multi_proc=False, render=False):
@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def profile(cfg: Config):
    validate_config(cfg)
    env, params = init_base_env(cfg)
    start_time = timer()
    total_reset_time = 0
    key = jax.random.PRNGKey(0)
    for i in range(n_episodes):
        reset_start = timer()
        key = jax.random.split(key)[0]
        state, obs = env.reset(key=key, params=params)
        total_reset_time = timer() - reset_start
        for i in range(n_steps):
            key = jax.random.split(key)[0]
            state, obs, reward, done, info = \
                env.step(key, action=env.action_space.sample(), state=state,
                         params=params)
            if cfg.render:
                env.render(state=state, params=params)
    done_time = timer()
    n_total_steps = n_episodes * n_steps
    print(f"overall FPS: {n_total_steps / (done_time - start_time)}")
    print(f"reset FPS: {n_episodes / total_reset_time}")
    print(f"step FPS: {n_total_steps / (done_time - start_time - total_reset_time)}")

@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def render_sol(cfg: Config):
    key = jax.random.PRNGKey(0)
    env, params = init_base_env(cfg)
    best_state_actions = None
    best_reward = 0
    # while best_reward == 0:
    env, params = init_base_env(cfg)
    state, obs = env.reset(key=key, params=params)
    best_state_actions, best_reward, n_iter_best, n_iter = \
        solve(env, state, max_steps=1000, params=params)
    # Render the solution
    frames = []
    if best_state_actions is not None:
        (final_state, action_seq) = best_state_actions
        # env.set_state(state)
        frame = env.render(mode='rgb_array', state=state, params=params)
        frames.append(frame)
        for action in action_seq:
            state, obs, reward, done, info = \
                env.step(key=key, action=action, state=state, params=params)
            frame = env.render(mode='rgb_array', state=state, params=params)
            frames.append(frame)
    
    # Save the solution as a video
    from gen_env.utils import save_video
    save_video(frames, f"sol_{cfg.game}.mp4", fps=10)

if __name__ == '__main__':
    render_sol()
    # profile()