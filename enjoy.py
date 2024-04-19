import os

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np

from evo_accel import distribute_evo_envs_to_train, RunnerState
from gen_env.configs.config import EnjoyConfig
# from envs.pcgrl_env import PCGRLEnv, render_stats
from gen_env.envs.play_env import PlayEnv
from gen_env.utils import init_base_env
from rl_player_jax import init_checkpointer
from pcgrl_utils import get_exp_dir, get_network, init_config


@hydra.main(version_base=None, config_path='gen_env/configs/', config_name='enjoy_xlife')
def main_enjoy(config: EnjoyConfig):
    config = init_config(config)

    # Convenienve HACK so that we can render progress without stopping training. Uncomment this or 
    # set JAX_PLATFORM_NAME=cpu in your terminal environment before running this script to run it on cpu.
    # WARNING: Be sure to set it back to gpu before training again!
    # os.system("export JAX_PLATFORM_NAME=cpu")

    exp_dir = get_exp_dir(config)
    if not config.random_agent:
        checkpoint_manager, restored_ckpt = init_checkpointer(config)
        runner_state: RunnerState = restored_ckpt['runner_state']
        network_params = runner_state.train_state.params
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    env: PlayEnv
    env, env_params = init_base_env(config)
    # env.prob.init_graphics()
    network = get_network(env, env_params, config)

    rng = jax.random.PRNGKey(config.seed)
    rng_reset = jax.random.split(rng, config.n_eps)

    # frz_map = jnp.zeros(env.map_shape, dtype=jnp.int8)
    # frz_map = frz_map.at[7, 3:-3].set(1)
    # env.queue_frz_map(frz_map)

    # obs, env_state = env.reset(rng, env_params)

    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(rng_reset, env_params)
    # As above, but explicitly jit

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act = jax.random.split(rng)
        if config.random_agent:
            action = env.action_space(env_params).sample(rng_act)
        else:
            # obs = jax.tree_map(lambda x: x[None, ...], obs)
            action = network.apply(network_params, obs)[
                0].sample(seed=rng_act)
        rng_step = jax.random.split(rng, config.n_eps)
        # obs, env_state, reward, done, info = env.step(
        #     rng_step, env_state, action[..., 0], env_params
        # )
        obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            rng_step, env_state, action, env_params

        )
        # frames = jax.vmap(env.render, in_axes=(0))(env_state, env_params)
        # frame = env.render(env_state)
        rng = jax.random.split(rng)[0]
        # Can't concretize these values inside jitted function (?)
        # So we add the stats on cpu later (below)
        # frame = render_stats(env, env_state, frame)
        return (rng, obs, env_state), (env_state, reward, done, info)

    step_env = jax.jit(step_env, backend='cpu')
    print('Scanning episode steps:')
    _, (states, rewards, dones, infos) = jax.lax.scan(
        step_env, (rng, obs, env_state), None,
        length=1*env.max_episode_steps)

    # Bring things onto the cpu to make rendering faster
    # jax.device_put(states, jax.devices('cpu'))
    jax.tree_map(lambda x: jax.device_put(x, jax.devices('cpu')[0]), states)
    
    print('Rendering gifs:')
    # Since we can't jit our render function (yet)
    frames = []
    for ep_i in range(states.ep_rew.shape[1]):
        ep_frames = []
        for step_i in range(states.ep_rew.shape[0]):
            state_i = jax.tree_map(lambda x: x[step_i, ep_i], states)
            ep_frames.append(env.render(state_i, env_params, mode='rgb_array'))
        frames.append(ep_frames)
        # Print reward
        print(f'Ep {ep_i}: {states.ep_rew[:, ep_i].sum()}')

    frames = np.array(frames)

    # frames = frames.reshape((config.n_eps*env.max_steps, *frames.shape[2:]))

    # assert len(frames) == config.n_eps * env.max_episode_steps, \
    #     "Not enough frames collected"
    assert frames.shape[0] == config.n_eps and frames.shape[1] == env.max_episode_steps, \
        "`frames` has wrong shape"


    # Save gifs.
    for ep_i in range(config.n_eps):
        # ep_frames = frames[ep_is*env.max_steps:(ep_is+1)*env.max_steps]
        ep_frames = frames[ep_i]

        frame_shapes = [frame.shape for frame in ep_frames]
        max_frame_w, max_frame_h = max(frame_shapes, key=lambda x: x[0])[0], \
            max(frame_shapes, key=lambda x: x[1])[1]
        # Pad frames to be same size
        new_ep_frames = []
        for frame in ep_frames:
            frame = np.pad(frame, ((0, max_frame_w - frame.shape[0]),
                                      (0, max_frame_h - frame.shape[1]),
                                      (0, 0)), constant_values=0)
            # frame[:, :, 3] = 255
            new_ep_frames.append(frame)
        ep_frames = new_ep_frames

        gif_name = f"{exp_dir}/anim_ep-{ep_i}" + \
            f"{('_randAgent' if config.random_agent else '')}.gif"
        imageio.v3.imwrite(
            gif_name,
            ep_frames,
            duration=config.gif_frame_duration
        )

        vid_name = f"{exp_dir}/anim_ep-{ep_i}" + \
            f"{('_randAgent' if config.random_agent else '')}.mp4"

        imageio.mimwrite(vid_name, frames[0], fps=25, quality=8, macro_block_size=1)



if __name__ == '__main__':
    main_enjoy()
