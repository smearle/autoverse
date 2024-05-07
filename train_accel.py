from functools import partial
import os
import shutil
from timeit import default_timer as timer
from typing import NamedTuple, Tuple

from gen_env.envs.play_env import GenEnvParams, GenEnvState, PlayEnv
import hydra
import jax
import logging

import numpy as np

from purejaxrl.wrappers import LogEnvState
from utils import init_il_config, init_rl_config, load_elite_envs
logging.getLogger('jax').setLevel(logging.INFO)
import jax.numpy as jnp
from flax import struct
import imageio
import optax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax
from tensorboardX import SummaryWriter

from gen_env.evo.individual import Individual
from evo_accel import EvoState, apply_evo, distribute_evo_envs_to_train, gen_discount_factors_matrix
from gen_env.configs.config import (RLConfig, TrainConfig, TrainAccelConfig, 
                                    GenEnvConfig)
from gen_env.utils import gen_rand_env_params, init_base_env, init_config
from purejaxrl.experimental.s5.wrappers import LogWrapper
from pcgrl_utils import get_rl_ckpt_dir, get_network


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: GenEnvState
    evo_state: EvoState
    last_obs: jnp.ndarray
    train_env_params: GenEnvParams
    val_env_params: GenEnvParams
    # rng_act: jnp.ndarray
#   ep_returns: jnp.ndarray
    rng: jnp.ndarray
    update_i: int


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # rng_act: jnp.ndarray


def log_callback(metric, steps_prev_complete, cfg: RLConfig, writer, train_start_time):
    timesteps = metric["timestep"][metric["returned_episode"]
                                    ] * cfg.n_envs
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]

    if len(timesteps) > 0:
        t = timesteps[0]
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()
        print(f"global step={t}; episodic return mean: {ep_return_mean} " + \
            f"max: {ep_return_max}, min: {ep_return_min}")
        ep_length = (metric["returned_episode_lengths"]
                        [metric["returned_episode"]].mean())

        # Add a row to csv with ep_return
        with open(os.path.join(cfg._log_dir_rl,
                                "progress.csv"), "a") as f:
            f.write(f"{t},{ep_return_mean}\n")

        writer.add_scalar("rl/ep_return", ep_return_mean, t)
        writer.add_scalar("rl/ep_return_max", ep_return_max, t)
        writer.add_scalar("rl/ep_return_min", ep_return_min, t)
        writer.add_scalar("rl/ep_length", ep_length, t)
        fps = (t - steps_prev_complete) / (timer() - train_start_time)
        writer.add_scalar("rl/fps", fps, t)

        print(f"fps: {fps}")


def eval(rng: jax.random.PRNGKey, cfg: TrainConfig, env: PlayEnv, env_params: GenEnvParams, network, network_params,
         update_i: int, writer, n_eps: int = 1):

    def step_env(carry, _):
        env_state: LogEnvState
        env_params: GenEnvParams
        rng, obs, env_state, env_params, network_params = carry
        rng, _rng = jax.random.split(rng)

        rand_idxs = jax.random.choice(rng, env_params.rule_dones.shape[0], (cfg.n_envs,), replace=True)
        next_params = jax.tree.map(lambda x: x[rand_idxs], env_params)

        pi, value = network.apply(network_params, obs)
        action_r = pi.sample(seed=rng)
        rng_step = jax.random.split(_rng, cfg.n_envs)

        # rng_step_r = rng_step_r.reshape((config.n_gpus, -1) + rng_step_r.shape[1:])
        vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))
        # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
        obs, env_state, reward_r, done_r, info_r, env_params = vmap_step_fn(
                        rng_step, env_state, action_r,
                        env_params, next_params)
        
        return (rng, obs, env_state, env_params, network_params),\
            (env_state, reward_r, done_r, info_r, value)

    eval_rng = jax.random.split(rng, cfg.n_envs)

    rand_idxs = jax.random.choice(rng, env_params.rule_dones.shape[0], (cfg.n_envs,), replace=True)
    curr_env_params = jax.tree.map(lambda x: x[rand_idxs], env_params)

    obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
            eval_rng, curr_env_params
    )

    _, (states, rewards, dones, infos, values) = jax.lax.scan(
        step_env, (rng, obsv, env_state, env_params, network_params),
        None, n_eps*env.max_episode_steps)

    returns = rewards.sum(axis=0)

    _eval_log_callback = partial(eval_log_callback, writer=writer)

    jax.experimental.io_callback(_eval_log_callback, None, metric={
        'mean_return': returns.mean(),
        'max_return': returns.max(),
        'min_return': returns.min(),
    }, t=update_i)


def eval_log_callback(metric, writer, t):
    writer.add_scalar("rl/eval/ep_return", metric['mean_return'], t)
    writer.add_scalar("rl/eval/ep_return_max", metric['max_return'], t)
    writer.add_scalar("rl/eval/ep_return_min", metric['min_return'], t)


def make_train(cfg: TrainAccelConfig, restored_ckpt, checkpoint_manager, train_env_params: GenEnvParams,
               val_env_params: GenEnvParams):
    cfg.NUM_UPDATES = (
        cfg.total_timesteps // cfg.num_steps // cfg.n_envs
    )
    cfg.MINIBATCH_SIZE = (
        cfg.n_envs * cfg.num_steps // cfg.NUM_MINIBATCHES
    )
    env_r, base_env_params = init_base_env(cfg)
    # env_r, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env_r)
    # env_r.init_graphics()

    evo_individual = Individual(cfg, env.tiles)
    discount_factors_matrix = gen_discount_factors_matrix(cfg.GAMMA, max_episode_steps=env.max_episode_steps)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (cfg.NUM_MINIBATCHES * cfg.update_epochs))
            / cfg.NUM_UPDATES
        )
        return cfg["LR"] * frac

    def train(rng, cfg: TrainConfig, train_env_params: GenEnvParams, val_env_params: GenEnvParams):
        train_start_time = timer()

        # INIT NETWORK
        network = get_network(env, base_env_params, cfg)

        # Create a tensorboard writer
        writer = SummaryWriter(cfg._log_dir_rl)

        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(base_env_params)
        # init_x = env.observation_space(env_params).sample(_rng)[None]
        network_params = network.init(_rng, init_x)
        print(network.subnet.tabulate(_rng, init_x.map, init_x.flat))
        # print(network.subnet.tabulate(_rng, init_x, jnp.zeros((init_x.shape[0], 0))))

        # Print number of learnable parameters in the network
        if cfg.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
                optax.adam(cfg.lr, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV FOR TRAIN
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, cfg.n_envs)
        # Apply pmap
        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, 0))
        # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))

        # Sample n_envs many random indices between in the range of train_env_params.shape[0] without replacement
        rand_idxs = jax.random.choice(_rng, train_env_params.rule_dones.shape[0], (cfg.n_envs,), replace=True)
        curr_env_params = jax.tree.map(lambda x: x[rand_idxs], train_env_params)
        obsv, env_state = vmap_reset_fn(reset_rng, curr_env_params)

        # INIT ENV FOR RENDER
        # rng_r, _rng_r = jax.random.split(rng)
        # reset_rng_r = jax.random.split(_rng_r, cfg.n_render_eps)
        # render_env_params = jax.tree.map(lambda x: x[:cfg.n_render_eps], train_env_params)

        # Apply pmap
        # reset_rng_r = reset_rng_r.reshape((config.n_gpus, -1) + reset_rng_r.shape[1:])
        vmap_reset_fn = jax.vmap(env_r.reset, in_axes=(0, 0))
        # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
        # obsv_r, env_state_r = vmap_reset_fn(reset_rng_r, render_env_params)  # Replace None with your env_params if any
        
        # obsv_r, env_state_r = jax.vmap(
        #     env_r.reset, in_axes=(0, None))(reset_rng_r, env_params)

        rng, _rng = jax.random.split(rng)
#       ep_returns = jnp.full(shape=config.NUM_UPDATES,
#       ep_returns = jnp.full(shape=1,
#                             fill_value=jnp.nan, dtype=jnp.float32)

        # Note that there may be duplicates in this first generation due to sampling with replacement above.
        evo_env_params = curr_env_params

        evo_state = EvoState(env_params=evo_env_params, top_fitness=jnp.zeros(cfg.n_envs))

        # train_env_params = distribute_evo_envs_to_train(cfg, evo_env_params)

        if cfg.evo_freq != -1:
            # To deal with mismatched shapes after calling _update_step
            train_env_params = evo_state.env_params

        steps_prev_complete = 0
        runner_state = RunnerState(
            train_state=train_state, env_state=env_state, last_obs=obsv, rng=rng,
            train_env_params=train_env_params,
            val_env_params=val_env_params,
            evo_state=evo_state,
            update_i=0,
            )

        # exp_dir = get_exp_dir(config)
        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt['steps_prev_complete']
            runner_state = restored_ckpt['runner_state']
            steps_remaining = cfg.total_timesteps - steps_prev_complete
            cfg.NUM_UPDATES = int(
                steps_remaining // cfg.num_steps // cfg.n_envs)

            # TODO: Overwrite certain config values

        _log_callback = partial(log_callback, cfg=cfg, writer=writer,
                               train_start_time=train_start_time,
                               steps_prev_complete=steps_prev_complete)


        # def render_frames_np(i, network_params):
        #     frames, states = render_episodes_np(network_params)
        #     return render_frames_gif(frames, i, states)

        # def render_frames_gif(frames, i, env_states=None):
        #     if i % cfg.render_freq != 0:
        #     # if jnp.all(frames == 0):
        #         return
        #     print(f"Rendering episode gifs at update {i}")
        #     assert len(frames) == cfg.n_render_eps * 1 * env.max_episode_steps,\
        #         "Not enough frames collected"

        #     if cfg.env_name == 'Candy':
        #         # Render intermediary frames.
        #         pass

        #     # Save gifs.
        #     for ep_is in range(cfg.n_render_eps):
        #         gif_name = f"{cfg.exp_dir}/update-{i}_ep-{ep_is}.gif"
        #         ep_frames = frames[ep_is*env.max_episode_steps:(ep_is+1)*env.max_episode_steps]

        #         # new_frames = []
        #         # for i, frame in enumerate(frames):
        #         #     state_i = jax.tree_map(lambda x: x[i], env_states)
        #         #     frame = render_stats(env_r, state_i, frame)
        #         #     new_frames.append(frame)
        #         # frames = new_frames

        #         try:
        #             imageio.v3.imwrite(
        #                 gif_name,
        #                 ep_frames,
        #                 duration=cfg.gif_frame_duration
        #             )
        #         except jax.errors.TracerArrayConversionError:
        #             print("Failed to save gif. Skipping...")
        #             return
        #     print(f"Done rendering episode gifs at update {i}")

        # # def render_episodes(network_params):
        # #     _, (states, rewards, dones, infos, frames) = jax.lax.scan(
        # #         step_env_render, (rng_r, obsv_r, env_state_r, network_params),
        # #         None, 1*env.max_episode_steps)

        # def render_episodes_np(network_params):
        #     state_r = None
        #     key = jax.random.PRNGKey(0)
        #     frames = []
        #     for ep_idx in range(cfg.n_render_eps):
        #         env_params = jax.tree.map(lambda x: x[ep_idx], train_env_params)
        #         obs_r, state_r = env_r.reset_env(key, env_params)
        #         for _ in range(env.max_episode_steps):
        #             obs_r = jax.tree.map(lambda x: x[None], obs_r)
        #             pi, value = network.apply(network_params, obs_r)
        #             action = pi.sample(seed=key)
        #             key = jax.random.split(key)[0]
        #             obs_r, state_r, reward, done, info = env_r.step_env(key, state_r, action[0], env_params)
        #             # Concatenate the gpu dimension
        #             frame = env_r.render(state_r, env_params, mode='rgb_array')
        #             frames.append(frame)


        #         states = state_r

        #     frames = jnp.stack(frames, 0)
        #     return frames, states

        # def step_env_render(carry, _):
        #     rng_r, env_state_r, obs_r, network_params = carry
        #     rng_r, _rng_r = jax.random.split(rng_r)

        #     pi, value = network.apply(network_params, obs_r)
        #     action_r = pi.sample(seed=rng_r)

        #     rng_step = jax.random.split(_rng_r, cfg.n_render_eps)

        #     # rng_step_r = rng_step_r.reshape((config.n_gpus, -1) + rng_step_r.shape[1:])
        #     vmap_step_fn = jax.vmap(env_r.step, in_axes=(0, 0, 0, None))
        #     # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
        #     obs_r, env_state_r, reward_r, done_r, info_r = vmap_step_fn(
        #                     rng_step, env_state_r, action_r,
        #                     train_env_params)
        #     vmap_render_fn = jax.vmap(env_r.render, in_axes=(0, None))
        #     # pmap_render_fn = jax.pmap(vmap_render_fn, in_axes=(0,))
        #     frames = vmap_render_fn(env_state_r, train_env_params)
        #     # Get rid of the gpu dimension
        #     # frames = jnp.concatenate(jnp.stack(frames, 1))
        #     return (rng_r, obs_r, env_state_r, network_params),\
        #         (env_state_r, reward_r, done_r, info_r, frames)

        def save_checkpoint(runner_state, info, steps_prev_complete):
            try:
                timesteps = info["timestep"][info["returned_episode"]
                                             ] * cfg.n_envs
            except jax.errors.NonConcreteBooleanIndexError:
                return
            for t in timesteps:
                if t > 0:
                    latest_ckpt_step = checkpoint_manager.latest_step()
                    if (latest_ckpt_step is None or
                            t - latest_ckpt_step >= cfg.ckpt_freq):
                        print(f"Saving checkpoint at step {t}")
                        ckpt = {'runner_state': runner_state,
                                'config': cfg, 
                                'step_i': t}
                        # ckpt = {'step_i': t}
                        save_args = orbax_utils.save_args_from_target(ckpt)
                        checkpoint_manager.save(t, ckpt, save_kwargs={
                                                'save_args': save_args})
                    break

        # frames, states_r = render_episodes(train_state.params)
        # jax.debug.callback(render_frames, frames, runner_state.update_i, states_r)
        # jax.debug.print("Rendering episode gifs at update 0")
        # jax.debug.callback(render_frames_np, runner_state.update_i,
        #                    train_state.params)
        # jax.debug.print(f"Done rendering episode gifs at update 0")
        # old_render_results = (frames, states)

        # jax.debug.print(f'Rendering episode gifs took {timer() - start_time} seconds')

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(carry: Tuple[RunnerState, GenEnvParams], unused):
                runner_state, next_env_params = carry
                train_state, env_state, evo_state, last_obs, rng, update_i = (
                    runner_state.train_state, runner_state.env_state,
                    runner_state.evo_state,
                    runner_state.last_obs,
                    runner_state.rng, runner_state.update_i,
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # Squash the gpu dimension (network only takes one batch dimension)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg.n_envs)

                # Note that we are mapping across environment parameters as well to train on different environments
                vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))
                # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
                obsv, env_state, reward, done, info = vmap_step_fn(
                    rng_step, env_state, action, env_state.env_state.params, next_env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = runner_state.replace(
                    train_state=train_state, env_state=env_state, last_obs=obsv, rng=rng,
                    update_i=update_i, evo_state=evo_state)
                return (runner_state, next_env_params), transition

            # TODO: Move this inside env_step (in case of long rollouts) and make sure no slowdown results.
            # Randomly sample new environments from the training set to be used in case of reset during rollout.
            rand_env_idxs = jax.random.choice(runner_state.rng, runner_state.train_env_params.rule_dones.shape[0], 
                                              (cfg.n_envs,), replace=True)
            next_env_params = jax.tree.map(lambda x: x[rand_env_idxs], runner_state.train_env_params)
            # next_env_params = runner_state.train_env_params

            runner_state_next_env_params, traj_batch = jax.lax.scan(
                _env_step, (runner_state, next_env_params), None, cfg.num_steps
            )
            runner_state, next_env_params = runner_state_next_env_params

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, evo_state = \
                runner_state.train_state, \
                runner_state.env_state, \
                runner_state.last_obs, runner_state.rng, runner_state.evo_state
            
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + cfg.GAMMA * \
                        next_value * (1 - done) - value
                    gae = (
                        delta
                        + cfg.GAMMA * cfg.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        # obs = traj_batch.obs[None]
                        pi, value = network.apply(params, traj_batch.obs)
                        # action = traj_batch.action.reshape(pi.logits.shape[:-1])
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-cfg.CLIP_EPS, cfg.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses,
                                              value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - cfg.CLIP_EPS,
                                1.0 + cfg.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + cfg.VF_COEF * value_loss
                            - cfg.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    # jax.debug.print("total_loss={total_loss}", total_loss=total_loss)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = \
                    update_state
                rng, _rng = jax.random.split(rng)
                batch_size = cfg.MINIBATCH_SIZE * cfg.NUM_MINIBATCHES
                assert (
                    batch_size == cfg.num_steps * cfg.n_envs
                ), "batch size must be equal to number of steps * number " + \
                    "of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [cfg.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch,
                                advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, cfg.update_epochs
            )
            train_state = update_state[0]
            
            # take the elite params, mutate some new offspring, evaluate 
            # everything, and keep the best
            # NOTE: If you vmap the train function, both of these branches will (most probably)
            # be evaluated each time.                  
            if cfg.evo_freq != -1:
                do_evo = runner_state.update_i % cfg.evo_freq == 0
                evo_state: EvoState = jax.lax.cond(
                    do_evo,
                    lambda: apply_evo(
                        rng=rng, env=env, ind=evo_individual, evo_state=evo_state, 
                        network_params=network_params, network=network,
                        cfg=cfg,
                        discount_factor_matrix=discount_factors_matrix),
                    lambda: evo_state)

                next_train_env_params = evo_state.env_params
                # Tile and slice env_params so that we have `n_envs` many
                # n_reps = int(np.ceil(cfg.n_envs / cfg.evo_pop_size))
                # train_env_params = distribute_evo_envs_to_train(cfg, elite_env_params)
                # Now when we step the environments, each will reset to `queued` 
                # env_params on next reset (inside PlayEnv.step())
                # env_state = env_state.replace(env_state = env_state.env_state.replace(queued_params=train_env_params))
            else:
                # If we're not evolving the envs, just keep the same training set.
                next_train_env_params = train_env_params

            # FIXME: shouldn't assume size of render map.
            # frames_shape = (cfg.n_render_eps * 1 * env.max_episode_steps, 
            #                 env.tile_size * (env.map_shape[0] + 2),
            #                 env.tile_size * (env.map_shape[1] + 2), 4)

            # FIXME: Inside vmap, both conditions are likely to get executed. Any way around this?
            # Currently not vmapping the train loop though, so it's ok.
            # start_time = timer()
            # frames, states = jax.lax.cond(
            #     runner_state.update_i % config.render_freq == 0,
            #     lambda: render_episodes(train_state.params),
            #     lambda: old_render_results,)
            # jax.debug.callback(render_frames, frames, runner_state.update_i, states)
            # if runner_state.update_i % config.render_freq == 0:
            # jax.debug.print("Rendering episode gifs at update 0")
            # jax.debug.callback(render_frames_np, runner_state.update_i,
            #                 train_state.params)
            # jax.debug.print(f"Done rendering episode gifs at update 0")
            # old_render_results = (frames, states)
            # jax.debug.print(f'Rendering episode gifs took {timer() - start_time} seconds')

            if cfg.val_freq != -1:
                do_eval = runner_state.update_i % cfg.val_freq == 0
                jax.lax.cond(do_eval,
                             lambda: eval(rng, cfg, env, val_env_params, network, network_params, runner_state.update_i,
                                          writer),
                             lambda: None)

            metric = traj_batch.info
            rng = update_state[-1]

            jax.debug.callback(_log_callback, metric)
            jax.debug.callback(save_checkpoint, runner_state,
                               metric, steps_prev_complete)


            runner_state = runner_state.replace(
                train_state=train_state, env_state=env_state, last_obs=last_obs, rng=rng,
                update_i=runner_state.update_i+1, evo_state=evo_state, train_env_params=next_train_env_params)

            return runner_state, metric

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, cfg.NUM_UPDATES
        )

        jax.debug.callback(save_checkpoint, runner_state,
                           metric, steps_prev_complete)

        return {"runner_state": runner_state, "metrics": metric}

    return lambda rng: train(rng, cfg, train_env_params, val_env_params)


# def plot_ep_returns(ep_returns, config):
#     plt.plot(ep_returns)
#     plt.xlabel("Timesteps")
#     plt.ylabel("Episodic Return")
#     plt.title(f"Episodic Return vs. Timesteps ({config.ENV_NAME})")
#     plt.savefig(os.path.join(get_exp_dir(config), "ep_returns.png"))


def init_checkpointer(config: RLConfig, train_env_params: GenEnvParams, val_env_params: GenEnvParams):
    # This will not affect training, just for initializing dummy env etc. to load checkpoint.
    rng = jax.random.PRNGKey(30)
    # Set up checkpointing
    ckpt_dir = get_rl_ckpt_dir(config)
    # Get absolute path
    ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)

    # Create a dummy checkpoint so we can restore it to the correct dataclasses
    # env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env, env_params = init_base_env(config)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    rng, _rng = jax.random.split(rng)
    network = get_network(env, env_params, config)
    init_x = env.gen_dummy_obs(env_params)
    # init_x = env.observation_space(env_params).sample(_rng)[None, ]
    network_params = network.init(_rng, init_x)
    tx = optax.chain(
        optax.clip_by_global_norm(config.MAX_GRAD_NORM),
        optax.adam(config.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)

    # reset_rng_r = reset_rng.reshape((config.n_gpus, -1) + reset_rng.shape[1:])
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None))
    # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
    obsv, env_state = vmap_reset_fn(
        reset_rng, 
        env_params, 
    )
    n_train_envs = train_env_params.rule_dones.shape[0]
    # evo_env_params = jax.tree.map(lambda x: jnp.array([x for _ in range(n_train_envs)]), env_params)
    evo_state = EvoState(env_params=train_env_params, top_fitness=jnp.full(config.evo_pop_size, -jnp.inf))
    runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv,
                               train_env_params=train_env_params, val_env_params=val_env_params,
                               # ep_returns=jnp.full(config.num_envs, jnp.nan), 
                               rng=rng, update_i=0, evo_state=evo_state)
    target = {'runner_state': runner_state, 'config': config, 'step_i': 0}
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), options)

    if checkpoint_manager.latest_step() is None:
        restored_ckpt = None
    else:
        steps_prev_complete = checkpoint_manager.latest_step()
        restored_ckpt = checkpoint_manager.restore(
            steps_prev_complete, items=target)
        restored_ckpt['steps_prev_complete'] = steps_prev_complete

        # # Load the csv as a dataframe and delete all rows after the last checkpoint
        # progress_csv_path = os.path.join(get_exp_dir(config), "progress.csv")
        # progress_df = pd.read_csv(progress_csv_path, names=["timestep", "ep_return"])
        # # Convert timestep to int

        # progress_df = progress_df[progress_df["timestep"] <= steps_prev_complete]
        # progress_df.to_csv(progress_csv_path, header=False, index=False)

    return checkpoint_manager, restored_ckpt

    

@hydra.main(version_base=None, config_path='gen_env/configs', config_name='train_accel')
def main(cfg: RLConfig):
    init_config(cfg)
    latest_gen = init_il_config(cfg)
    init_rl_config(cfg, latest_gen)

    rng = jax.random.PRNGKey(cfg.seed)

    train_elites, val_elites, test_elites = load_elite_envs(cfg, latest_gen)
    # train_env_params = jax.tree.map(lambda x: x[:cfg.n_envs], train_elites.env_params)
    val_env_params = val_elites.env_params
    if cfg.blank_env_start:
        env, env_params = init_base_env(cfg)
        train_env_params = jax.vmap(gen_rand_env_params, in_axes=(None, 0, None, None))(
            cfg, jax.random.split(rng, cfg.n_envs), env.game_def, env_params.rules)
    else:
        train_env_params = train_elites.env_params

    # Get 20 random indices for train envs
    # idxs = jax.random.permutation(jax.random.PRNGKey(cfg.seed), jnp.arange(train_env_params.rule_dones.shape[0]))[:cfg.n_envs]
    # train_env_params = jax.tree.map(lambda x: x[idxs], train_env_params)

    # Take the first 20 params
    # train_env_params = jax.tree_map(lambda x: x[:cfg.n_envs], train_env_params)

    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if cfg.overwrite and os.path.exists(cfg._log_dir_rl):
        shutil.rmtree(cfg._log_dir_rl)

    checkpoint_manager, restored_ckpt = init_checkpointer(cfg, train_env_params=train_env_params,
                                                          val_env_params=val_env_params)

    # if restored_ckpt is not None:
    #     ep_returns = restored_ckpt['runner_state'].ep_returns
    #     plot_ep_returns(ep_returns, config)
    # else:
    if restored_ckpt is None:
        progress_csv_path = os.path.join(cfg._log_dir_rl, "progress.csv")
        assert not os.path.exists(progress_csv_path), "Progress csv already exists, but have no checkpoint to restore " +\
            "from. Run with `overwrite=True` to delete the progress csv."
        # Create csv for logging progress
        with open(os.path.join(cfg._log_dir_rl, "progress.csv"), "w") as f:
            f.write("timestep,ep_return\n")

    train_jit = jax.jit(make_train(cfg, restored_ckpt, checkpoint_manager, train_env_params, val_env_params))
    out = train_jit(rng)

#   ep_returns = out["runner_state"].ep_returns


if __name__ == "__main__":
    main()
