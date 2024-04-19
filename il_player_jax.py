import collections
import functools
import glob
import os
import shutil
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import gymnax
import hydra
import jax
from jax import numpy as jnp
from jax.random import PRNGKey
import numpy as np
import optax
from tensorboardX import SummaryWriter
import tqdm

from gen_env.configs.config import GenEnvConfig, ILConfig
from gen_env.envs.gen_env import GenEnv
from gen_env.utils import init_base_env, validate_config
from pcgrl_utils import get_network
from purejaxrl.experimental.s5.wrappers import LogWrapper



Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        assert (loss_fn is not None or grads is not None,
                'Either a loss function or grads must be specified.')
        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            assert (has_aux,
                    'When grads are provided, expects no aux outputs.')

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


@functools.partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == 'det':
        return rng, actor_apply_fn({'params': actor_params}, observations,
                                   temperature)
    else:
        dist = actor_apply_fn({'params': actor_params}, observations,
                              temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


def sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_apply_fn, actor_params, observations,
                           temperature, distribution)


def log_prob_update(actor: Model, batch: Batch,
                    rng: PRNGKey) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params},
                              batch.observations,
                              training=True,
                              rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -log_probs.mean()
        return actor_loss, {'actor_loss': actor_loss}

    return (rng, *actor.apply_gradient(loss_fn))


def mse_update(actor: Model, batch: Batch,
               rng: PRNGKey) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions = actor.apply_fn({'params': actor_params},
                                 batch.observations,
                                 training=True,
                                 rngs={'dropout': key})
        actor_loss = ((actions - batch.actions)**2).mean()
        return actor_loss, {'actor_loss': actor_loss}

    return (rng, *actor.apply_gradient(loss_fn))


_log_prob_update_jit = jax.jit(log_prob_update)
_mse_update_jit = jax.jit(mse_update)


class BCLearner(object):

    def __init__(self,
                 cfg: ILConfig,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-3,
                 num_steps: int = int(1e6),
                 hidden_dims: Sequence[int] = (256, 256),
                 distribution: str = 'det'):

        self.distribution = distribution

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)


        schedule_fn = optax.cosine_decay_schedule(-actor_lr, num_steps)
        tx = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(schedule_fn))

        env, env_params = init_base_env(cfg)
        env = LogWrapper(env)
        network = get_network(env, env_params, cfg)
        init_x = env.gen_dummy_obs(env_params)
        network_params = network.init(actor_key, init_x)
        

        self.actor = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = sample_actions(self.rng,
                                                    self.actor.apply_fn,
                                                    self.actor.params,
                                                    observations, temperature,
                                                    self.distribution)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self.distribution == 'det':
            self.rng, self.actor, info = _mse_update_jit(
                self.actor, batch, self.rng)
        else:
            self.rng, self.actor, info = _log_prob_update_jit(
                self.actor, batch, self.rng)
        return info

        
def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):

    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

    def get_initial_states(
        self,
        and_action: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        states = []
        if and_action:
            actions = []
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        for traj in trajs:
            states.append(traj[0][0])
            if and_action:
                actions.append(traj[0][1])

        states = np.stack(states, 0)
        if and_action:
            actions = np.stack(actions, 0)
            return states, actions
        else:
            return states

    def get_monte_carlo_returns(self, discount) -> np.ndarray:
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        mc_returns = []
        for traj in trajs:
            mc_return = 0.0
            for i, (_, _, reward, _, _, _) in enumerate(traj):
                mc_return += reward * (discount**i)
            mc_returns.append(mc_return)

        return np.asarray(mc_returns)

    def take_top(self, percentile: float = 100.0):
        assert percentile > 0.0 and percentile <= 100.0

        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        N = int(len(trajs) * percentile / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def take_random(self, percentage: float = 100.0):
        assert percentage > 0.0 and percentage <= 100.0

        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        np.random.shuffle(trajs)

        N = int(len(trajs) * percentage / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def train_validation_split(self,
                               train_fraction: float = 0.8
                               ) -> Tuple['Dataset', 'Dataset']:
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        train_size = int(train_fraction * len(trajs))

        np.random.shuffle(trajs)

        (train_observations, train_actions, train_rewards, train_masks,
         train_dones_float,
         train_next_observations) = merge_trajectories(trajs[:train_size])

        (valid_observations, valid_actions, valid_rewards, valid_masks,
         valid_dones_float,
         valid_next_observations) = merge_trajectories(trajs[train_size:])

        train_dataset = Dataset(train_observations,
                                train_actions,
                                train_rewards,
                                train_masks,
                                train_dones_float,
                                train_next_observations,
                                size=len(train_observations))
        valid_dataset = Dataset(valid_observations,
                                valid_actions,
                                valid_rewards,
                                valid_masks,
                                valid_dones_float,
                                valid_next_observations,
                                size=len(valid_observations))

        return train_dataset, valid_dataset

        
class D4RLDataset(Dataset):

    def __init__(self,
                 datasets,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        for ds_name, dataset in zip(('train', 'val', 'test'), datasets):
            breakpoint()

            dataset_dict = {}
            # dataset = d4rl.qlearning_dataset(env)

            if clip_to_eps:
                lim = 1 - eps
                dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

            dones_float = np.zeros_like(dataset['rewards'])

            for i in range(len(dones_float) - 1):
                if np.linalg.norm(dataset['observations'][i + 1] -
                                dataset['next_observations'][i]
                                ) > 1e-6 or dataset['terminals'][i] == 1.0:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0

            dones_float[-1] = 1

            dataset = super().__init__(dataset['observations'].astype(np.float32),
                            actions=dataset['actions'].astype(np.float32),
                            rewards=dataset['rewards'].astype(np.float32),
                            masks=1.0 - dataset['terminals'].astype(np.float32),
                            dones_float=dones_float.astype(np.float32),
                            next_observations=dataset['next_observations'].astype(
                                np.float32),
                            size=len(dataset['observations']))

            setattr(self, ds_name, dataset)

                         
def evaluate(agent, env: GenEnv, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats


@hydra.main(version_base="1.3", config_path="gen_env/configs", config_name="il")
def main(cfg: ILConfig):
    validate_config(cfg)

    # glob files of form `gen-XX*elites.npz` and get highest gen number
    gen_files = glob.glob(os.path.join(cfg._log_dir_common, "gen-*_elites.pkl"))
    gen_nums = [int(os.path.basename(f).split("_")[0].split("-")[1]) for f in gen_files]
    latest_gen = max(gen_nums)

    cfg._log_dir_il += f"_env-evo-gen-{latest_gen}"

    # Environment class doesn't matter and will be overwritten when we load in an individual.
    # env = maze.make_env(10, 10)
    env, env_params = init_base_env(cfg)
    rng = np.random.default_rng(cfg.env_exp_id)

    if cfg.overwrite:
        if os.path.exists(cfg._log_dir_il):
            shutil.rmtree(cfg._log_dir_il)

    if not os.path.exists(cfg._log_dir_il):
        os.makedirs(cfg._log_dir_il)

    # Initialize tensorboard logger
    writer = SummaryWriter(cfg._log_dir_il)

    # HACK to load trained run after refactor
    # import sys
    # from gen_env import evo, configs, tiles, rules
    # sys.modules['evo'] = evo
    # sys.modules['configs'] = configs
    # sys.modules['tiles'] = tiles
    # sys.modules['rules'] = rules
    # end HACK

    # Load the transitions from the training set
    train_elites, val_elites, test_elites = load_elite_envs(cfg, latest_gen)

    summary_writer = SummaryWriter(
        os.path.join(cfg._log_dir_il, 'tb', str(cfg.seed)))

    video_save_folder = None if cfg.render_freq == -1 else os.path.join(
        cfg._log_dir_il, 'video', 'eval')

    env, dataset = D4RLDataset(datasets=(train_elites, val_elites, test_elites))

    kwargs = dict(cfg)
    kwargs['num_steps'] = cfg.max_steps
    agent = BCLearner(cfg.seed,
                      env.observation_space.sample()[np.newaxis],
                      env.action_space.sample()[np.newaxis], **kwargs)

    eval_returns = []
    for i in tqdm.tqdm(range(1, cfg.il_max_steps + 1),
                       smoothing=0.1,
                       disable=not cfg.il_tqdm):
        batch = dataset.sample(cfg.il_batch_size)

        update_info = agent.update(batch)

        if i % cfg.log_interval == 0:
            for k, v in update_info.items():
                summary_writer.add_scalar(f'training/{k}', v, i)
            summary_writer.flush()

        if i % cfg.eval_interval == 0:
            eval_stats = evaluate(agent, env, cfg.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(cfg._log_dir_il, f'{cfg.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


def load_elite_envs(cfg, latest_gen):
    # elites = np.load(os.path.join(cfg.log_dir_evo, "unique_elites.npz"), allow_pickle=True)['arr_0']
    # train_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.npz"), allow_pickle=True)['arr_0']
    # val_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.npz"), allow_pickle=True)['arr_0']
    # test_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.npz"), allow_pickle=True)['arr_0']
    # load with pickle instead
    import pickle
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.pkl"), 'rb') as f:
        train_elites = pickle.load(f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.pkl"), 'rb') as f:
        val_elites = pickle.load(f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.pkl"), 'rb') as f:
        test_elites = pickle.load(f)

    return train_elites, val_elites, test_elites

    
if __name__ == "__main__":
    main()