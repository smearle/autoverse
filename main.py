import argparse
from collections import namedtuple
import itertools
import os
from pathlib import Path
from pdb import set_trace as TT
from typing import Tuple
import cv2
from einops import rearrange, repeat
import gym
from gym import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.tune import CLIReporter, ExperimentAnalysis, grid_search
from ray.tune.logger import Logger
from ray.tune.utils import validate_save_restore
import torch as th
from torch import nn 

parser = argparse.ArgumentParser()
parser.add_argument("--infer", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--render", action="store_true")
# parser.add_argument("--algo", type=str, default="PPO")
# parser.add_argument("--torch", action="store_true")
# parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop_iters", type=int, default=50)
parser.add_argument("--stop_timesteps", type=int, default=100000)
parser.add_argument("--stop_reward", type=float, default=200)


class CustomPPOTrainer(PPOTrainer):
    log_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # wandb.init(**self.config['wandb'])
        # self.checkpoint_path_file = kwargs['config']['checkpoint_path_file']
        # self.ctrl_metrics = self.config['env_config']['conditionals']
        # cbs = self.workers.foreach_env(lambda env: env.unwrapped.cond_bounds)
        # cbs = [cb for worker_cbs in cbs for cb in worker_cbs if cb is not None]
        # cond_bounds = cbs[0]
        # self.checkpoint_path_file = checkpoint_path_file

    def setup(self, config):
        ret = super().setup(config)
        n_params = 0
        param_dict = self.get_weights()['default_policy']

        for v in param_dict.values():
            n_params += np.prod(v.shape)
        print(f'default_policy has {n_params} parameters.')
        # print('model overview: \n', self.get_policy('default_policy').model)
        return ret

    @classmethod
    def get_default_config(cls):
        def_cfg = PPOTrainer.get_default_config()
        def_cfg.update({
            # 'wandb': {
            #     'project': 'PCGRL',
            #     'name': 'default_name',
            #     'id': 'default_id',
            # },
            "exp_id": 0,
        })
        return def_cfg

    # def save(self, *args, **kwargs):
    #     ckp_path = super().save(*args, **kwargs)
    #     with open(self.checkpoint_path_file, 'w') as f:
    #         f.write(ckp_path)
    #     return ckp_path

    # @wandb_mixin
    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        log_result = {k: v for k, v in result.items() if k in self.log_keys}
        log_result['info: learner:'] = result['info']['learner']

        # FIXME: sometimes timesteps_this_iter is 0. Maybe a ray version problem? Weird.
        result['fps'] = result['num_agent_steps_trained'] / result['time_this_iter_s']
        return result


class HamiltonGridEnv(gym.Env):
    adjs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    adj_mask = th.Tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    # def __init__(self, h, w, static_prob=0.1):
    def __init__(self, config: EnvContext):
        self.h, self.w = config['h'], config['w']
        self.static_prob = config['static_prob']
        self.map: np.ndarray = None
        self.static_builds: np.ndarray = None
        self.curr_pos_arr: np.ndarray = None
        self.curr_pos: Tuple[int] = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, 3))
        self.build_hist: list = []
        self.window = None
        self.rend_im: np.ndarray = None

    def reset(self):
        self.map = np.zeros((self.w, self.h), dtype=np.uint8)
        self.static_builds = (np.random.random((self.w, self.h)) < self.static_prob).astype(np.uint8)
        nonstatic_idxs = np.argwhere(self.static_builds != True)
        self.curr_pos = tuple(nonstatic_idxs[np.random.choice(len(nonstatic_idxs))])
        self.curr_pos_arr = np.zeros_like(self.map)
        self.curr_pos_arr[tuple(self.curr_pos)] = 1
        self.map[self.curr_pos] = 1
        self.build_hist = [self.curr_pos]
        return self.get_obs()

    def render(self, mode='human'):
        if self.window is None:
            self.window = cv2.namedWindow('Hamilton Grid', cv2.WINDOW_NORMAL)
            self.rend_im = np.zeros_like(self.map)
            self.rend_im = repeat(self.rend_im, 'h w -> h w 3')
        self.rend_im[self.curr_pos] = [1, 0, 0]
        self.rend_im[np.where(self.static_builds == True)] = [0, 0, 1]
        # self.rend_im[np.where(self.map == 1)] = [0, 1, 0]
        tile_size = 16
        pw = 4
        self.rend_im = repeat(self.rend_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
        b0 = self.build_hist[0]
        for b1 in self.build_hist[1:]:
            x0, x1 = sorted([b0[0], b1[0]])
            y0, y1 = sorted([b0[1], b1[1]])
            self.rend_im[
                x0 * tile_size + tile_size // 2 - pw: x1 * tile_size + tile_size // 2 + pw,
                y0 * tile_size + tile_size // 2 - pw: y1 * tile_size + tile_size // 2 + pw] = [0, 1, 0]
            b0 = b1
        cv2.imshow('Hamilton Grid', self.rend_im * 255)
        cv2.waitKey(1)

    def step(self, action):
        new_pos = tuple(np.clip(np.array(self.curr_pos) + self.adjs[action], (0, 0), (self.w - 1, self.h - 1)))
        if self.map[new_pos] == 1 or self.static_builds[new_pos] == 1:
            reward = -1
            done = True
        else:
            self.map[new_pos] = 1
            self.build_hist.append(new_pos)
            self.curr_pos = new_pos
            self.curr_pos_arr = np.zeros_like(self.map)
            self.curr_pos_arr[tuple(self.curr_pos)] = 1
            done = False
            reward = 1
        nb_idxs = np.array(self.curr_pos) + self.adjs + 1
        neighb_map = np.pad(self.map, 1, mode='constant', constant_values=1)[nb_idxs[:, 0], nb_idxs[:, 1]]
        neighb_static = np.pad(self.static_builds, 1, mode='constant', constant_values=1)[nb_idxs[:, 0], nb_idxs[:, 1]]
        # Terminate if all neighboring tiles already have path or do not belong to graph.
        done = done or (neighb_map | neighb_static).all()
        return self.get_obs(), reward, done, {}

    def get_obs(self):
        obs = rearrange([self.map, self.static_builds, self.curr_pos_arr], 'b h w -> h w b')
        return obs.astype(np.float32)


class TrialProgressReporter(CLIReporter):
    """Progress reporter for ray. More sparing console logs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_timesteps = []

    def should_report(self, trials, done=False):
        old_num_timesteps = self.num_timesteps
        # Only update when new timesteps have occured. # TODO: Only after policy updates?
        self.num_timesteps = [t.last_result['timesteps_total'] if 'timesteps_total' in t.last_result else 0 for t in trials]
        # self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        done = np.any(self.num_timesteps > old_num_timesteps)
        return done


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.torch_sub_model = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )
    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []
    def value_function(self):
        return th.reshape(self.torch_sub_model.value_function(), [-1])


reporter = TrialProgressReporter(
    metric_columns={
        "training_iteration": "itr",
        "timesteps_total": "timesteps",
        "episode_reward_mean": "reward",
        "fps": "fps",
    },
    max_progress_rows=10,
    )


def trial_name_creator(trial):
    # return str(trial)
    cfg = trial.config
    return f"lr-{cfg['lr']:.1e}"

def trial_dirname_creator(trial):
    return trial_name_creator(trial)

PROJ_DIR = Path(__file__).parent.parent
DEBUG = False


if __name__ == "__main__":
    args = parser.parse_args()

    if DEBUG:
        env = HamiltonGridEnv(dict(w=16, h=16))
        for i in range(50):
            env.reset()
            env.render()
            done = False
            while not done:
                obs, reward, done, info = env.step(env.action_space.sample())
                env.render()

    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel
    )

    config = {
        "lr": grid_search([
            1e-2, 
            1e-3, 
            # 1e-4, 
            1e-5, 
            1e-6, 
            1e-7
        ]),
        "exp_id": 0,
        "env": HamiltonGridEnv,  # or "corridor" if registered above
        "env_config": {
            "h": 16,
            "w": 16,
            "static_prob": 0.0,
        },
        # Use GPUs iff 'RLLIB_NUM_GPUS' env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "num_workers": 0,  # parallelism
        "framework": "torch",
        "train_batch_size": 16000,
        "render_env": args.render,
    }
    stop = {
        "training_iteration": args.stop_iters,
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,

    }
 
    ray.init()
    # validate_save_restore(CustomPPOTrainer, config=config)
    env_name = "Hamilton"
    algo_name = "PPO"
    # For convenience, organizing log directories.
    trainer_name = f"{env_name}_PPO"
    tune.register_trainable(trainer_name, CustomPPOTrainer)

    if args.infer:
        analysis = ExperimentAnalysis(f"~/ray_results/{trainer_name}")
        TT()
    else:
        analysis = tune.run(
            trainer_name, 
            checkpoint_at_end = True,
            checkpoint_freq=10,
            config=config, 
            keep_checkpoints_num=1,
            progress_reporter=reporter,
            reuse_actors = True,
            resume="AUTO" if args.resume else False,
            stop=stop,
            trial_name_creator=trial_name_creator,
            trial_dirname_creator=trial_dirname_creator,
            verbose=1,
        )
        # TODO: save analysis object
        # TODO: load checkpoints if inferring/evaluating
        #   - get trials that config *would* create if passed to run
        #   - load each checkpoint
        ray.shutdown()



