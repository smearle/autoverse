import argparse
from collections import namedtuple
import itertools
import os
from pathlib import Path
from pdb import set_trace as TT
from typing import List, Optional, Tuple
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
from ray.tune import Callback, CLIReporter, ExperimentAnalysis, grid_search, Stopper
from ray.tune.logger import Logger
from ray.tune.trial import Trial
from ray.tune.utils import validate_save_restore
import torch as th
from torch import nn

from env import HamiltonGrid 


parser = argparse.ArgumentParser()
parser.add_argument("--infer", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume_sequential", action="store_true", help="A bit of a hack. Resumes one experiment at a time "
    "if we need to run it for more iterations than specified in the initial config")
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


class CustomCallbacks(Callback):
    # arguments here match Experiment.public_spec
    def setup(
        self,
        stop: Optional["Stopper"] = None,
        num_samples: Optional[int] = None,
        total_num_samples: Optional[int] = None,
        **info,
    ):
        """Called once at the very beginning of training.

        Any Callback setup should be added here (setting environment
        variables, etc.)

        Arguments:
            stop: Stopping criteria.
                If ``time_budget_s`` was passed to ``tune.run``, a
                ``TimeoutStopper`` will be passed here, either by itself
                or as a part of a ``CombinedStopper``.
            num_samples: Number of times to sample from the
                hyperparameter space. Defaults to 1. If `grid_search` is
                provided as an argument, the grid will be repeated
                `num_samples` of times. If this is -1, (virtually) infinite
                samples are generated until a stopping condition is met.
            total_num_samples: Total number of samples factoring
                in grid search samplers.
            **info: Kwargs dict for forward compatibility.
        """
        pass

    def on_trial_restore(
        self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        """Called after restoring a trial instance.

        Arguments:
            iteration: Number of iterations of the tuning loop.
            trials: List of trials.
            trial: Trial that just has been restored.
            **info: Kwargs dict for forward compatibility.
        """
        pass


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
EnvCls = HamiltonGrid


if __name__ == "__main__":
    args = parser.parse_args()

    if DEBUG:
        env = EnvCls(dict(w=16, h=16))
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
        ]),
        "exp_id": 0,
        "env": EnvCls,  # or "corridor" if registered above
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

    # TODO: if we want to resumt training with more training iterations, we have to restore each trial sequentially 
    #   (eesh!)
    if args.infer or args.resume_sequential:
        config['lr'] = 0.0  # dummy to allow us to initialize the trainer without issue
        trainer = CustomPPOTrainer(config=config)
        analysis = ExperimentAnalysis(f"~/ray_results/{trainer_name}")
        ckp_paths = [analysis.get_trial_checkpoints_paths(analysis.trials[i]) for i in range(len(analysis.trials))]
        assert np.all([len(paths) == 1 for paths in ckp_paths])
        ckp_paths = [p for paths in ckp_paths for p in paths]
        for ckp_path in ckp_paths:
            if args.infer:
                trainer.restore(ckp_path[0])
                trainer.evaluate()
            elif args.resume_sequential:
                analysis = tune.run(
                    trainer_name, 
                    callbacks=[CustomCallbacks()],
                    checkpoint_at_end = True,
                    checkpoint_freq=10,
                    config=config, 
                    keep_checkpoints_num=1,
                    progress_reporter=reporter,
                    reuse_actors=True,
                    resume="AUTO" if args.resume else False,
                    stop=stop,
                    trial_name_creator=trial_name_creator,
                    trial_dirname_creator=trial_dirname_creator,
                    verbose=1,
                )
    else:
        analysis = tune.run(
            trainer_name, 
            callbacks=[CustomCallbacks()],
            checkpoint_at_end = True,
            checkpoint_freq=10,
            config=config, 
            keep_checkpoints_num=1,
            progress_reporter=reporter,
            reuse_actors=True,
            resume="AUTO" if args.resume else False,
            stop=stop,
            trial_name_creator=trial_name_creator,
            trial_dirname_creator=trial_dirname_creator,
            verbose=1,
        )
        ray.shutdown()



