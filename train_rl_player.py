import argparse
from collections import namedtuple
from functools import partial
import os
from pathlib import Path
from typing import List, Optional, Tuple
import hydra
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune import Callback, CLIReporter, ExperimentAnalysis, grid_search, Stopper
from ray.tune.automl import ContinuousSpace, DiscreteSpace
from ray.tune.automl.search_policy import AutoMLSearcher, GridSearch
from ray.tune.automl.search_space import SearchSpace
from ray.tune.registry import register_env
from ray.tune.suggest import Repeater, SearchAlgorithm
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.experiment.trial import Trial
from ray.tune.utils import validate_save_restore

# from env import HamiltonGrid 
from games import GAMES, maze, dungeon, make_env_rllib
from model import CustomFeedForwardModel
from utils import save_video




parser = argparse.ArgumentParser()
parser.add_argument("--infer", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume_sequential", action="store_true", help="A bit of a hack. Resumes one experiment at a time "
    "if we need to run it for more iterations than specified in the initial config")
parser.add_argument("--render", action="store_true")
parser.add_argument("--record", action="store_true")
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


reporter = TrialProgressReporter(
    metric_columns={
        "training_iteration": "itr",
        "timesteps_total": "timesteps",
        "episode_reward_mean": "reward",
        "episode_len_mean": "len",
        "fps": "fps",
    },
    max_progress_rows=10,
    )

def create_trial_name(cfg):
    return f"lr-{cfg['lr']:.1e}"

def trial_name_creator(trial):
    # return str(trial)
    return create_trial_name(trial.config)

def trial_dirname_creator(trial):
    return trial_name_creator(trial)


PROJ_DIR = Path(__file__).parent.parent
# EnvCls = HamiltonGrid
EnvCls = maze.make_env(width=16, height=16)


@hydra.main(config_path="configs", config_name="rl")
def main(cfg):
    # Register custom environment with ray
    # register_env("maze", maze.make_env)
    # register_env("dungeon", dungeon.make_env)
    env_name = cfg.env
    # env_name = exp_cfg["env"]
    make_env_func = GAMES[env_name].make_env
    make_rllib_env = partial(make_env_rllib, make_env_func=make_env_func)
    register_env("gen_env", make_rllib_env)

    if cfg.debug:
        # env = EnvCls(dict(w=16, h=16))

        print("Debug: Creating env...")
        env = maze.make_env(width=16, height=16)
        print("Debuging environment with random actions.")
        for i in range(5):
            env.reset()
            # env.render()
            done = False
            while not done:
                obs, reward, done, info = env.step(env.action_space.sample())
                # env.render()
        
        # print("Debug: Creating model...")
        # model = CustomFeedForwardModel(env.observation_space, env.action_space, cfg)
        # print("Debugging environment with model actions.")
        # for i in range(5):
        #     env.reset()
        #     # env.render()
        #     done = False
        #     while not done:
        #         obs, reward, done, info = env.step(model.forward(obs))
        #         # env.render()

    ModelCatalog.register_custom_model(
        "my_model", CustomFeedForwardModel
    )

    config = {
        # **exp_cfg,
        # "lr": grid_search([
        #     1e-2, 
        #     1e-3, 
        #     # 1e-4,
        #     # 1e-5,
        # ]),
        "lr": cfg.lr,
        "exp_id": 0,
        # "env": EnvCls,  # or "corridor" if registered above
        "env": "gen_env",
        "env_config": {
            "width": 16,
            "height": 16,
            # "static_prob": 0.0,
        },
        # Use GPUs iff 'RLLIB_NUM_GPUS' env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
        "num_gpus": cfg.num_gpus,
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "num_workers": cfg.num_workers if not cfg.infer else 0,  # parallelism
        "framework": "torch",
        "train_batch_size": 16000,
        "render_env": cfg.render,
        "num_envs_per_worker": 20,
        "monitor": True,
        # "evaluation_config":
        #     {
        #         "render_env": True,
        #     },
    }

    # FIXME: Can't reload multiple trials at once with different (longer) stop condition. So not including any stop
    #  stop conditions for now (will this train indefinitely as intended?).
    stop = {
        # "training_iteration": args.stop_iters,
        # "episode_reward_mean": args.stop_reward,
        # "timesteps_total": args.stop_timesteps,
    }
 
    ray.init()
    # validate_save_restore(CustomPPOTrainer, config=config)
    # env_name = "Hamilton"
    algo_name = "PPO"
    # For convenience, organizing log directories.
    trainer_name = f"{env_name}_{algo_name}"
    tune.register_trainable(trainer_name, CustomPPOTrainer)
    local_dir = "./runs"
    exp_name =f"{trainer_name}_{create_trial_name(cfg)}"

    def launch_analysis():
        return tune.run(
            run_or_experiment=trainer_name, 
            name=exp_name,
            callbacks=[CustomCallbacks()],
            checkpoint_at_end = True,
            checkpoint_freq=10,
            config=config, 
            keep_checkpoints_num=2,
            local_dir=local_dir,
            progress_reporter=reporter,
            reuse_actors=True,
            resume="AUTO" if cfg.resume else False,
            # resume=False,
            # search_alg=search_alg if args.resume else None,
            stop=stop,
            # trial_name_creator=trial_name_creator,
            # trial_dirname_creator=trial_dirname_creator,
            verbose=1,
        )

    if cfg.infer:
        config['lr'] = 0.0  # dummy to allow us to initialize the trainer without issue
        trainer = CustomPPOTrainer(config=config)
        analysis = ExperimentAnalysis(os.path.join(local_dir, exp_name))
        ckp_paths = [analysis.get_trial_checkpoints_paths(analysis.trials[i]) for i in range(len(analysis.trials))]
        assert np.all([len(paths) == 1 for paths in ckp_paths]), f"Expected 1 checkpoint per trial, got {[len(paths) for paths in ckp_paths]}."
        ckp_paths = [p for paths in ckp_paths for p in paths]
        for ckp_path in ckp_paths:
            trainer.restore(ckp_path[0])
            if cfg.record:
                # Manually step through an apisode and collect RGB frames from rendering
                env = trainer.workers.local_worker().env
                for ep_i in range(10):
                    obs = env.reset()
                    done = False
                    frames = []
                    while not done:
                        # Get actions from the policy
                        action_dict = trainer.compute_action(obs)
                        # Take actions in the environment
                        obs, reward, done, info = env.step(action_dict)
                        # Render the environment
                        frames.append(env.render(mode="rgb_array"))
                    # Save the frames as a video
                    video_path = os.path.join(local_dir, exp_name, f"{os.path.basename(ckp_path[0])}_ep-{ep_i}.mp4")
                    save_video(frames, video_path, fps=10)

            for i in range(10):
                print(f'eval {i}')
                trainer.evaluate()
                breakpoint()
            # elif args.resume_sequential:
                # analysis = launch_analysis()
    else:
        analysis = launch_analysis()
        ray.shutdown()


if __name__ == "__main__":
    main()

    # args = parser.parse_args()
    # batch_cfg = {
    #     "lr": [
    #         # 1e-2,
    #         1e-3,
    #     ],
    #     "env": ["dungeon"],
    # }
    # exp_cfgs = [{k: batch_cfg[k][0] for k in batch_cfg}]
    # for k, v in batch_cfg.items():
    #     for i in v[1:]:
    #         exp_cfgs += [{**exp_cfg, k: i} for exp_cfg in exp_cfgs]
    # for exp_cfg in exp_cfgs:
