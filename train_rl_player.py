import hydra
from stable_baselines3.common.env_util import make_vec_env
from functools import partial

@hydra.main(version_base="1.3", config_path="configs", config_name="rl")
def main(cfg):
    # Now take the imitation-learned policy and do RL with it using sb3 PPO
    policy = bc_trainer.policy
    make_env = partial(init_base_env, cfg)
    env = make_vec_env(make_env, n_envs=100, vec_env_cls=SubprocVecEnv)
    # model = PPO.load(os.path.join(cfg.log_dir, 'policy'))
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=cfg.log_dir, policy_kwargs=policy_kwargs)
    model.set_parameters({'policy': policy.state_dict(), 'policy.optimizer': policy.optimizer.state_dict()})
    model.learn(total_timesteps=1000000, tb_log_name="ppo")
    model.save(os.path.join(cfg.log_dir, "ppo"))
