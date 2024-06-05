from dataclasses import dataclass
import itertools
from typing import Tuple

# from cross_eval import cross_eval_il
from cross_eval import cross_eval_il, cross_eval_rl
from evaluate_il import eval_il
from evaluate_rl import eval_rl
import hydra
import submitit

from gen_env.configs.config import ILConfig, RLConfig, SweepConfig
from il_player_jax import main as train_il
from rl_player_jax import main as train_rl
from plot_il import main as plot_il
from plot_rl import main as plot_rl


@dataclass
class HyperParamsIL:
    name: str = 'NONE'
    il_seed: Tuple[int] = (0, 1, 2,)
    load_gen: Tuple[int] = (100,)
    # load_gen: Tuple[int] = (10, 50)
    # load_gen: Tuple[int] = (10, 50, 100)

    # il_lr: Tuple[float] = (1e-4, 1e-5, 1e-6)
    # il_lr: Tuple[float] = (1e-4, 1e-5)
    il_lr: Tuple[float] = (1e-4,)

    obs_window: Tuple[int] = (-1,)
    # obs_window: Tuple[int] = (5, 10, 20, -1)
    # obs_window: Tuple[int] = (5, 10, 20)

    hide_rules: Tuple[bool] = (False,)
    obs_rew_norm: Tuple[bool] = (False,)


@dataclass
class HyperParamsRL:
    name: str = 'NONE'
    rl_seed: int = (0, 1, 2)
    # load_gen: Tuple[int] = (5, 10, 50, 100, 136)
    load_gen: Tuple[int] = (100,)
    # load_il: Tuple[bool] = (True, False)
    load_il: Tuple[bool] = (False,)
    # evo_freq: Tuple[int] = (-1, 1, 10)
    evo_freq: Tuple[int] = (-1,)
    # n_train_envs: Tuple[int] = (1, 10, 50, 100, -1)
    n_train_envs: Tuple[int] = (-1,)
    # obs_window: Tuple[int] = (5, 10, 20, -1)
    obs_window: Tuple[int] = (-1,)

il_sweeps = {
    'obs_win': HyperParamsIL(
        obs_window=(5, 10, 20, -1),
    )
}

rl_sweeps = {
    'obs_win': HyperParamsRL(
        obs_window=(5, 10, 20, -1),
    )
}

for s, dd in il_sweeps.items():
    dd.name = s
for s, dd in rl_sweeps.items():
    dd.name = s


@hydra.main(config_path="gen_env/configs", config_name="sweep")
def main(cfg: SweepConfig):
    if cfg.algo == 'il':
        hypers = il_sweeps[cfg.name]
    elif cfg.algo == 'rl':
        hypers = rl_sweeps[cfg.name]
    hypers_dict = dict(hypers.__dict__)
    sweep_name = hypers_dict.pop('name')
    h_ks, h_vs = zip(*hypers_dict.items())
    all_hyper_combos = list(itertools.product(*h_vs))
    all_hyper_combos = [dict(zip(h_ks, h_v)) for h_v in all_hyper_combos]
    sweep_cfgs = []
    for h in all_hyper_combos:
        if cfg.algo == 'il':
            e_cfg = ILConfig(
                **h,
                env_exp_id=14,
                save_freq=10_000,
                eval_freq=10_000,
                il_max_steps=100_000,
                # overwrite=True,
            )
        elif cfg.algo == 'rl':
            e_cfg = RLConfig(
                **h,
                env_exp_id=14,
                # overwrite=True,
                total_timesteps=100_000_000,
        )
        # Filter out invalid combinations of hyperparameters
        if e_cfg.hide_rules and e_cfg.obs_rew_norm:
            continue

        sweep_cfgs.append(e_cfg)

    # Cross-eval considers the results of all experiments together
    if cfg.mode == 'cross-eval':
        if cfg.algo == 'il':
            cross_eval_il(sweep_cfgs, hypers=h_ks, sweep_name=sweep_name)
        elif cfg.algo == 'rl':
            cross_eval_rl(sweep_cfgs, hypers=h_ks, sweep_name=sweep_name)
        return

    # Set main function based on mode
    elif cfg.mode == 'train':
        if cfg.algo == 'il':
            main_fn = train_il
        elif cfg.algo == 'rl':
            main_fn = train_rl

    elif cfg.mode == 'plot':
        main_fn = plot_il
    elif cfg.mode == 'eval':
        main_fn = eval_il

    if cfg.slurm:
        print('Submitting jobs to SLURM cluster.')
        executor = submitit.AutoExecutor(folder=f'submitit_logs_{cfg.algo}')
        executor.update_parameters(
                job_name=f"{cfg.algo}_{cfg.mode}_{sweep_name}",
                mem_gb=30,
                tasks_per_node=1,
                cpus_per_task=1,
                # gpus_per_node=1,
                timeout_min=2880,
                slurm_gres='gpu:1',
            )
        executor.map_array(main_fn, sweep_cfgs)

    else:
        print('Running jobs locally (in sequence).')

        for s_cfg in sweep_cfgs:
            if cfg.skip_failures:
                try:
                    main_fn(s_cfg)
                except Exception as e:
                    print(f"Failed with {e}.")
                    breakpoint()
            else:
                main_fn(s_cfg)


if __name__ == '__main__':
    main()