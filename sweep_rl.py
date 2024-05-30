from dataclasses import dataclass
import itertools
from typing import Tuple

import hydra
import submitit

from gen_env.configs.config import SweepRLConfig, RLConfig
from plot import main as plot_rl
from rl_player_jax import main as train_rl


# hypers = {
#     'load_gen': (5, 10, 15, 20, 65),
#     # 'load_il': (True, False),
#     'load_il': Fals
#     'evo_freq': (-1, 1, 10),
#     'n_train_envs': (1, 5, 10, 50, 100, -1),
# }

@dataclass
class HyperParams:
    load_gen: Tuple[int] = (5, 10, 50, 100, 136)
    # load_il: Tuple[bool] = (True, False)
    load_il: Tuple[bool] = (False,)
    # evo_freq: Tuple[int] = (-1, 1, 10)
    evo_freq: Tuple[int] = (-1,)
    n_train_envs: Tuple[int] = (1, 10, 50, 100, -1)


@hydra.main(config_path="gen_env/configs", config_name="sweep_rl_config")
def main(cfg):
    if cfg.mode == 'train':
        main_fn = train_rl
    elif cfg.mode == 'plot':
        main_fn = plot_rl
    hypers = HyperParams()
    h_ks, h_vs = zip(*hypers.__dict__.items())
    all_hyper_combos = list(itertools.product(*h_vs))
    all_hyper_combos = [dict(zip(h_ks, h_v)) for h_v in all_hyper_combos]
    sweep_cfgs = []
    for h in all_hyper_combos:
        e_cfg = RLConfig(
            **h,
            env_exp_id=14,
            overwrite=True,
        )
        sweep_cfgs.append(e_cfg)
    
    if cfg.slurm:
        executor = submitit.AutoExecutor(folder='submitit_logs_rl')
        executor.update_parameters(
                job_name=f"rl",
                mem_gb=30,
                tasks_per_node=1,
                cpus_per_task=1,
                # gpus_per_node=1,
                timeout_min=2880,
                slurm_gres='gpu:1',
            )
        executor.map_array(main_fn, sweep_cfgs)
    else:
        ret = []
        for sc in sweep_cfgs:
            try:
                ret.append(main_fn(sc))
            except Exception as e:
                print(f"Failed with: \n{e}")
        return ret


if __name__ == '__main__':
    main()