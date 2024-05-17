from dataclasses import dataclass
import itertools
from typing import Tuple

import submitit

from gen_env.configs.config import RLConfig
from rl_player_jax import main as train_rl


# mode = 'local'
mode = 'slurm'

# hypers = {
#     'load_gen': (5, 10, 15, 20, 65),
#     # 'load_il': (True, False),
#     'load_il': Fals
#     'evo_freq': (-1, 1, 10),
#     'n_train_envs': (1, 5, 10, 50, 100, -1),
# }

@dataclass
class HyperParams:
    load_gen: Tuple[int] = (5, 10, 15, 20, 65)
    # load_il: Tuple[bool] = (True, False)
    load_il: Tuple[bool] = (False,)
    # evo_freq: Tuple[int] = (-1, 1, 10)
    evo_freq: Tuple[int] = (-1,)
    n_train_envs: Tuple[int] = (1, 5, 10, 50, 100, -1)


def main():
    hypers = HyperParams()
    h_ks, h_vs = zip(*hypers.__dict__.items())
    all_hyper_combos = list(itertools.product(*h_vs))
    all_hyper_combos = [dict(zip(h_ks, h_v)) for h_v in all_hyper_combos]
    sweep_cfgs = []
    for h in all_hyper_combos:
        cfg = RLConfig(
            **h,
            env_exp_id=14,
            overwrite=True,
        )
        sweep_cfgs.append(cfg)

    if mode == 'slurm':
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
        executor.map_array(train_rl, sweep_cfgs)
    else:
        ret = [train_rl(sc) for sc in sweep_cfgs]


if __name__ == '__main__':
    main()