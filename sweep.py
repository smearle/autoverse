from typing import Tuple

import submitit

from gen_env.configs.config import ILConfig
from il_player_jax import main as train_il


@dataclass
class ILSweepParams:
    load_gen: Tuple = (5, 10, 15, 20)


def main():
    sweep_cfgs = []
    for load_gen in (5, 10, 15, 20):
        cfg = ILConfig(
            load_gen=load_gen,
            env_exp_id=14,
            save_freq=10_000,
            eval_freq=10_000,
        )
        
        sweep_cfgs.append(cfg)

    executor = submitit.AutoExecutor(folder='submitit_logs_il')
    executor.update_parameters(
            job_name=f"il",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            # gpus_per_node=1,
            timeout_min=2880,
            slurm_gres='gpu:1',
        )
    executor.map_array(train_il, sweep_cfgs)


if __name__ == '__main__':
    main()