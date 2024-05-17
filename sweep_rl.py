from typing import Tuple

import submitit

from gen_env.configs.config import RLConfig
from rl_player_jax import main as train_rl


# mode = 'local'
mode = 'slurm'


def main():
    sweep_cfgs = []
    for load_gen in (5, 10, 15, 20, 65):
    # for load_gen in (65,):
        for load_il in (True, False):
        # for load_il in (False,):
        # for load_il in (True,):
            for evo_freq in (-1, 1, 10):
            # for evo_freq in (-1,):
            # for evo_freq in (1, 10):
                cfg = RLConfig(
                    load_gen=load_gen,
                    load_il=load_il,
                    evo_freq=evo_freq,
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