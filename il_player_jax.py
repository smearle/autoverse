import glob
import os
import shutil

import hydra
import numpy as np
from tensorboardX import SummaryWriter

from gen_env.configs.config import GenEnvConfig
from gen_env.utils import init_base_env, validate_config


@hydra.main(version_base="1.3", config_path="gen_env/configs", config_name="il")
def main(cfg: GenEnvConfig):
    validate_config(cfg)

    # glob files of form `gen-XX*elites.npz` and get highest gen number
    gen_files = glob.glob(os.path.join(cfg._log_dir_common, "gen-*_elites.npz"))
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

    # elites = np.load(os.path.join(cfg.log_dir_evo, "unique_elites.npz"), allow_pickle=True)['arr_0']
    train_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.npz"), allow_pickle=True)['arr_0']
    val_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.npz"), allow_pickle=True)['arr_0']
    test_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.npz"), allow_pickle=True)['arr_0']

    transitions_path = os.path.join(cfg._log_dir_il, "transitions.npz")

    breakpoint()

    
if __name__ == "__main__":
    main()