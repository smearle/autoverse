import glob
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from plot_rl import load_tensorboard_logs, plot_data
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf

from gen_env.configs.config import RLConfig
from gen_env.utils import init_config
from utils import init_il_config, init_rl_config


@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="rl")
def main(cfg: RLConfig):
    init_config(cfg)
    latest_evo_gen = init_il_config(cfg)

    keys = [
        'il/actor_loss',
        'il/fps',
        'il/train_pct_correct',
        'il/val_pct_correct',
        'il/eval/val_ep_return',
        'il/eval/val_ep_return_max',
        'il/eval/val_ep_return_min',
        'il/eval/train_ep_return',
        'il/eval/train_ep_return_max',
        'il/eval/train_ep_return_min',
        'il/eval/train_mean_return_search',
        'il/eval/train_max_return_search',
        'il/eval/train_min_return_search',
        'il/eval/val_mean_return_search',
        'il/eval/val_max_return_search',
        'il/eval/val_min_return_search',
    ]

    # Load and plot the data
    stepss, valss = load_tensorboard_logs(os.path.join(cfg._log_dir_il, 'tb', '0'), keys)
    plot_data(cfg._log_dir_il, keys, stepss, valss)

    
if __name__ == "__main__":
    main()