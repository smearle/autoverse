import glob
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf

from gen_env.configs.config import RLConfig
from gen_env.utils import init_config
from utils import init_il_config, init_rl_config


smoothing_factor = 1


# Function to load TensorBoard logs
def load_tensorboard_logs(log_dir, keys):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()  # Load events from file

    stepss, valss = [], []

    # Retrieve scalars logged under the specified tag
    for k in keys:
        scalar_events = event_acc.Scalars(k)
        
        # Extract step and value information
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        stepss.append(steps)
        valss.append(values)
    
    return stepss, valss

# Plotting the data using matplotlib
def plot_data(log_dir, keys, stepss, valuess):
    for key, steps, values in zip(keys, stepss, valuess):
        if 'search' in key:
            # Assert all valuess are the same
            assert all([v == values[0] for v in values])
            # Save this value to a text file
            with open(os.path.join(log_dir, f"{key.replace('/', '_')}.txt"), 'w') as f:
                f.write(str(values[0]))
            # And print it
            print(f"{key}: {values[0]}")
            continue
        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, label=f'{key}')
        plt.xlabel('Step')
        plt.ylabel(f'{key}')
        plt.title(f'{key}')
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(log_dir, f"{key.replace('/', '_')}.png")
        plt.savefig(fig_path)
        print(f"Plot saved to {fig_path}")


@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="rl")
def main(cfg: RLConfig):
    init_config(cfg)
    latest_evo_gen = init_il_config(cfg)
    init_rl_config(cfg, latest_evo_gen)

    keys = [
        'rl/ep_return',
        'rl/ep_return_max',
        'rl/ep_return_min',
        'rl/ep_length',
        'rl/fps',
        'rl/eval/ep_return',
        'rl/eval/ep_return_max',
        'rl/eval/ep_return_min',
    ]

    # Load and plot the data
    stepss, valss = load_tensorboard_logs(cfg._log_dir_rl, keys)
    plot_data(cfg._log_dir_rl, keys, stepss, valss)

    # progress_csv = os.path.join(cfg._log_dir_rl, 'progress.csv')
    # # Load data from progress.csv into a pandas dataframe
    # progress_df = pd.read_csv(progress_csv)
    # # Smooth out with convolution
    # progress_df['ep_return'] = progress_df['ep_return'].rolling(window=smoothing_factor).mean()
    # # Plot the data using the pandas plot function, save to file
    # fig_path = os.path.join(cfg._log_dir_rl, 'progress.png')
    # progress_df.plot(x='timestep', y='ep_return', title='Mean Episode Reward vs. Timestep').get_figure()\
    #     .savefig(fig_path)
    # print(f"Plot saved to {fig_path}")

    
if __name__ == "__main__":
    main()