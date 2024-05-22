import os

import hydra
import pandas as pd

from gen_env.configs.config import RLConfig
from gen_env.utils import init_config
from utils import init_il_config, init_rl_config


smoothing_factor = 1


@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="rl")
def main(cfg: RLConfig):
    init_config(cfg)
    latest_evo_gen = init_il_config(cfg)
    init_rl_config(cfg, latest_evo_gen)
    progress_csv = os.path.join(cfg._log_dir_rl, 'progress.csv')
    # Load data from progress.csv into a pandas dataframe
    progress_df = pd.read_csv(progress_csv)
    # Smooth out with convolution
    progress_df['ep_return'] = progress_df['ep_return'].rolling(window=smoothing_factor).mean()
    # Plot the data using the pandas plot function, save to file
    fig_path = os.path.join(cfg._log_dir_rl, 'progress.png')
    progress_df.plot(x='timestep', y='ep_return', title='Mean Episode Reward vs. Timestep').get_figure()\
        .savefig(fig_path)
    print(f"Plot saved to {fig_path}")

    
if __name__ == "__main__":
    main()