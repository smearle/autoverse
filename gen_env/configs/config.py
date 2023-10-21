from dataclasses import dataclass
from typing import Optional
import hydra
from hydra.core.config_store import ConfigStore
from numpy import int32
from omegaconf import DictConfig


@dataclass
class Config:
    env_exp_id: int = 0
    player_exp_id: int = 0
    overwrite: bool = False
    n_proc: int = 4
    batch_size: int = 40
    game: str = "maze_for_evo_2"
    mutate_rules: bool = True
    fix_map: bool = False
    evaluate: bool = False
    eval_freq: int = 1
    save_freq: int = 1
    render: bool = False
    record: bool = True
    workspace: str = "../gen-game-runs"
    runs_dir_evo: str = "evo_env"
    runs_dir_rl: str = "rl_player"
    runs_dir_il: str = "il_player"
    load_gen: Optional[int] = None
    collect_elites: bool = False

    _log_dir_il: Optional[str] = None
    _log_dir_rl: Optional[str] = None
    _log_dir_evo: Optional[str] = None
    _log_dir_common: Optional[str] = None
    _log_dir_player_common: Optional[str] = None

    n_il_batches: float = 1_000_000

    load_il: bool = False
    load_rl: bool = False

    n_rl_iters: float = 1e6

    hide_rules: bool = False
    
    map_shape: tuple = (10, 10)

    window_shape: tuple = (800, 800)

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)