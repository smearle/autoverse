from dataclasses import dataclass
from typing import Optional
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class Config:
    exp_id: int = 0
    overwrite: bool = False
    n_proc: int = 40
    batch_size: int = 40
    game: str = "maze_for_evo"
    mutate_rules: bool = True
    evaluate: bool = False
    save_freq: int = 1
    render: bool = False
    record: bool = False
    workspace: str = "../gen-game-runs"
    runs_dir_evo: str = "evo_env"
    runs_dir_rl: str = "rl_player"
    runs_dir_il: str = "il_player"
    load_gen: Optional[int] = None
    aggregate_playtraces: bool = False

    log_dir_il: Optional[str] = None
    log_dir_rl: Optional[str] = None
    log_dir_evo: Optional[str] = None
    log_dir_common: Optional[str] = None

    n_il_batches: int = 1000

    load_il: bool = False
    load_rl: bool = False

    n_iters: int = 1000

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)