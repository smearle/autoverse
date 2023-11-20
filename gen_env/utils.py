from dataclasses import dataclass
import os
import cv2
import imageio
import numpy as np
from gen_env.configs.config import Config
from gen_env.envs.play_env import GameDef, PlayEnv, SB3PlayEnv, EnvParams, gen_random_map
from gen_env.evo.individual import Individual
from gen_env.games import GAMES


def validate_config(cfg: Config):
    env_exp_name = (f"{cfg.game}_{'mutRule_' if cfg.mutate_rules else ''}{'fixMap_' if cfg.fix_map else ''}" + 
        f"exp-{cfg.env_exp_id}")

    cfg._log_dir_common = os.path.join(cfg.workspace, env_exp_name)

    player_exp_name = (f"player{'_hideRule' if cfg.hide_rules else ''}")

    cfg._log_dir_player_common = os.path.join(cfg._log_dir_common, player_exp_name)
    cfg._log_dir_rl = os.path.join(cfg._log_dir_player_common, cfg.runs_dir_rl)
    cfg._log_dir_il = os.path.join(cfg._log_dir_player_common, cfg.runs_dir_il)
    # cfg.log_dir_evo = os.path.join(cfg.workspace, cfg.runs_dir_evo, f"exp-{cfg.exp_id}")
    cfg._log_dir_evo = os.path.join(cfg._log_dir_common, cfg.runs_dir_evo)


def save_video(frames, video_path, fps=10):
    """Save a list of frames to a video file.
    Args:
        frames (list): list of frames to save
        video_path (str): path to save the video
        fps (int): frame rate of the video
    """
    imageio.mimwrite(video_path, frames, fps=25, quality=8, macro_block_size=1)


def init_base_env(cfg: Config, sb3=False):
    # env = GAMES[cfg.game].make_env(10, 10, cfg=cfg)
    game_def: GameDef = GAMES[cfg.game].make_env()
    for rule in game_def.rules:
        rule.n_tile_types = len(game_def.tiles)
        rule.compile()
    if game_def.map is None:
        map_arr = gen_random_map(game_def, cfg.map_shape)
    env_params = EnvParams(rules=game_def.rules, map=map_arr)
    if not sb3:
        env = PlayEnv(
            cfg=cfg, height=cfg.map_shape[0], width=cfg.map_shape[1],
            game_def=game_def, params=env_params,
        )
    else:
        env = SB3PlayEnv(
            cfg=cfg, height=cfg.map_shape[0], width=cfg.map_shape[1],
            game_def=game_def, params=env_params,
        )
    # env = evo_base.make_env(10, 10)
    # env = maze.make_env(10, 10)
    # env = maze_for_evo.make_env(10, 10)
    # env = maze_spike.make_env(10, 10)
    # env = sokoban.make_env(10, 10)
    # env.search_tiles = [t for t in env.tiles]
    return env


def load_game_to_env(env: PlayEnv, individual: Individual):
    env._map_queue = [individual.map,]
    env.rules = individual.rules
    env.tiles = individual.tiles
    env._init_rules = individual.rules
    env.init_obs_space()
    return env
