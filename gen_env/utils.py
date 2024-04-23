from dataclasses import dataclass
import os
from typing import Tuple
import chex

import cv2
import imageio
from jax import numpy as jnp
import jax
import numpy as np

from gen_env.configs.config import GenEnvConfig
from gen_env.envs.play_env import GameDef, PlayEnv, SB3PlayEnv, GenEnvParams, gen_random_map
from gen_env.evo.individual import IndividualData
from gen_env.games import GAMES
from gen_env.rules import RuleData, compile_rule, gen_rand_rule


def init_config(cfg: GenEnvConfig):
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


def init_base_env(cfg: GenEnvConfig, sb3=False) -> Tuple[PlayEnv, GenEnvParams]:
    # env = GAMES[cfg.game].make_env(10, 10, cfg=cfg)
    game_def: GameDef = GAMES[cfg.game].make_env()
    for rule in game_def.rules:
        rule.n_tile_types = len(game_def.tiles)
        rule = compile_rule(rule)
    if game_def.map is None:
        key = jax.random.PRNGKey(cfg.seed)
        map_arr = gen_random_map(key, game_def, cfg.map_shape).astype(jnp.int16)
    # TODO: Flatten rule and subrule dimensions!
    rules_int = [rule.subrules_int for rule in game_def.rules]

    # FIXME: Below is misplaced. Needs to happen before subrules get rotated!
    # Find largest rule dimensions
    # max_rule_dims = np.max([rule.subrules_int.shape[-2:] for rule in game_def.rules], axis=0)
    # Pad rules with -1s to make them all the same size
    # rules_int = [np.pad(rule_int, ((0, 0), (0, 0), (0, 0), (0, max_rule_dims[0] - rule_int.shape[-2]), (0, max_rule_dims[1] - rule_int.shape[-1])), constant_values=-1) for rule_int in rules_int]

    # TODO: Deal with non-rotate/rotated mix of rules. Gotta flatten along subrule dimension basically
    rules_int = jnp.array(rules_int, dtype=jnp.int16)

    rule_rewards = jnp.array([rule.reward for rule in game_def.rules])
    rules = RuleData(rule=rules_int, reward=rule_rewards)
    rule_dones = jnp.array([rule.done for rule in game_def.rules], dtype=bool)
    player_placeable_tiles = \
        jnp.array([tile.idx for tile, placement_rule in game_def.player_placeable_tiles], dtype=int)
    params = GenEnvParams(rules=rules, map=map_arr,
                       rule_dones=rule_dones,
                       player_placeable_tiles=player_placeable_tiles)
    if not sb3:
        env = PlayEnv(
            cfg=cfg, height=cfg.map_shape[0], width=cfg.map_shape[1],
            game_def=game_def, params=params, max_episode_steps=cfg.max_episode_steps,
        )
    else:
        env = SB3PlayEnv(
            cfg=cfg, height=cfg.map_shape[0], width=cfg.map_shape[1],
            game_def=game_def, params=params,
        )
    # env = evo_base.make_env(10, 10)
    # env = maze.make_env(10, 10)
    # env = maze_for_evo.make_env(10, 10)
    # env = maze_spike.make_env(10, 10)
    # env = sokoban.make_env(10, 10)
    # env.search_tiles = [t for t in env.tiles]
    return env, params



def gen_rand_env_params(cfg: GenEnvConfig, rng: jax.random.PRNGKey, game_def, rules: RuleData) -> GenEnvParams:
    rand_rule = gen_rand_rule(rng, rules)
    rule_rewards = jax.random.randint(rng, (len(game_def.rules),), minval=-1, maxval=2, dtype=jnp.int32)
    rules = RuleData(rule=rand_rule, reward=rule_rewards)

    map_arr = gen_random_map(rng, game_def, cfg.map_shape).astype(jnp.int16)
    # Make rules random
    rule_dones = jnp.zeros((len(game_def.rules),), dtype=bool)
    player_placeable_tiles = \
        jnp.array([tile.idx for tile, placement_rule in game_def.player_placeable_tiles], dtype=int)
    params = GenEnvParams(rules=rules, map=map_arr,
                          rule_dones=rule_dones,
                          player_placeable_tiles=player_placeable_tiles)
    return params

# def load_game_to_env(env: PlayEnv, individual: IndividualData):
#     env._map_queue = [individual.params.map,]
#     env.rules = individual.params.rules
#     env.tiles = individual.tiles
#     env._init_rules = individual.rules
#     params = get_params_from_individual(env, individual)
#     env.init_obs_space(params=params)
#     return env


# TODO: individual should basically be its own dataclass
# def get_params_from_individual(env: PlayEnv, individual: IndividualData):
#     params = GenEnvParams(rules=jnp.array([rule.subrules_int for rule in individual.rules], dtype=jnp.int16),
#                        map=individual.map,
#                        rule_rewards=jnp.array([rule.reward for rule in individual.rules]),
#                        rule_dones=jnp.array([rule.done for rule in individual.rules], dtype=bool),
#                        player_placeable_tiles=jnp.array([tile.idx for tile, placement_rule in env.game_def.player_placeable_tiles]))
#     return params
