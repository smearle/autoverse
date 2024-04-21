import os

import jax
import numpy as np

from gen_env.configs.config import RLConfig, TrainConfig
from gen_env.envs.play_env import GenEnvParams, PlayEnv
from models import ActorCritic, AutoEncoder, ConvForward, Dense, NCA, SeqNCA


def get_exp_dir(config: RLConfig):
    # if config.env_name == 'PCGRL':
    #     ctrl_str = '_ctrl_' + '_'.join(config.ctrl_metrics) if len(config.ctrl_metrics) > 0 else '' 
    #     exp_dir = os.path.join(
    #         'saves',
    #         f'{config.problem}{ctrl_str}_{config.representation}_{config.model}-' +
    #         f'{config.activation}_w-{config.map_width}_vrf-{config.vrf_size}_' +
    #         (f'cp-{config.change_pct}' if config.change_pct > 0 else '') +
    #         f'arf-{config.arf_size}_sp-{config.static_tile_prob}_' + \
    #         f'bs-{config.max_board_scans}_' + \
    #         f'fz-{config.n_freezies}_' + \
    #         f'act-{"x".join([str(e) for e in config.act_shape])}_' + \
    #         f'nag-{config.n_agents}_' + \
    #         f'{config.seed}_{config.exp_name}')
    # elif config.env_name == 'PlayPCGRL':
    #     exp_dir = os.path.join(
    #         'saves',
    #         f'play_w-{config.map_width}_' + \
    #         f'{config.model}-{config.activation}_' + \
    #         f'vrf-{config.vrf_size}_arf-{config.arf_size}_' + \
    #         f'{config.seed}_{config.exp_name}',
    #     )
    # elif config.env_name == 'Candy':
    #     exp_dir = os.path.join(
    #         'saves',
    #         'candy_' + \
    #         f'{config.seed}_{config.exp_name}',
    #     )
    exp_dir = os.path.join(
        'saves',
        f'{config.env_name}-{config.game}' + \
        f'_{config.seed}_{config.exp_name}',
    )
    return exp_dir


def init_config(config: RLConfig, evo=True):
    config.n_gpus = jax.local_device_count()

    # if config.env_name == 'Candy':
    #     config.exp_dir = get_exp_dir(config)
    #     return config

    config.arf_size = (2 * config.map_width -
                       1 if config.arf_size is None else config.arf_size)
    config.arf_size = config.arf_size if config.arf_size is None \
        else config.arf_size
    config.exp_dir = get_exp_dir(config)
    if evo and hasattr(config, 'n_envs') and hasattr(config, 'evo_pop_size'):
        assert config.n_envs % (config.evo_pop_size * 2) == 0, "n_envs must be divisible by evo_pop_size * 2"
    return config


def get_ckpt_dir(config: RLConfig):
    return os.path.join(get_exp_dir(config), 'ckpts')


def get_network(env: PlayEnv, env_params: GenEnvParams, config: RLConfig):
    action_dim = env.num_actions

    if config.model == "dense":
        network = Dense(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, vrf_size=config.vrf_size,
        )
    if config.model == "conv":
        network = ConvForward(
            action_dim=action_dim, activation=config.activation,
            act_shape=config.act_shape,
            hidden_dims=config.hidden_dims,
        )
    if config.model == "seqnca":
        network = SeqNCA(
            action_dim, activation=config.activation,
            arf_size=config.arf_size,
            vrf_size=config.vrf_size,
        )
    if config.model in {"nca", "autoencoder"}:
        if config.model == "nca":
            network = NCA(
                representation=config.representation,
                action_dim=action_dim,
                activation=config.activation,
            )
        elif config.model == "autoencoder":
            network = AutoEncoder(
                representation=config.representation,
                action_dim=action_dim,
                activation=config.activation,
            )
    # if config.env_name == 'PCGRL':
    # elif config.env_name == 'PlayPCGRL':
    #     network = ActorCriticPlayPCGRL(network)
    else:
        network = ActorCritic(network)
    return network