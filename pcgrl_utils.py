import os

import jax
import numpy as np

from gen_env.configs.config import RLConfig
from gen_env.envs.play_env import GenEnvParams, PlayEnv
from models import ActorCritic, AutoEncoder, ConvForward, Dense, NCA, SeqNCA


def get_rl_ckpt_dir(cfg: RLConfig):
    return os.path.join(cfg._log_dir_rl, 'ckpts')


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