import glob
import os
import jax
from jax import numpy as jnp

from gen_env.configs.config import GenEnvConfig, ILConfig, RLConfig

# Function to stack leaves of PyTrees
def stack_leaves(trees):
    # Make sure each leaf is an array
    def to_array(x):
        x = jnp.array(x) if not isinstance(x, jnp.ndarray) else x
        if x.shape == ():
            x = x.reshape(1)
        return x

    trees = [jax.tree.map(lambda x: to_array(x), tree) for tree in trees]

    # Flatten each tree
    flat_trees_treedefs = [jax.tree.flatten(tree) for tree in trees]
    flat_trees, treedefs = zip(*flat_trees_treedefs)

 
    # Concatenate the flattened lists
    concatenated_leaves = [jnp.stack(leaves) for leaves in zip(*flat_trees)]

    # Rebuild PyTree
    return jax.tree.unflatten(treedefs[0], concatenated_leaves)


def concatenate_leaves(trees):
    # Flatten each tree
    flat_trees_treedefs = [jax.tree.flatten(tree) for tree in trees]
    flat_trees, treedefs = zip(*flat_trees_treedefs)
 
    # Concatenate the flattened lists
    concatenated_leaves = [jnp.concatenate(leaves, axis=0) for leaves in zip(*flat_trees)]

    # Rebuild PyTree
    return jax.tree.unflatten(treedefs[0], concatenated_leaves)


def load_elite_envs(cfg, latest_gen):
    # elites = np.load(os.path.join(cfg.log_dir_evo, "unique_elites.npz"), allow_pickle=True)['arr_0']
    # train_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.npz"), allow_pickle=True)['arr_0']
    # val_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.npz"), allow_pickle=True)['arr_0']
    # test_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.npz"), allow_pickle=True)['arr_0']
    # load with pickle instead
    import pickle
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.pkl"), 'rb') as f:
        train_elites = pickle.load(f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.pkl"), 'rb') as f:
        val_elites = pickle.load(f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.pkl"), 'rb') as f:
        test_elites = pickle.load(f)

    return train_elites, val_elites, test_elites

    
def init_il_config(cfg: ILConfig):
    # glob files of form `gen-XX*elites.npz` and get highest gen number
    gen_files = glob.glob(os.path.join(cfg._log_dir_common, "gen-*_elites.pkl"))
    gen_nums = [int(os.path.basename(f).split("_")[0].split("-")[1]) for f in gen_files]
    latest_gen = max(gen_nums)

    cfg._log_dir_il += f"_env-evo-gen-{latest_gen}"
    cfg._il_ckpt_dir = os.path.abspath(os.path.join(cfg._log_dir_il, "ckpt"))

    return latest_gen


def init_rl_config(config: RLConfig, evo=True):
    config._n_gpus = jax.local_device_count()
    config._rl_exp_dir = get_rl_exp_dir(config)
    if evo and hasattr(config, 'n_envs') and hasattr(config, 'evo_pop_size'):
        assert config.n_envs % (config.evo_pop_size * 2) == 0, "n_envs must be divisible by evo_pop_size * 2"
    return config


def get_rl_exp_dir(config: RLConfig):
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