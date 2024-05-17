import glob
import os
from typing import Tuple
import jax
from jax import numpy as jnp

from gen_env.configs.config import GenEnvConfig, ILConfig, RLConfig
from gen_env.evo.individual import IndividualPlaytraceData

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


def load_elite_envs(cfg, latest_gen) -> Tuple[IndividualPlaytraceData]:
    # elites = np.load(os.path.join(cfg.log_dir_evo, "unique_elites.npz"), allow_pickle=True)['arr_0']
    # train_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.npz"), allow_pickle=True)['arr_0']
    # val_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.npz"), allow_pickle=True)['arr_0']
    # test_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.npz"), allow_pickle=True)['arr_0']
    # load with pickle instead
    import pickle
    # with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.pkl"), 'rb') as f:
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_filtered_train_elites.pkl"), 'rb') as f:
        train_elites = pickle.load(f)
    # with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.pkl"), 'rb') as f:
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_filtered_val_elites.pkl"), 'rb') as f:
        val_elites = pickle.load(f)
    # with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.pkl"), 'rb') as f:
    with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_filtered_test_elites.pkl"), 'rb') as f:
        test_elites = pickle.load(f)
    
    elites = []
    for e in [train_elites, val_elites, test_elites]:
        e: IndividualPlaytraceData
        n_elites = e.env_params.rule_dones.shape[0]
        e = e.replace(env_params=e.env_params.replace(env_idx=jnp.arange(n_elites)))
        elites.append(e)

    return elites

    
def init_il_config(cfg: ILConfig):
    # glob files of form `gen-XX*elites.npz` and get highest gen number
    if cfg.load_gen == -1:
        gen_files = glob.glob(os.path.join(cfg._log_dir_common, "gen-*_elites.pkl"))
        gen_nums = [int(os.path.basename(f).split("_")[0].split("-")[1]) for f in gen_files]
        if len(gen_nums) == 0:
            raise FileExistsError(f"No elite files found in {cfg._log_dir_common}")
        latest_gen = max(gen_nums)

    else:
        latest_gen = cfg.load_gen

    cfg._log_dir_il += f"_env-evo-gen-{latest_gen}"
    cfg._il_ckpt_dir = os.path.abspath(os.path.join(cfg._log_dir_il, "ckpt"))

    return latest_gen


def init_rl_config(cfg: RLConfig, latest_evo_gen: int):
    cfg._n_gpus = jax.local_device_count()
    if cfg.load_il:
        il_ckpt_files = glob.glob(os.path.join(cfg._il_ckpt_dir, "[0-9]*"))
        update_steps = [os.path.basename(f) for f in il_ckpt_files]
        update_steps = [int(us) for us in update_steps if us.isnumeric()]
        latest_il_update_step = max(update_steps)
    else:
        latest_il_update_step = None
    cfg._log_dir_rl +=  f'_evogen-{cfg.load_gen}_accel-{cfg.evo_freq}_' + \
        f'ilstep-{latest_il_update_step}_' + \
        f'tenvs-{cfg.n_train_envs}_' + \
        f's-{cfg.rl_seed}_{cfg.rl_exp_name}'
    return latest_il_update_step