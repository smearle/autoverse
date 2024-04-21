import os
import jax
from jax import numpy as jnp

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

    