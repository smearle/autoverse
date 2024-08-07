# autoverse

- To evolve games for maximum complexity according to a search-based player agent, run `python evo_env.py`.
- To render the fittest environment-solution pairs found so far, run `python evo_env.py evaluate=True`
- To aggregate all unique environments that have been generated over the course of evolution, and record playtraces of their solutions as generated by search, run `python evo_env.py collect_elites=True`
- To imitation learn on these "oracle" playtraces, run `python train_il_player.py`.

## Installation

```
conda create -n autoverse python==3.12
conda activate autoverse
pip install -r requirements.txt
```

Install jax (with GPU support, if you have one), by following the instructions at https://jax.readthedocs.io/en/latest/installation.html



## Interactive environment:
```
python human_env.py game=lava_maze window_shape=[1000,1000]
```
Use the left and right arrow keys to rotate the agent, and `q` to place a `force` tile in front of the agent, which will
move it forward where appropriate. The game is defined in `games/lava_maze.py`.

## Profile environment:
```
python profile_env.py game=lava_maze
```
This will initialize a level in the give game, and take random actions. It will then print the FPS of `reset()` and `step()`.

## Render solution:
```
python render_solution.py game=lava_maze
```
This will initialize a random level in the given game, and search for some number of iterations for a solution. It will
then save a video of the solution.

## Render in blender (deprecated)
To render in blender:
```bash
blender render_scene.blend --python enjoy_blender.py
```

# Notes

- Because jax allocates gpu memory in advance, making `total_timesteps` too large will cause an out-of-memory error. However, complete jobs can be resumed with increased `total_timesteps` where jax will only allocate the difference between completed and pending timesteps in advance.

- Looks like certain rules can be applied every turn in the same spot. Must be because a rotation (subrule) of this rule is overwriting itself? We are applying rules and subrules in parallel with `vmap`. I guess this is not a bug, they could overwrite each other in a different way if we were to apply the (subrules) sequentially...

# TODO:

- [] Deal with case where dataset of playtraces generated by evo+search gets very large. In this case, we cannot place the whole thing on GPU without OOM. One solution could be to use a `jax.experimental.io_callback()` to pass a batch of environments from GPU-->CPU every so often. See, e.g., how this is done in github.com/daphnecor/waymax
- [] Make search more efficient, taking advantage of jax parallelism somehow (since right now we take one step in one environment at a time, then hash the state to see if it's visited; would it be advantageous to step a huge number of mutants at once, then hash the resultant states---potentially distributing this process across CPUs? Or is there some clever way to hash the states in jax on the GPU?)
- [] When the search cap is increased, maybe we want to pick up where we left off searching in the current elite environments? It's only fair. We should also hash mutant env params against existing elites, if we're not already
