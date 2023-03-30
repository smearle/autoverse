import random

import gym
import numpy as np

from games import GAMES
from envs.play_env import PlayEnv
from rules import Rule
from tiles import TileType


class GenEnv(gym.Env):
    """A wrapper that lets us edit an environment's map and rules."""
    def __init__(self, *args, width, height, **kwargs):
        self.width = width
        self.height = height
        self.play_env: PlayEnv = GAMES['maze_for_evo'].make_env(height=width, width=height)
        self.tiles = self.play_env.tiles
        self.rules = self.play_env.rules
        self.map = self.play_env.map

        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype='uint8')
        self.action_space = gym.spaces.MultiDiscrete((
            width * height * len(self.tiles), # tile to edit (flattened idx)
            # len(self.tiles) + 1, # new tile type

            2 * 2 * 1 * 3 * (len(self.tiles) + 1),
            # 2, 2, 1*3, len(self.tiles) + 1, # rule to edit (io, subp, ij, new tile

            # FIXME... box?
            3, # new reward
            ))

    # TODO: mutate map and rules (and tiles, objects, events)
    def step(self, action):
        """
        FOR NOW: We'll assume a multidiscrete action space / MultiBinary
        LATER: We'll assume action is flattened box between 0 and 1. (?)

        One edit to map:
        - h + w + T elements to indicate tile position and new type.

        One rule edit
        - 3 + 3 + 2 + T elements are new tile in an rule with 3 * 3 patch and 2 co-occrrence layers ("subpatterns" elsewhere)
        - 3 for reward value (kinda silly would make more sense with box...)
        """
        tiles_none = self.tiles + [None]

        ij_tild = action[0]
        ij = ij_tild // len(self.tiles)
        tile_idx = ij_tild % len(self.tiles)

        disc_map = self.map.argmax(axis=0)
        disc_map.flat[ij] = tile_idx
        self.map = PlayEnv.repair_map(disc_map, self.tiles)
        self.play_env.map = self.map 
        
        rule = self.rules[-1]  # HARDCODE edit last rule of maze_for_evo
    
        idx_tile = action[1]
        print(idx_tile)
        io_idx = idx_tile // len(tiles_none)
        print(io_idx)
        new_tile_idx = idx_tile % len(tiles_none)

        new_rew = action[2]

        # old_tile = rule._in_out[io_idx, subp_idx, i, j]
        old_tile = rule._in_out.flat[io_idx]
        old_tile_idx = old_tile.get_idx() if old_tile is not None else len(self.tiles) # None <--> -1
        new_in_out = rule._in_out.copy()
        # new_tile_idx = random.randint(0, len(tiles_none) - 1)
        new_in_out.flat[io_idx] = tiles_none[new_tile_idx]
        # if Rule.is_valid(new_in_out):
        rule._in_out = new_in_out


        print(f'old tile: {old_tile_idx}, new tile: {new_tile_idx}, new rew: {new_rew}, timestep: {self.play_env.n_step}')
        print(f'tiles are same: {old_tile_idx == new_tile_idx}')

        rule.reward = new_rew

        obs = None
        rew = 0 
        done = False
        info = {}

        p_obs, p_rew, p_done, p_info = self.play_env.step(0)
        done = p_done

        return obs, rew, done, info

    def reset(self):
        play_obs =  self.play_env.reset()
        self.map = self.play_env.map
        return play_obs

    def render(self, mode='human'):
        return self.play_env.render(mode=mode)




if __name__ == '__main__':
    # Play a gen_env episode with random actions. Render rgb frames and save a video
    env = GenEnv(width=10, height=10)
    obs = env.reset()
    frames = [env.render(mode='rgb_array')]
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        frames.append(env.render(mode='rgb_array'))
    env.close()
    # Save video
    import imageio
    imageio.mimsave('gen_env.mp4', frames, fps=10)


