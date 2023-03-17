from enum import Enum
import cv2
from pdb import set_trace as TT
from typing import Iterable, List, Tuple

from einops import rearrange, repeat
import gym
from gym import spaces
import numpy as np
# import pygame
# from tkinter import *      

class HamiltonGrid(gym.Env):
    adjs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    
    def __init__(self, config: EnvContext):
        self.h, self.w = config['h'], config['w']
        self.static_prob = config['static_prob']
        self.map: np.ndarray = None
        self.static_builds: np.ndarray = None
        self.curr_pos_arr: np.ndarray = None
        self.curr_pos: Tuple[int] = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, 3))
        self.build_hist: list = []
        self.window = None
        self.rend_im: np.ndarray = None

    def reset(self):
        self.map = np.zeros((self.w, self.h), dtype=np.uint8)
        self.static_builds = (np.random.random((self.w, self.h)) < self.static_prob).astype(np.uint8)
        nonstatic_idxs = np.argwhere(self.static_builds != True)
        self.curr_pos = tuple(nonstatic_idxs[np.random.choice(len(nonstatic_idxs))])
        self.curr_pos_arr = np.zeros_like(self.map)
        self.curr_pos_arr[tuple(self.curr_pos)] = 1
        self.map[self.curr_pos] = 1
        self.build_hist = [self.curr_pos]
        return self.get_obs()

    def render(self, mode='human'):
        if self.window is None:
            self.window = cv2.namedWindow('Hamilton Grid', cv2.WINDOW_NORMAL)
            self.rend_im = np.zeros_like(self.map)
            self.rend_im = repeat(self.rend_im, 'h w -> h w 3')
        self.rend_im[self.curr_pos] = [1, 0, 0]
        self.rend_im[np.where(self.static_builds == True)] = [0, 0, 1]
        # self.rend_im[np.where(self.map == 1)] = [0, 1, 0]
        tile_size = 16
        pw = 4
        self.rend_im = repeat(self.rend_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
        b0 = self.build_hist[0]
        for b1 in self.build_hist[1:]:
            x0, x1 = sorted([b0[0], b1[0]])
            y0, y1 = sorted([b0[1], b1[1]])
            self.rend_im[
                x0 * tile_size + tile_size // 2 - pw: x1 * tile_size + tile_size // 2 + pw,
                y0 * tile_size + tile_size // 2 - pw: y1 * tile_size + tile_size // 2 + pw] = [0, 1, 0]
            b0 = b1
        cv2.imshow('Hamilton Grid', self.rend_im * 255)
        cv2.waitKey(1)

    def step(self, action):
        new_pos = tuple(np.clip(np.array(self.curr_pos) + self.adjs[action], (0, 0), (self.w - 1, self.h - 1)))
        if self.map[new_pos] == 1 or self.static_builds[new_pos] == 1:
            reward = -1
            done = True
        else:
            self.map[new_pos] = 1
            self.build_hist.append(new_pos)
            self.curr_pos = new_pos
            self.curr_pos_arr = np.zeros_like(self.map)
            self.curr_pos_arr[tuple(self.curr_pos)] = 1
            done = False
            reward = 1
        nb_idxs = np.array(self.curr_pos) + self.adjs + 1
        neighb_map = np.pad(self.map, 1, mode='constant', constant_values=1)[nb_idxs[:, 0], nb_idxs[:, 1]]
        neighb_static = np.pad(self.static_builds, 1, mode='constant', constant_values=1)[nb_idxs[:, 0], nb_idxs[:, 1]]
        # Terminate if all neighboring tiles already have path or do not belong to graph.
        done = done or (neighb_map | neighb_static).all()
        return self.get_obs(), reward, done, {}

    def get_obs(self):
        obs = rearrange([self.map, self.static_builds, self.curr_pos_arr], 'b h w -> h w b')
        return obs.astype(np.float32)


class PlayerForces(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NONE = 4