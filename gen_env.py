from enum import Enum
from pdb import set_trace as TT
from typing import Iterable, List, Tuple

import cv2
from einops import rearrange, repeat
import gym
from gym import spaces
import numpy as np
import pygame

from games.common import colors
from rules import Rule
from tiles import TileType


class GenEnv(gym.Env):
    adjs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    tile_size = 16
    
    def __init__(self, width: int, height: int, tiles: Iterable[TileType], rules: List[Rule], 
            player_placeable_tiles: List[TileType], baseline_rew: int = 0, done_at_reward: int = None,
            max_episode_steps: int = 100):
        """_summary_

        Args:
            width (int): The width of the 2D game map.
            height (int): The height of the 2D game map.
            tiles (list): A list of TileType objects. Must include a tile-type with name `player`.
            rules (list): A list of Rule objects, between TileTypes.
            done_at_reward (int): Defaults to None. Otherwise, episode ends when reward reaches this number.
        """
        self._done = False
        self._n_step = 0
        self.max_episode_steps = max_episode_steps
        self.w, self.h = width, height
        self.tiles = tiles
        # [setattr(tile, 'idx', i) for i, tile in enumerate(tiles)]
        self._baseline_rew = baseline_rew
        tiles_by_name = {t.name: t for t in tiles}
        # Assuming here that we always have player and floor...
        self.player_idx = tiles_by_name['player'].idx
        self.tile_probs = [tile.prob for tile in tiles]
        self.tile_colors = np.array([tile.color if tile.color is not None else colors['error'] for tile in tiles], dtype=np.uint8)
        self.rules = rules
        self.map: np.ndarray = None
        self.static_builds: np.ndarray = None
        self.player_pos: Tuple[int] = None
        self.player_force_arr: np.ndarray = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, len(self.tiles)))
        self.build_hist: list = []
        self.window = None
        self.rend_im: np.ndarray = None
        self.keys_to_acts = {
            pygame.K_LEFT: 0,
            pygame.K_RIGHT: 1,
            pygame.K_UP: 2,
            pygame.K_DOWN: 3,
        }
        self.screen = None
        # Here we assume that the player can place tiles at positions adjacent to the player. Could make this more 
        # flexible in the future.
        self._actions = [(tile_type.idx, i) for i in range(4) for tile_type in player_placeable_tiles]
        self._done_at_reward = done_at_reward
        self._map_queue = []
        self._map_id = 0

    def queue_maps(self, maps: Iterable[np.ndarray]):
        self._map_queue = maps

    def _update_player_pos(self, map_arr):
        self.player_pos = np.argwhere(map_arr[self.player_idx] == 1)
        assert self.player_pos.shape[0] == 1
        self.player_pos = tuple(self.player_pos[0])
        
    def _update_cooccurs(self, map_arr):
        for tile_type in self.tiles:
            if tile_type.cooccurs:
                for cooccur in tile_type.cooccurs:
                    map_arr[cooccur.idx, map_arr[tile_type.idx] == 1] = 1

    def gen_random_map(self):
        # Generate frequency-based tiles with certain probabilities.
        int_map = np.random.choice(len(self.tiles), size=(self.w, self.h), p=self.tile_probs)
        map_coords = np.argwhere(int_map != -1)
        # Overwrite frequency-based tiles with tile-types that require fixed numbers of instances.
        n_fixed = sum([tile.num for tile in self.tiles if tile.num is not None])
        fixed_coords = map_coords[np.random.choice(map_coords.shape[0], size=n_fixed, replace=False)]
        i = 0
        for tile in self.tiles:
            if tile.prob == 0 and tile.num is not None:
                coord_list = fixed_coords[i: i + tile.num]
                int_map[coord_list[:, 0], coord_list[:, 1]] = tile.idx
                i += tile.num
        map_arr = np.eye(len(self.tiles))[int_map]
        map_arr = rearrange(map_arr, "h w c -> c h w")
        self._update_player_pos(map_arr)
        # Activate parent/co-occuring tiles.
        for tile in self.tiles:
            coactive_tiles = tile.parents + tile.cooccurs
            if len(coactive_tiles) > 0:
                for cotile in coactive_tiles:
                    # Activate parent channels of any child tiles wherever the latter are active.
                    map_arr[cotile.idx, map_arr[tile.idx] == 1] = 1
        return map_arr

    def reset(self):
        self._n_step = 0
        self._reward = 0
        if len(self._map_queue) == 0:
            self.map = self.gen_random_map()
        else:
            self.map = self._map_queue[self._map_id]
            self._map_id = (self._map_id + 1) % len(self._map_queue)
        # Extra channel to denote passable/impassable for player.
        # self.static_builds = (np.random.random((self.w, self.h)) < self.static_prob).astype(np.uint8)
        # nonstatic_idxs = np.argwhere(self.static_builds != True)
        # self.curr_pos = tuple(nonstatic_idxs[np.random.choice(len(nonstatic_idxs))])
        # self.curr_pos_arr = np.zeros_like(self.map)
        # self.curr_pos_arr[tuple(self.curr_pos)] = 1
        # self.player_force_arr = np.zeros_like(self.map)
        # self.map[self.curr_pos] = 1
        # self.build_hist = [self.curr_pos]
        obs = self.get_obs()
        return obs

    def step(self, action):
        self.act(action)
        self.tick()
        reward = self._reward
        self._reward = self._baseline_rew
        self._n_step += 1
        return self.get_obs(), reward, self._done, {}

    def act(self, action):
        new_tile, adj_id = self._actions[action]
        # Do not place anything over the edge of the map. Should we wrap by default instead?
        pos = self.player_pos + self.adjs[adj_id]
        if np.any(pos < 0) or pos[0] >= self.w or pos[1] >= self.h:
            return
        # pos = tuple(np.clip(np.array(self.player_pos) + self.adjs[adj_id], (0, 0), (self.w - 1, self.h - 1)))
        self.map[new_tile, pos[0], pos[1]] = 1
        # if self.map[new_pos] == 1 or self.static_builds[new_pos] == 1:
        #     reward = -1
        #     done = True
        # else:
        #     self.map[new_pos] = 1
        #     self.build_hist.append(new_pos)
        #     self.player_pos = new_pos
        #     self.curr_pos_arr = np.zeros_like(self.map)
        #     self.curr_pos_arr[tuple(self.player_pos)] = 1
        #     done = False
        #     reward = 1
        # nb_idxs = np.array(self.player_pos) + self.adjs + 1
        # neighb_map = np.pad(self.map, 1, mode='constant', constant_values=1)[nb_idxs[:, 0], nb_idxs[:, 1]]
        # neighb_static = np.pad(self.static_builds, 1, mode='constant', constant_values=1)[nb_idxs[:, 0], nb_idxs[:, 1]]
        # Terminate if all neighboring tiles already have path or do not belong to graph.

    def get_obs(self):
        obs = rearrange(self.map, 'b h w -> h w b')
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        tile_size = self.tile_size
        if mode == 'human':
            if self.window is None:
                self.window = cv2.namedWindow('Generated Environment', cv2.WINDOW_NORMAL)
            # self.rend_im = np.zeros_like(self.int_map)
        # Create an int map where the last tiles in `self.tiles` take priority.
        int_map = np.zeros(self.map[0].shape, dtype=np.uint8)
        for tile in self.tiles:
            if tile.color is not None:
                int_map[self.map[tile.idx] == 1] = tile.idx
        self.rend_im = self.tile_colors[int_map]
        # self.rend_im = repeat(self.rend_im, 'h w -> h w 3')
        self.rend_im = repeat(self.rend_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
        # b0 = self.build_hist[0]
        # for b1 in self.build_hist[1:]:
            # x0, x1 = sorted([b0[0], b1[0]])
            # y0, y1 = sorted([b0[1], b1[1]])
            # self.rend_im[
                # x0 * tile_size + tile_size // 2 - pw: x1 * tile_size + tile_size // 2 + pw,
                # y0 * tile_size + tile_size // 2 - pw: y1 * tile_size + tile_size // 2 + pw] = [0, 1, 0]
            # b0 = b1
        # self.rend_im *= 255
        if mode == "human":
            # pass
            cv2.imshow('Generated Environment', self.rend_im)
            cv2.waitKey(1)
        if mode == "pygame":
            if self.screen is None:
                pygame.init()
                # Set up the drawing window
                self.screen = pygame.display.set_mode([self.h*GenEnv.tile_size, self.w*GenEnv.tile_size])
            pygame_render_im(self.screen, self.rend_im)
        else:
            return self.rend_im

    def tick(self):
        self.map, self._reward, self._done = apply_rules(self.map, self.rules)
        self._update_player_pos(self.map)
        self._update_cooccurs(self.map)
        if self._done_at_reward is not None:
            self._done = self._done or self._reward == self._done_at_reward
        self._done = self._done or self._n_step >= self.max_episode_steps

    def tick_human(self):
        done = False
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key in self.keys_to_acts:
                    action = self.keys_to_acts[event.key]
                    obs, rew, done, info = self.step(action)

                    self.render(mode='pygame')
                    # print('\nreward:', rew)
                elif event.key == pygame.K_x:
                    done = True
            if done:
                self.reset()
                done = False
                self.render(mode='pygame')


def apply_rules(map: np.ndarray, rules: List[Rule]):
    """Apply rules to a one-hot encoded map state, to return a mutated map.

    Args:
        map (np.ndarray): A one-hot encoded map representing the game state.
        rules (List[Rule]): A list of rules for mutating the onehot-encoded map.
    """
    # print(map)
    done = False
    reward = 0
    h, w = map.shape[1:]
    for rule in rules:
        for subrule in rule.subrules:
            inp, out = subrule
            for x in range(h):
                # print(f'x: {x}')
                for y in range(w):
                    # print(f'y: {y}')
                    match = True
                    for subp in inp:
                        if subp.shape[0] + x > h or subp.shape[1] + y > w:
                            match = False
                            break
                        if not match:
                            break
                        for i in range(subp.shape[0]):
                            if not match:
                                break
                            for j in range(subp.shape[1]):
                                if subp[i, j] == -1:
                                    continue
                                if map[subp[i, j], x + i, y + j] != 1:
                                    match = False
                                    break
                    if match:
                        # print(f'matched rule {rule.name} at {x}, {y}')
                        # print(f'rule has input \n{inp}\n and output \n{out}')
                        reward += rule.reward
                        done = done or rule.done
                        for k, subp in enumerate(out):
                            for i in range(subp.shape[0]):
                                for j in range(subp.shape[1]):
                                    # Remove the corresponding tile in the input pattern if one exists.
                                    if inp[k, i, j] != -1:
                                        map[inp[k, i, j], x + i, y + j] = 0
                                    if subp[i, j] == -1:
                                        continue
                                    map[subp[i, j], x + i, y + j] = 1
    return map, reward, done


def pygame_render_im(screen, img):
    surf = pygame.surfarray.make_surface(img)
    # Fill the background with white
    # screen.fill((255, 255, 255))
    screen.blit(surf, (0, 0))
    # Flip the display
    pygame.display.flip()
