import cv2
from pdb import set_trace as TT
from typing import Iterable, List, Tuple

from einops import rearrange, repeat
import gym
from gym import spaces
import numpy as np
import pygame


colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'brown': (165, 42, 42),
    'error': (255, 192, 203),  # pink
}
colors = {k: np.array(v) for k, v in colors.items()}


class TileType():
    def __init__(self, name: str, prob: float, color: Tuple[int], passable: bool = False, num: int = None,
            parents: List = []):
        """Initialize a new tile type.

        Args:
            name (str): Name of the tile
            prob (float): Probability of the tile spawning during random map generation, if applicable. If 0, then `num`
                must be specified to spawn an exact number of this tile.
            color (Tuple[int]): The color of the tile during rendering.
            passable (bool, optional): Whether this tile is passable to player. Defaults to False.
            num (int, optional): Exactly how many instances of this tile should spawn during random map generation. 
                Defaults to None.
            parents (List[TileType]): A list of tile types from which this tile can be considered a sub-type. Useful 
                for rules that consider parent types.
        """
        self.name = name
        self.prob = prob
        self.color = color
        self.passable = passable
        self.num = num
        self.idx = None  # This needs to be set externally, given some consistent ordering of the tiles.
        self.parents = parents

    def get_idx(self):
        if self is None:
            return -1
        return self.idx

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"TileType: {self.name}"


class Rule():
    """Rules match input to output patterns. When the environment ticks, the input patters are identified and replaced 
    by the output patterns. Each rule may consist of multiple sub-rules, any one of which can be applied. Each rule may
    consist of multiple subpatterns, which may all match in order for the subrule to be applied.
    Input and output are 2D patches of tiles.
    """
    def __init__(self, name: str, in_out: Iterable[TileType], rotate: bool = True):
        """Process the main subrule `in_out`, potentially rotating it to produce a set of subrules. Also convert 
        subrules from containing TileType objects to their indices for fast matching during simulation.

        Args:
            name (str): The rule's name. For human-readable purposes.
            in_out (Iterable[TileType]): A sub-rule with shape (2, n_subpatterns, h, w)
            rotate (bool): Whether subpatterns .
        """
        self.name = name
        # List of subrules resulting from rule (i.e. if applying rotation).
        in_out_int = np.vectorize(TileType.get_idx)(in_out)
        subrules = [in_out_int]
        if rotate:
            subrules += [np.rot90(in_out_int, k=1, axes=(2, 3)), np.rot90(in_out_int, k=2, axes=(2,3)), 
                np.rot90(in_out_int, k=3, axes=(2,3))]
        self.subrules = subrules


class GenEnv(gym.Env):
    adjs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    tile_size = 16
    
    def __init__(self, width: int, height: int, tiles: List[TileType], rules: List[Rule], 
            player_placeable_tiles: List[TileType]):
        """_summary_

        Args:
            width (int): The width of the 2D game map.
            height (int): The height of the 2D game map.
            tiles (list): A list of TileType objects. Must include a tile-type with name `player`.
            rules (list): A list of Rule objects, between TileTypes.
        """
        self._is_done = False
        self.w, self.h = width, height
        self.tiles = tiles
        [setattr(tile, 'idx', i) for i, tile in enumerate(tiles)]
        tiles_by_name = {t.name: t for t in tiles}
        # Assuming here that we always have player and floor...
        self.player_idx = tiles_by_name['player'].idx
        self.floor_idx = tiles_by_name['floor'].idx
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

    def _update_player_pos(self, map_arr):
        self.player_pos = np.argwhere(map_arr[self.player_idx] == 1)
        assert self.player_pos.shape[0] == 1
        self.player_pos = tuple(self.player_pos[0])

    def gen_random_map(self):
        int_map = np.random.choice(len(self.tiles), size=(self.w, self.h), p=self.tile_probs)
        map_coords = np.argwhere(int_map != -1)
        for tile in self.tiles:
            if tile.prob == 0 and tile.num is not None:
                coord_list = map_coords[np.random.choice(map_coords.shape[0], size=tile.num, replace=False)]
                int_map[coord_list[:, 0], coord_list[:, 1]] = tile.idx
        map_arr = np.eye(len(self.tiles))[int_map]
        map_arr = rearrange(map_arr, "h w c -> c h w")
        self._update_player_pos(map_arr)
        # Hard-code that there is floor, and nothing else, is underneath the player to start. Make this more general later?
        map_arr[:, self.player_pos[0], self.player_pos[1]] = 0
        map_arr[self.floor_idx, self.player_pos[0], self.player_pos[1]] = 1
        map_arr[self.player_idx, self.player_pos[0], self.player_pos[1]] = 1
        for tile in self.tiles:
            if len(tile.parents) > 0:
                for parent in tile.parents:
                    # Activate parent channels of any child tiles wherever the latter are active.
                    map_arr[parent.idx, map_arr[tile.idx] == 1] = 1
        return map_arr

    def reset(self):
        self._reward = 0
        self.map = self.gen_random_map()
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
        self._reward = 0
        return self.get_obs(), reward, self._is_done, {}

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
        if self.window is None:
            if mode == 'human':
                self.window = cv2.namedWindow('Generated Environment', cv2.WINDOW_NORMAL)
            # self.rend_im = np.zeros_like(self.int_map)
        # Create an int map where the last tiles in `self.tiles` take priority.
        int_map = np.zeros(self.map[0].shape, dtype=np.uint8)
        for tile in tiles:
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
            cv2.imshow('Generated Environment', self.rend_im)
            cv2.waitKey(1)
        elif mode == "pygame":
            if self.screen is None:
                pygame.init()
                # Set up the drawing window
                self.screen = pygame.display.set_mode([h*GenEnv.tile_size, w*GenEnv.tile_size])
            pygame_render_im(self.screen, self.rend_im)
        else:
            return self.rend_im

    def tick(self):
        apply_rules(self.map, self.rules)
        self._update_player_pos(self.map)

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
                    print('\nreward:', rew)
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
    print(map)
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
                        print(f'matched rule {rule.name} at {x}, {y}')
                        print(f'rule has input \n{inp}\n and output \n{out}')
                        for k, subp in enumerate(out):
                            for i in range(subp.shape[0]):
                                for j in range(subp.shape[1]):
                                    if subp[i, j] == -1:
                                        # Remove the corresponding tile in the input pattern if appropriate.
                                        if inp[k, i, j] != -1:
                                            map[inp[k, i, j], x + i, y + j] = 0
                                        continue
                                    map[subp[i, j], x + i, y + j] = 1
    return map


def pygame_render_im(screen, img):
    surf = pygame.surfarray.make_surface(img)
    # Fill the background with white
    # screen.fill((255, 255, 255))
    screen.blit(surf, (0, 0))
    # Flip the display
    pygame.display.flip()


if __name__ == "__main__":
    force = TileType(name='force', prob=0, color=None)
    passable = TileType(name='passable', prob=0, color=None)
    floor = TileType('floor', prob=0.8, color=colors['white'], parents=[passable])
    player = TileType('player', prob=0, color=colors['blue'], num=1, parents=[passable])
    wall = TileType('wall', prob=0.2, color=colors['black'])

    tiles = [force, passable, floor, player, wall]
    [setattr(tile, 'idx', i) for i, tile in enumerate(tiles)]

    player_move = Rule(
        'player_move', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [[[player, passable]],  # Player next to a passable tile.
            [[None, force]], # A force is active on said passable tile.
              ]  
            ,
          # Both changes are applied to the relevant channels, given by the respective input subpatterns.
            [[[None, player]],  # Player moves to target. No change at source.
            [[None, None]],  # Force is removed from target tile.
            ],
        ]
        ),
        rotate=True,)
    rules = [player_move]

    h = 10
    w = 10
    env = GenEnv(height=h, width=w, tiles=tiles, rules=rules, player_placeable_tiles=[force])
    env.reset()
    env.render(mode='pygame')
    done = False

    running = True

    while running:
        env.tick_human()

    # Done! Time to quit.
    pygame.quit()