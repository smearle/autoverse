import os
import cv2
import imageio
import numpy as np
from gen_env.configs.config import Config


def validate_config(cfg: Config):
    env_exp_name = (f"{cfg.game}_{'mutRule_' if cfg.mutate_rules else ''}{'fixMap_' if cfg.fix_map else ''}" + 
        f"exp-{cfg.env_exp_id}")

    cfg._log_dir_common = os.path.join(cfg.workspace, env_exp_name)

    player_exp_name = (f"player{'_hideRule' if cfg.hide_rules else ''}")

    cfg._log_dir_player_common = os.path.join(cfg._log_dir_common, player_exp_name)
    cfg._log_dir_rl = os.path.join(cfg._log_dir_player_common, cfg.runs_dir_rl)
    cfg._log_dir_il = os.path.join(cfg._log_dir_player_common, cfg.runs_dir_il)
    # cfg.log_dir_evo = os.path.join(cfg.workspace, cfg.runs_dir_evo, f"exp-{cfg.exp_id}")
    cfg._log_dir_evo = os.path.join(cfg._log_dir_common, cfg.runs_dir_evo)


def save_video(frames, video_path, fps=10):
    """Save a list of frames to a video file.
    Args:
        frames (list): list of frames to save
        video_path (str): path to save the video
        fps (int): frame rate of the video
    """
    imageio.mimwrite(video_path, frames, fps=25, quality=8, macro_block_size=1)

def draw_triangle(im, pos, rot, color, tile_size):
    """Draw a triangle on a tile grid.
    Args:
        im (np.ndarray): the image to draw on
        pos (tuple): the position of the triangle
        rot (int): the rotation of the triangle
        color (tuple): the color of the triangle
        tile_size (int): the size of a tile
    Returns:
        np.ndarray: the image with the triangle drawn"""
    pos = pos[1], pos[0] + 2

    # Define the triangle
    triangle = np.array([
        [0, 0],
        [tile_size, 0],
        [tile_size / 2, tile_size],
    ])

    # Rotate the triangle
    rot_mat = cv2.getRotationMatrix2D((tile_size / 2, tile_size / 2), (rot - 1) * 90, 1)
    triangle = cv2.transform(triangle.reshape(1, -1, 2), rot_mat).reshape(-1, 2)

    # Draw the triangle
    triangle = triangle.astype(np.int32)
    # Convert color so cv2 doesn't complain
    color = tuple([int(c) for c in color])

    cv2.fillConvexPoly(im, triangle + np.array(pos) * tile_size, color)

    return im