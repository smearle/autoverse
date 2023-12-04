import cv2
import numpy as np


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
    rot_mat = cv2.getRotationMatrix2D((tile_size / 2, tile_size / 2), (float(rot) - 1) * 90, 1)
    triangle = cv2.transform(triangle.reshape(1, -1, 2), rot_mat).reshape(-1, 2)

    # Draw the triangle
    triangle = triangle.astype(np.int32)
    # Convert color so cv2 doesn't complain
    color = tuple([int(c) for c in color])

    cv2.fillConvexPoly(im, triangle + np.array(pos) * tile_size, color)

    return im
