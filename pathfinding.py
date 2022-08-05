import networkx as nx
import numpy as np


def id_to_xy(idx, width):
    return idx // width, idx % width

def xy_to_id(x, y, width):
    return x * width + y

def shortest_path(multihot, traversable_tiles, src_tile, trg_tile, out_tile):
    traversable_tile_idxs = [t.idx for t in traversable_tiles]
    src, trg = None, None
    graph = nx.Graph()
    width, height = multihot.shape[1:]
    size = width * height
    graph.add_nodes_from(range(size))
    edges = []
    for u in range(size):
        ux, uy = id_to_xy(u, width)
        if np.all(multihot[traversable_tile_idxs, ux, uy] != 1):
            continue
        if multihot[src_tile.idx, ux, uy] == 1:
            src = u
        if multihot[trg_tile.idx, ux, uy] == 1:
            trg = u
        neighbs_xy = [(ux - 1, uy), (ux, uy-1), (ux+1, uy), (ux, uy+1)]
        # adj_feats = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        neighbs = [xy_to_id(x, y, width) for x, y in neighbs_xy]
        for v, (vx, vy) in zip(neighbs, neighbs_xy):
            if not 0 <= v < size or vx < 0 or vx >= width or vy < 0 or vy >= height or \
                    np.all(multihot[traversable_tile_idxs, vx, vy] != 1):
                continue
            graph.add_edge(u, v)
            edges.append((u, v))
        edges.append((u, u))

    path = nx.shortest_path(graph, src, trg)
    path = np.array([id_to_xy(idx, width) for idx in path])
    multihot[out_tile.idx, path[1:, 0], path[1:, 1]] = 1

    return multihot


def draw_shortest_path(env, traversable_tiles, src_tile, trg_tile, out_tile):
    env.map = shortest_path(env.map, traversable_tiles, src_tile, trg_tile, out_tile)

