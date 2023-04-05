from functools import partial
from math import inf
from pdb import set_trace as TT
# from turtle import back

import numpy as np

from gen_env.envs.play_env import PlayEnv
from gen_env.events import Event, activate_rules, on_start
from gen_env.rules import MJRule, MJRuleNode, Rule, RuleSet
from gen_env.tiles import TileNot, TilePlacement, TileSet, TileType
from gen_env.variables import Variable


def make_env(height, width):
    # a = MJRule("RBB=WWR")
    # b = MJRule("RBW=GWP")
    # c = MJRule("PWG=PBU")
    # d = MJRule("UWW=BBU")
    # e = MJRule("UWP=BBR")

    rules = MJRuleNode([
        "RBB=WWR",
        ])
    tiles = rules.tiles

    env = PlayEnv(height, width, tiles=tiles, rules=rules, player_placeable_tiles=[])
    return env