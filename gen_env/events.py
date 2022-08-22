from functools import partial
from typing import Iterable
import networkx as nx
import numpy as np

# Dummy hardcoded. Use generic global variable objects instead.
def on_start(env):
    return env.n_step == 0


def activate_rules(env, rules):
    env.rules = rules

class Event():
    def __init__(self, name, init_cond = lambda: True, tick_func = lambda: None, done_cond=lambda: True, children=[]):
        self.name = name
        self.init_cond = init_cond
        self._tick_func = tick_func
        self.children = children
        self.done_cond = done_cond

    def tick_func(self, env):
        # print(f'{self.name} tick')
        self._tick_func(env)


class RuleNode():
    def __init__(self, name, rules, children):
        super().__init__(name=name, init_cond=lambda: True, 
            tick_func=lambda: partial(activate_rules, rules), 
            done_cond=lambda env: not env._has_applied_rule, 
            children=children)
        self.name = name
        self.rules = rules


def get_rule_node_sequence(rule_sets):
    rule_nodes = []
    child_nodes = []
    for rules in rule_sets[::-1]:
        r_node = RuleNode(name=rules.name, rules=rules, children=child_nodes)
        child_nodes = [r_node]
        rule_nodes.append(r_node)
    return rule_nodes


class EventGraph():
    def __init__(self, events: Iterable[Event]):
        self.events = events
        self.frontier_events = set(events)

    def tick(self, env):
        to_pop = []
        # print(f'frontier events: {[e.name for e in self.frontier_events]}')
        for event in self.frontier_events:
            if event.init_cond():
                event.tick_func(env)
                if event.done_cond():
                    print(f'{event.name} done')
                    to_pop.append(event)
        for event in to_pop:
            self.frontier_events.remove(event)
            for child in event.children:
                if child.init_cond():
                    child.tick_func(env)
                if not child.done_cond():
                    [self.frontier_events.add(child) for child in event.children]

    def reset(self):
        self.frontier_events = set(self.events)

        
class GlobalVariables():
    def __init__(self, glob_vars):
        self.n_step = 0
        self._glob_vars = glob_vars
        [setattr(self, k, v) for k, v in glob_vars.items()]
        
    def reset(self):
        [setattr(self, k, v) for k, v in self._glob_vars.items()]

    def tick(self):
        self.n_step += 1