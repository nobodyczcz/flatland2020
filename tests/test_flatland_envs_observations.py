#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.generators import rail_from_GridTransitionMap_generator
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import DummyPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from make_simple_rail import make_simple_rail

"""Tests for `flatland` package."""


def test_tree_obs(rendering=True):
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_GridTransitionMap_generator(rail),
                  number_of_agents=4,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=DummyPredictorForRailEnv(max_depth=10)))

    env.reset()

    # set initial position and direction for testing...
    env.agents[0].position = (3, 0)
    env.agents[0].direction = Grid4TransitionsEnum.EAST
    env.agents[0].target = (3, 9)

    # opposing, conflict
    env.agents[1].position = (3, 1)
    env.agents[1].direction = Grid4TransitionsEnum.EAST
    env.agents[1].target = (3, 0)

    # opposing, conflict???
    env.agents[2].position = (3, 7)
    env.agents[2].direction = Grid4TransitionsEnum.EAST
    env.agents[2].target = (3, 0)

    # same direction
    env.agents[3].position = (3, 1)
    env.agents[3].direction = Grid4TransitionsEnum.EAST
    env.agents[3].target = (3, 9)

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.renderEnv(show=True, show_observations=False)
        input("Continue?")


def test_global_obs():
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_GridTransitionMap_generator(rail),
                  number_of_agents=1,
                  obs_builder_object=GlobalObsForRailEnv())

    global_obs = env.reset()

    assert (global_obs[0][0].shape == rail_map.shape + (16,))

    rail_map_recons = np.zeros_like(rail_map)
    for i in range(global_obs[0][0].shape[0]):
        for j in range(global_obs[0][0].shape[1]):
            rail_map_recons[i, j] = int(
                ''.join(global_obs[0][0][i, j].astype(int).astype(str)), 2)

    assert (rail_map_recons.all() == rail_map.all())

    # If this assertion is wrong, it means that the observation returned
    # places the agent on an empty cell
    assert (np.sum(rail_map * global_obs[0][1][:, :, :4].sum(2)) > 0)
