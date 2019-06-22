#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.generators import rail_from_GridTransitionMap_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import DummyPredictorForRailEnv, ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.utils.rendertools import RenderTool
from make_simple_rail import make_simple_rail

"""Test predictions for `flatland` package."""


def test_dummy_predictor(rendering=False):
    rail, rail_map = make_simple_rail()

    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_GridTransitionMap_generator(rail),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=DummyPredictorForRailEnv(max_depth=10)),
                  )
    env.reset()

    # set initial position and direction for testing...
    env.agents[0].position = (5, 6)
    env.agents[0].direction = 0
    env.agents[0].target = (3, 0)

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.renderEnv(show=True, show_observations=False)
        input("Continue?")

    # test assertions
    predictions = env.obs_builder.predictor.get(None)
    positions = np.array(list(map(lambda prediction: [*prediction[1:3]], predictions[0])))
    directions = np.array(list(map(lambda prediction: [prediction[3]], predictions[0])))
    time_offsets = np.array(list(map(lambda prediction: [prediction[0]], predictions[0])))
    actions = np.array(list(map(lambda prediction: [prediction[4]], predictions[0])))

    # compare against expected values
    expected_positions = np.array([[5., 6.],
                                   [4., 6.],
                                   [3., 6.],
                                   [3., 5.],
                                   [3., 4.],
                                   [3., 3.],
                                   [3., 2.],
                                   [3., 1.],
                                   # at target (3,0): stay in this position from here on
                                   [3., 0.],
                                   [3., 0.],
                                   [3., 0.],
                                   ])
    expected_directions = np.array([[0.],
                                    [0.],
                                    [0.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    # at target (3,0): stay in this position from here on
                                    [3.],
                                    [3.],
                                    [3.]
                                    ])
    expected_time_offsets = np.array([[0.],
                                      [1.],
                                      [2.],
                                      [3.],
                                      [4.],
                                      [5.],
                                      [6.],
                                      [7.],
                                      [8.],
                                      [9.],
                                      [10.],
                                      ])
    expected_actions = np.array([[0.],
                                 [2.],
                                 [2.],
                                 [1.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 # reaching target by straight
                                 [2.],
                                 # at target: stopped moving
                                 [4.],
                                 [4.],
                                 ])
    assert np.array_equal(positions, expected_positions)
    assert np.array_equal(directions, expected_directions)
    assert np.array_equal(time_offsets, expected_time_offsets)
    assert np.array_equal(actions, expected_actions)


def test_shortest_path_predictor(rendering=False):
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_GridTransitionMap_generator(rail),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.reset()

    agent = env.agents[0]
    agent.position = (5, 6)  # south dead-end
    agent.direction = 0  # north
    agent.target = (3, 9)  # east dead-end

    agent.moving = True

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.renderEnv(show=True, show_observations=False)
        input("Continue?")

    agent = env.agents[0]
    assert agent.position == (5, 6)
    assert agent.direction == 0
    assert agent.target == (3, 9)
    assert agent.moving

    env.obs_builder._compute_distance_map()

    distance_map = env.obs_builder.distance_map
    assert distance_map[agent.handle, agent.position[0], agent.position[
        1], agent.direction] == 5.0, "found {} instead of {}".format(
        distance_map[agent.handle, agent.position[0], agent.position[1], agent.direction], 5.0)

    # test assertions
    env.obs_builder.get_many()
    predictions = env.obs_builder.predictions
    positions = np.array(list(map(lambda prediction: [*prediction[1:3]], predictions[0])))
    directions = np.array(list(map(lambda prediction: [prediction[3]], predictions[0])))
    time_offsets = np.array(list(map(lambda prediction: [prediction[0]], predictions[0])))
    actions = np.array(list(map(lambda prediction: [prediction[4]], predictions[0])))

    expected_positions = [
        [5, 6],
        [4, 6],
        [3, 6],
        [3, 7],
        [3, 8],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
    ]
    expected_directions = [
        [Grid4TransitionsEnum.NORTH],  # next is [5,6] heading north
        [Grid4TransitionsEnum.NORTH],  # next is [4,6] heading north
        [Grid4TransitionsEnum.NORTH],  # next is [3,6] heading north
        [Grid4TransitionsEnum.EAST],  # next is [3,7] heading east
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
    ]

    expected_time_offsets = np.array([
        [0.],
        [1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.],
        [7.],
        [8.],
        [9.],
        [10.],
        [11.],
        [12.],
        [13.],
        [14.],
        [15.],
        [16.],
        [17.],
        [18.],
        [19.],
        [20.],
    ])

    expected_actions = np.array([
        [RailEnvActions.DO_NOTHING],  # next [5,6]
        [RailEnvActions.MOVE_FORWARD],  # next [4,6]
        [RailEnvActions.MOVE_FORWARD],  # next [3,6]
        [RailEnvActions.MOVE_RIGHT],  # next [3,7]
        [RailEnvActions.MOVE_FORWARD],  # next [3,8]
        [RailEnvActions.MOVE_FORWARD],  # next [3,9]
        [RailEnvActions.STOP_MOVING],  # at [3,9] == target
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
        [RailEnvActions.STOP_MOVING],
    ])

    assert np.array_equal(positions, expected_positions), \
        "positions {}, expected {}".format(positions, expected_positions)
    assert np.array_equal(directions, expected_directions), \
        "directions {}, expected {}".format(directions, expected_directions)
    assert np.array_equal(time_offsets, expected_time_offsets), \
        "time_offsets {}, expected {}".format(time_offsets, expected_time_offsets)
    assert np.array_equal(actions, expected_actions), \
        "actions {}, expected {}".format(actions, expected_actions)
