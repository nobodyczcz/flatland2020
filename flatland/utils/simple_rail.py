from typing import Tuple

import numpy as np

from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap


def make_simple_rail() -> Tuple[GridTransitionMap, np.array]:
    # We instantiate a very simple rail network on a 7x10 grid:
    # Note that that cells have invalid RailEnvTransitions!
    #        |
    #        |
    #        |
    # _ _ _ _\ _ _  _  _ _ _
    #                /
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_east_west_north = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_east_west_south = transitions.rotate_transition(simple_switch_north_left, 270)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [simple_switch_east_west_north] +
         [horizontal_straight] * 2 + [simple_switch_east_west_south] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    return rail, rail_map


def make_simple_rail2() -> Tuple[GridTransitionMap, np.array]:
    # We instantiate a very simple rail network on a 7x10 grid:
    #        |
    #        |
    #        |
    # _ _ _ _\ _ _  _  _ _ _
    #               \
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_right = cells[10]
    simple_switch_east_west_north = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_east_south = transitions.rotate_transition(simple_switch_north_right, 90)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [simple_switch_east_west_north] +
         [horizontal_straight] * 2 + [simple_switch_west_east_south] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    return rail, rail_map


def make_simple_rail_unconnected() -> Tuple[GridTransitionMap, np.array]:
    # We instantiate a very simple rail network on a 7x10 grid:
    # Note that that cells have invalid RailEnvTransitions!
    #        |
    #        |
    #        |
    # _ _ _  _ _ _  _  _ _ _
    #                /
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    # simple_switch_east_west_north = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_east_west_south = transitions.rotate_transition(simple_switch_north_left, 270)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] +
        [[empty] * 3 + [dead_end_from_north] + [empty] * 6] +
        [[dead_end_from_east] + [horizontal_straight] * 5 + [simple_switch_east_west_south] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    return rail, rail_map


def make_invalid_simple_rail() -> Tuple[GridTransitionMap, np.array]:
    # We instantiate a very simple rail network on a 7x10 grid:
    #        |
    #        |
    #        |
    # _ _ _ /_\ _ _  _  _ _ _
    #               \ /
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list

    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    double_switch_south_horizontal_straight = horizontal_straight + cells[6]
    double_switch_north_horizontal_straight = transitions.rotate_transition(
        double_switch_south_horizontal_straight, 180)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [double_switch_north_horizontal_straight] +
         [horizontal_straight] * 2 + [double_switch_south_horizontal_straight] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    return rail, rail_map
