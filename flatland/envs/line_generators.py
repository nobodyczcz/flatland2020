"""Line generators (railway undertaking, "EVU")."""
import warnings
from typing import Tuple, List, Callable, Mapping, Optional, Any

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.timetable_utils import Line
from flatland.envs import persistence

AgentPosition = Tuple[int, int]
LineGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int]], Line]


def speed_initialization_helper(nb_agents: int, speed_ratio_map: Mapping[float, float] = None,
                                seed: int = None, np_random: RandomState = None) -> List[float]:
    """
    Parameters
    ----------
    nb_agents : int
        The number of agents to generate a speed for
    speed_ratio_map : Mapping[float,float]
        A map of speeds mappint to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
    List[float]
        A list of size nb_agents of speeds with the corresponding probabilistic ratios.
    """
    if speed_ratio_map is None:
        return [1.0] * nb_agents

    nb_classes = len(speed_ratio_map.keys())
    speed_ratio_map_as_list: List[Tuple[float, float]] = list(speed_ratio_map.items())
    speed_ratios = list(map(lambda t: t[1], speed_ratio_map_as_list))
    speeds = list(map(lambda t: t[0], speed_ratio_map_as_list))
    return list(map(lambda index: speeds[index], np_random.choice(nb_classes, nb_agents, p=speed_ratios)))


class BaseLineGen(object):
    def __init__(self, speed_ratio_map: Mapping[float, float] = None, seed: int = 1):
        self.speed_ratio_map = speed_ratio_map
        self.seed = seed

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any=None, num_resets: int = 0,
        np_random: RandomState = None) -> Line:
        pass

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def sparse_line_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> LineGenerator:
    return SparseLineGen(speed_ratio_map, seed)


class SparseLineGen(BaseLineGen):
    """

    This is the line generator which is used for Round 2 of the Flatland challenge. It produces lines
    to railway networks provided by sparse_rail_generator.
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    """

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: dict, num_resets: int,
                  np_random: RandomState) -> Line:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the line
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """

        _runtime_seed = self.seed + num_resets

        train_stations = hints['train_stations']
        city_positions = hints['city_positions']
        city_orientation = hints['city_orientations']
        max_num_agents = hints['num_agents']
        city_orientations = hints['city_orientations']
        if num_agents > max_num_agents:
            num_agents = max_num_agents
            warnings.warn("Too many agents! Changes number of agents.")
        # Place agents and targets within available train stations
        agents_position = []
        agents_target = []
        agents_direction = []


        city1, city2 = None, None
        city1_num_stations, city2_num_stations = None, None
        city1_possible_orientations, city2_possible_orientations = None, None


        for agent_idx in range(num_agents):

            if (agent_idx % 2 == 0):
                # Setlect 2 cities, find their num_stations and possible orientations
                city_idx = np_random.choice(len(city_positions), 2, replace=False)
                city1 = city_idx[0]
                city2 = city_idx[1]
                city1_num_stations = len(train_stations[city1])
                city2_num_stations = len(train_stations[city2])
                city1_possible_orientations = [city_orientation[city1],
                                                (city_orientation[city1] + 2) % 4]
                city2_possible_orientations = [city_orientation[city2],
                                                (city_orientation[city2] + 2) % 4]

                # Agent 1 : city1 > city2, Agent 2: city2 > city1
                agent_start_idx = ((2 * np_random.randint(0, 10))) % city1_num_stations
                agent_target_idx = ((2 * np_random.randint(0, 10)) + 1) % city2_num_stations

                agent_start = train_stations[city1][agent_start_idx]
                agent_target = train_stations[city2][agent_target_idx]

                agent_orientation = np_random.choice(city1_possible_orientations)


            else:
                agent_start_idx = ((2 * np_random.randint(0, 10))) % city2_num_stations
                agent_target_idx = ((2 * np_random.randint(0, 10)) + 1) % city1_num_stations
                
                agent_start = train_stations[city2][agent_start_idx]
                agent_target = train_stations[city1][agent_target_idx]
                            
                agent_orientation = np_random.choice(city2_possible_orientations)

            
            # agent1 details
            agents_position.append((agent_start[0][0], agent_start[0][1]))
            agents_target.append((agent_target[0][0], agent_target[0][1]))
            agents_direction.append(agent_orientation)


        if self.speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, self.speed_ratio_map, seed=_runtime_seed, np_random=np_random)
        else:
            speeds = [1.0] * len(agents_position)

        # We add multiply factors to the max number of time steps to simplify task in Flatland challenge.
        # These factors might change in the future.
        timedelay_factor = 4
        alpha = 2
        max_episode_steps = int(
            timedelay_factor * alpha * (rail.width + rail.height + num_agents / len(city_positions)))

        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None)


def line_from_file(filename, load_from_package=None) -> LineGenerator:
    """
    Utility to load pickle file

    Parameters
    ----------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Line:

        env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)

        max_episode_steps = env_dict.get("max_episode_steps", 0)
        if (max_episode_steps==0):
            print("This env file has no max_episode_steps (deprecated) - setting to 100")
            max_episode_steps = 100
            
        agents = env_dict["agents"]
        
        # setup with loaded data
        agents_position = [a.initial_position for a in agents]

        # this logic is wrong - we should really load the initial_direction as the direction.
        #agents_direction = [a.direction for a in agents]
        agents_direction = [a.initial_direction for a in agents]
        agents_target = [a.target for a in agents]
        agents_speed = [a.speed_data['speed'] for a in agents]

        # Malfunctions from here are not used.  They have their own generator.
        #agents_malfunction = [a.malfunction_data['malfunction_rate'] for a in agents]

        return Line(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=agents_speed, 
                        agent_malfunction_rates=None)

    return generator
