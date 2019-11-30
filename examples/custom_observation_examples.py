import random
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
import getopt
import sys
import time
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.misc import str2bool
from flatland.utils.rendertools import RenderTool
from typing import Optional, List, Dict
from flatland.core.env import Environment
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.utils.ordered_set import OrderedSet
import PIL
from IPython.core.display import display, clear_output
from IPython.display import HTML
from matplotlib import animation
import matplotlib.pyplot as plt


# Test if we are running in a notebook
# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
in_notebook = False
try:
    # get_ipython() is only defined inside ipython; we have to ignore it in flake8 with noqa
    get_ipython()  # noqa F821
    in_notebook = True
except(NameError):
    in_notebook = False 

random.seed(100)
np.random.seed(100)


class SimpleObs(ObservationBuilder):
    """
    Simplest observation builder. The object returns observation vectors with 5 identical components,
    all equal to the ID of the respective agent.
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        return

    def get(self, handle: int = 0) -> np.ndarray:
        observation = handle * np.ones((5,))
        return observation


# Demonstrate the simple custom observation
def show_simple_obs():
    env = RailEnv(width=7, height=7, rail_generator=random_rail_generator(), number_of_agents=3,
                  obs_builder_object=SimpleObs())
    env.reset()

    # Print the observation vector for each agents
    obs, all_rewards, done, _ = env.step({0: 0})
    for i in range(env.get_num_agents()):
        print("Agent ", i, "'s observation: ", obs[i])


if __name__ == "__main__" or in_notebook:
    show_simple_obs()


class SingleAgentNavigationObs(ObservationBuilder):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


def show_single_agent_main(args):
    if not in_notebook:
        try:
            opts, args = getopt.getopt(args, "", ["sleep-for-animation=", ""])
        except getopt.GetoptError as err:
            print(str(err))  # will print something like "option -a not recognized"
            sys.exit(2)
        sleep_for_animation = True
        for o, a in opts:
            if o in ("--sleep-for-animation"):
                sleep_for_animation = str2bool(a)
            else:
                assert False, "unhandled option"
    else:
        sleep_for_animation = True

    env = RailEnv(width=7, height=7,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=5, max_dist=99999,
                                                        seed=1), schedule_generator=complex_schedule_generator(),
                  number_of_agents=1, obs_builder_object=SingleAgentNavigationObs())

    obs, info = env.reset()
    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env(show=not in_notebook, frames=True, show_observations=True)
    for step in range(100):
        action = np.argmax(obs[0]) + 1
        obs, all_rewards, done, _ = env.step({0: action})
        print("Rewards: ", all_rewards, "  [done=", done, "]")
        env_renderer.render_env(show=not in_notebook, frames=True, show_observations=True)
        if sleep_for_animation:
            time.sleep(0.1)
        if in_notebook:
            arrImage = env_renderer.get_image()
            pilImage = PIL.Image.fromarray(arrImage)
            clear_output()
            display(pilImage)
        if done["__all__"]:
            break
    
    if not in_notebook:
        env_renderer.close_window()


if __name__ == '__main__' or in_notebook:
    if 'argv' in globals():
        show_single_agent_main(argv)  # noqa F821
    else:
        show_single_agent_main(sys.argv[1:])


class ObservePredictions(ObservationBuilder):
    """
    We use the provided ShortestPathPredictor to illustrate the usage of predictors in your custom observation.
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        '''
        Because we do not want to call the predictor seperately for every agent we implement the get_many function
        Here we can call the predictor just ones for all the agents and use the predictions to generate our observations
        :param handles:
        :return:
        '''

        self.predictions = self.predictor.get()

        self.predicted_pos = {}

        if handles is None:
            handles = []

        for t in range(len(self.predictions[0])):
            pos_list = []
            for a in handles:
                pos_list.append(self.predictions[a][t][1:3])
            # We transform (x,y) coodrinates to a single integer number for simpler comparison
            self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})

        observations = super().get_many(handles)

        return observations

    def get(self, handle: int = 0) -> np.ndarray:
        '''
        Lets write a simple observation which just indicates whether or not the own predicted path
        overlaps with other predicted paths at any time. This is useless for the task of navigation but might
        help when looking for conflicts. A more complex implementation can be found in the TreeObsForRailEnv class

        Each agent recieves an observation of length 10, where each element represents a prediction step and its value
        is:
         - 0 if no overlap is happening
         - 1 where n i the number of other paths crossing the predicted cell

        :param handle: handeled as an index of an agent
        :return: Observation of handle
        '''

        observation = np.zeros(10)

        # We are going to track what cells where considered while building the obervation and make them accesible
        # For rendering

        visited = OrderedSet()
        for _idx in range(10):
            # Check if any of the other prediction overlap with agents own predictions
            x_coord = self.predictions[handle][_idx][1]
            y_coord = self.predictions[handle][_idx][2]

            # We add every observed cell to the observation rendering
            visited.add((x_coord, y_coord))
            if self.predicted_pos[_idx][handle] in np.delete(self.predicted_pos[_idx], handle, 0):
                # We detect if another agent is predicting to pass through the same cell at the same predicted time
                observation[handle] = 1

        # This variable will be access by the renderer to visualize the observation
        self.env.dev_obs_dict[handle] = visited

        return observation

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)


def create_movie(img_list):
    def init():
        img.set_data(img_list[0])
        return (img,)

    def animate(i):
        img.set_data(img_list[i])
        return (img,)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(img_list[0])
    anim = animation.FuncAnimation(fig, animate, init_func=init,
            frames=len(img_list), interval=200, blit=True)
    return anim


def show_predictions_main(args):
    if in_notebook:
        sleep_for_animation = True
    else:
        try:
            opts, args = getopt.getopt(args, "", ["sleep-for-animation=", ""])
        except getopt.GetoptError as err:
            print(str(err))  # will print something like "option -a not recognized"
            sys.exit(2)
        sleep_for_animation = True
        for o, a in opts:
            if o in ("--sleep-for-animation"):
                sleep_for_animation = str2bool(a)
            else:
                assert False, "unhandled option"

    # Initiate the Predictor
    custom_predictor = ShortestPathPredictorForRailEnv(10)

    # Pass the Predictor to the observation builder
    custom_obs_builder = ObservePredictions(custom_predictor)

    # Initiate Environment
    env = RailEnv(width=10, height=10,
                  rail_generator=complex_rail_generator(nr_start_goal=5, nr_extra=1, min_dist=8, max_dist=99999,
                                                        seed=1), schedule_generator=complex_schedule_generator(),
                  number_of_agents=3, obs_builder_object=custom_obs_builder)

    obs, info = env.reset()
    env_renderer = RenderTool(env, gl="PILSVG")
    list_images = []

    # We render the initial step and show the obsered cells as colored boxes
    env_renderer.render_env(show=not in_notebook, frames=True, show_observations=True, show_predictions=False)

    action_dict = {}
    for step in range(100):
        for a in range(env.get_num_agents()):
            action = np.random.randint(0, 5)
            action_dict[a] = action
        obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=not in_notebook, frames=True, show_observations=True, show_predictions=False)
            
        if in_notebook:
            arrImage = env_renderer.get_image()
            list_images.append(arrImage)
            pilImage = PIL.Image.fromarray(arrImage)
            clear_output()
            display(pilImage)

        print("Rewards: ", all_rewards, "  [done=", done, "]")

        if sleep_for_animation:
            time.sleep(0.1)
        
        if done["__all__"] is True:
            break
    
    if in_notebook:
        display(HTML(create_movie(list_images).to_html5_video()))


if in_notebook:
    show_predictions_main([])
elif __name__ == '__main__':
    if 'argv' in globals():
        show_predictions_main(argv)  # noqa F821
    else:
        show_predictions_main(sys.argv[1:])


