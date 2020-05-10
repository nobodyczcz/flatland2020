
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from gym import wrappers


class FlatlandRenderWrapper(RailEnv):

    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    observation_space = None

    def __init__(self, use_renderer=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_renderer = use_renderer
        self.renderer = None
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 10
        }
        if self.use_renderer:
            self.initialize_renderer()

    def reset(self, *args, **kwargs):
        if self.use_renderer:
            self.renderer.reset()
        return super().reset(self, *args, **kwargs)

    def render(self, mode='human'):
        """
        This methods provides the option to render the
        environment's behavior to a window which should be
        readable to the human eye if mode is set to 'human'.
        """
        if not self.use_renderer:
            return

        if not self.renderer:
            self.initialize_renderer(mode=mode)

        return self.update_renderer(mode=mode)

    def initialize_renderer(self, mode="human"):
        # Initiate the renderer
        self.renderer = RenderTool(self, gl="PGL",  # gl="TKPILSVG",
                                       agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                                       show_debug=False,
                                       screen_height=600,  # Adjust these parameters to fit your resolution
                                       screen_width=800)  # Adjust these parameters to fit your resolution

    def update_renderer(self, mode='human'):
        image = self.renderer.render_env(show=True, show_observations=False, show_predictions=False,
                                             return_image=True)
        return image[...,::-1]

    def set_renderer(self, renderer):
        self.use_renderer = renderer
        if self.use_renderer:
            self.initialize_renderer(mode=self.use_renderer)

    def close(self):
        if self.renderer:
            self.renderer.close_window()
            self.renderer = None

import numpy as np  # noqa e402

width = 25  # With of map
height = 25  # Height of map
nr_trains = 10  # Number of trains that have an assigned task in the env
cities_in_map = 2  # Number of cities where agents can start or end


seed = 14  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 3  # 6  # Max number of parallel tracks within a city, representing a realistic trainstation

rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities,
                                       )

# The schedule generator can make very basic schedules with a start point, end point and a speed profile for each agent.
# The speed profiles can be adjusted directly as well as shown later on. We start by introducing a statistical
# distribution of speed profiles

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We can now initiate the schedule generator with the given speed profiles

schedule_generator = sparse_schedule_generator(speed_ration_map)

# We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
# during an episode.

stochastic_data = MalfunctionParameters(malfunction_rate=10000,  # Rate of malfunction occurence
                                        min_duration=15,  # Minimal duration of malfunction
                                        max_duration=50  # Max duration of malfunction
                                        )
# Custom observation builder without predictor
observation_builder = GlobalObsForRailEnv()

env = FlatlandRenderWrapper(width=width,
                            height=height,
                            rail_generator=rail_generator,
                            schedule_generator=schedule_generator,
                            number_of_agents=nr_trains,
                            obs_builder_object=observation_builder,
                            malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                            remove_agents_at_target=True,
                            use_renderer='human')
env = wrappers.Monitor(env, "videos", force=True)

env.reset()

NUMBER_OF_AGENTS = env.get_num_agents()


def my_controller():
    """
    You are supposed to write this controller
    """
    _action = {}
    for _idx in range(NUMBER_OF_AGENTS):
        _action[_idx] = np.random.randint(0, 5)
    return _action


for step in range(100):

    _action = my_controller()
    obs, all_rewards, done, info = env.step(_action)
    print("Rewards: {}, [done={}]".format(all_rewards, done))

