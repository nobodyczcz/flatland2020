from flatland.envs.generators import rail_from_manual_specifications_generator, rail_from_file
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
         [(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)],
         [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]

env = RailEnv(width=6,
              height=4,
              rail_generator=rail_from_manual_specifications_generator(specs),
              number_of_agents=1)

file_name_2 = "C:\\Daten\\Innovation\\flatland\\notebooks\\temp.pkl"
env = RailEnv(width=1,
                   height=1,
                   rail_generator=rail_from_file(file_name_2),
                   number_of_agents=1,
                   obs_builder_object=GlobalObsForRailEnv(),
                   )

env.reset()

env_renderer = RenderTool(env)
env_renderer.render_env(show=True, show_predictions=False, show_observations=False)

input("Press Enter to continue...")
