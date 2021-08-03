import flatland.envs.observations as obs

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import empty_rail_generator, rail_from_file_with_cities
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool


pklfile="./env_data/railway/zz_small_with_cities.pkl"


obs_builder_object = obs.TreeObsForRailEnv(
            max_depth=2,
            predictor=None)
env = RailEnv(width=6, height=4, rail_generator=rail_from_file_with_cities(pklfile),schedule_generator=sparse_schedule_generator(seed=6), number_of_agents=6)

env.reset()

print("created env with " +str(env.agents))
for ag in env.agents:
    print(str(ag.initial_position)+' to '+str(ag.target))
env.reset()
env_renderer=RenderTool(env)
env_renderer.render_env(show=True, show_predictions=True, show_observations=True)

# uncomment to keep the renderer open
input("Press Enter to continue...")

