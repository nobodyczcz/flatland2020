from flatland.envs.rail_env import RailEnv,  RailEnvActions
from flatland.envs.rail_generators import rail_from_manual_specifications_generator
from flatland.utils.rendertools import RenderTool
import time
# These are used in the notebook version of this code, but not the plain python
from IPython.core.display import display, clear_output
import PIL

in_notebook = False
try:
    # get_ipython() is only defined inside ipython; we have to ignore it in flake8 with noqa
    get_ipython()  # noqa F821
    in_notebook = True
except(NameError):
    in_notebook = False 


# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
         [(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)],
         [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]
env = RailEnv(width=6, height=4, rail_generator=rail_from_manual_specifications_generator(specs), number_of_agents=1)
env.reset()

# Step the env, with an action for the agent, which causes it to become "active"
# You can re-run this cell to take another step.
# In this example the agent terminates after just 2 or 3 steps.
env.step({0: RailEnvActions.MOVE_FORWARD})
env_renderer = RenderTool(env)
env_renderer.render_env(show=not in_notebook, show_predictions=False, show_observations=False)
if in_notebook:
    arrImage = env_renderer.get_image()
    pilImage = PIL.Image.fromarray(arrImage)
    clear_output()
    display(pilImage)
else:
    # if running bare python, keep the TK window open for a short time before exiting
    time.sleep(1)
    # uncomment to keep the renderer open
    # input("Press Enter to continue...")
