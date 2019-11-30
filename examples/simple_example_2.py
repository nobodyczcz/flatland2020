import random

import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
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


random.seed(100)
np.random.seed(100)

# Relative weights of each cell type to be used by the random rail generators.
transition_probability = [1.0,  # empty cell - Case 0
                          1.0,  # Case 1 - straight
                          1.0,  # Case 2 - simple switch
                          0.3,  # Case 3 - diamond drossing
                          0.5,  # Case 4 - single slip
                          0.5,  # Case 5 - double slip
                          0.2,  # Case 6 - symmetrical
                          0.0,  # Case 7 - dead end
                          0.2,  # Case 8 - turn left
                          0.2,  # Case 9 - turn right
                          1.0]  # Case 10 - mirrored switch

# Example generate a random rail
env = RailEnv(width=10, height=10,
              rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
              number_of_agents=3)

env.reset()
env_renderer = RenderTool(env, gl="PIL")
env_renderer.render_env(show=not in_notebook)

# uncomment to keep the renderer open
# input("Press Enter to continue...")

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
