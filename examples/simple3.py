
import time
import os
import argparse as ap
import numpy as np  # noqa e402

# In Flatland you can use custom observation builders and predicitors
# Observation builders generate the observation needed by the controller
# Preditctors can be used to do short time prediction which can help in avoiding conflicts in the network
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import GlobalObsForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file
from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.persistence import RailEnvPersister

def loadEnv(sEnvFile):
    env = RailEnv(width=1, height=1,
        rail_generator=rail_from_file(sEnvFile),
        schedule_generator=schedule_from_file(sEnvFile),
        malfunction_generator_and_process_data=malfunction_from_file(sEnvFile),
        obs_builder_object=DummyObservationBuilder(),
        record_steps=True)

    obs, info = env.reset(
        regenerate_rail=True, 
        regenerate_schedule=True,
        activate_agents=False,
        random_seed=1001
        )

    print("agent 0:", env.agents[0])

    return env, obs, info

def loadEnv2(sEnvFile):
    env, env_dict = RailEnvPersister.load_new(sEnvFile)

    obs, info = env.reset(
        regenerate_rail=True, 
        regenerate_schedule=True,
        activate_agents=False,
        random_seed=1001
        )
    return env, obs, info, env_dict



# The first thing we notice is that some agents don't have feasible paths to their target.
# We first look at the map we have created

# nv_renderer.render_env(show=True)
# time.sleep(2)
# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent instead
class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT,
                                 RailEnvActions.STOP_MOVING])

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return


def runEnv(sfEnv, sfOut=None, delay=0):

    env, observations, information = loadEnv(sfEnv)
    

    # Initiate the renderer
    env_renderer = RenderTool(env, gl="PGL", # "TKPILSVG",
                            agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                            show_debug=False,
                            screen_height=1200,  # Adjust these parameters to fit your resolution
                            screen_width=1600)  # Adjust these parameters to fit your resolution

    # Initialize the agent with the parameters corresponding to the environment and observation_builder
    controller = RandomAgent(218, env.action_space[0])

    env_renderer.reset()

    action_dict={}
    score = 0
    frame_step = 0
    nSteps = 500

    #print("obs, rew, done, info:", observations, rewards, dones, information)
    print("obs:", len(observations), observations)

    for step in range(nSteps):
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            if type(observations) is bool:
                action = RailEnvActions.MOVE_FORWARD
            else:
                action = controller.act(observations[a])

            action_dict.update({a: action})

        next_obs, all_rewards, done, _ = env.step(action_dict)

        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
        # env_renderer.gl.save_image('./misc/Fames2/flatland_frame_{:04d}.png'.format(step))
        frame_step += 1
        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            controller.step((observations[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
            score += all_rewards[a]

        observations = next_obs.copy()
        if done['__all__']:
            break
        print('Episode: Steps {}\t Score = {:.2f}\r'.format(step, score), end="")

        time.sleep(delay)

    if sfOut is not None:
        env.save_episode(sfOut)

def replay_episode(sfEnv, delay=0):

    env, observations, information, env_dict = loadEnv2(sfEnv)
    

    # Initiate the renderer
    env_renderer = RenderTool(env, gl="PGL", # "TKPILSVG",
                            agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                            show_debug=False,
                            screen_height=1200,  # Adjust these parameters to fit your resolution
                            screen_width=1600)  # Adjust these parameters to fit your resolution

    # Initialize the agent with the parameters corresponding to the environment and observation_builder
    controller = RandomAgent(218, env.action_space[0])

    env_renderer.reset()

    action_dict={}
    score = 0
    frame_step = 0
    

    print(env_dict.keys())
    lllEp = env_dict["episode"]
    nSteps = len(lllEp)
    time.sleep(2)

    #print("obs, rew, done, info:", observations, rewards, dones, information)
    #print("obs:", len(observations), observations)

    for step in range(nSteps):
        # Chose an action for each agent in the environment
        # for a in range(env.get_num_agents()):
        #     if type(observations) is bool:
        #         action = RailEnvActions.MOVE_FORWARD
        #     else:
        #         action = controller.act(observations[a])

        #     action_dict.update({a: action})

        # next_obs, all_rewards, done, _ = env.step(action_dict)

        llStep = lllEp[step]
        for iAg, lPosDirMalf in enumerate(llStep):
            oAg = env.agents[iAg]
            oAg.position = lPosDirMalf[0:2]
            oAg.direction = lPosDirMalf[2]
            oAg.malfunction_data = dict(malfunction=lPosDirMalf[3])

        env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
        # env_renderer.gl.save_image('./misc/Fames2/flatland_frame_{:04d}.png'.format(step))
        frame_step += 1
        # Update replay buffer and train agent
        #for a in range(env.get_num_agents()):
        #    controller.step((observations[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
        #    score += all_rewards[a]

        #observations = next_obs.copy()
        #if done['__all__']:
        #    break
        #print('Episode: Steps {}\t Score = {:.2f}\r'.format(step, score), end="")

        time.sleep(delay)


def main():
    parser = ap.ArgumentParser(description='Run envs')

    parser.add_argument("-e", '--env',  type=str,
        metavar=("EnvFile"),
        help='load env from EnvFile')

    parser.add_argument("-r", '--replay',  type=str,
        metavar=("EnvFile"),
        help='load env with episode from EnvFile and replay')

    parser.add_argument("-o", '--output',  type=str,
        metavar=("OutputEnvFile"),
        help='save env episode to OutputEnvFile')

    parser.add_argument("-d", '--delay',  type=float,
        #metavar=("OutputEnvFile"),
        default=0,
        help='time delay between steps in seconds')


    args=parser.parse_args()
    print(args)

    if args.env:
        print("Load :", args.env)
        runEnv(args.env, args.output, delay=args.delay)

    if args.replay:
        print("replay from :", args.replay)
        replay_episode(args.replay, delay=args.delay)



if __name__=="__main__":
    main()