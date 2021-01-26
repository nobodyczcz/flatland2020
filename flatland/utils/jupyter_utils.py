

import PIL
from IPython import display
from ipycanvas import canvas
import time
from numpy.random import RandomState
from flatland.envs import malfunction_generators as malgen
from flatland.envs.agent_utils import EnvAgent
#from flatland.envs import sparse_rail_gen as spgen
from flatland.envs import rail_generators as rail_gen
from flatland.envs import agent_chains as ac
from flatland.envs.rail_env import RailEnv, RailEnvActions
import flatland.envs.malfunction_generators as mfg

from flatland.envs.persistence import RailEnvPersister
from flatland.utils.rendertools import RenderTool
#from flatland.utils import env_edit_utils as eeu
import flatland.evaluators.service2 as fes 
from typing import List, NamedTuple, Optional, Dict

class Behaviour():
    def __init__(self, env):
        self.env = env
        self.nAg = len(env.agents)

    def getActions(self):
        return {}
    
class AlwaysForward(Behaviour):
    def getActions(self):
        return { i:RailEnvActions.MOVE_FORWARD for i in range(self.nAg) }

class DelayedStartForward(AlwaysForward):
    def __init__(self, env, nStartDelay=2):
        self.nStartDelay = nStartDelay
        super().__init__(env)

    def getActions(self):
        iStep = self.env._elapsed_steps + 1
        nAgentsMoving = min(self.nAg, iStep // self.nStartDelay)
        return { i:RailEnvActions.MOVE_FORWARD for i in range(nAgentsMoving) }

AgentPause = NamedTuple("AgentPause", 
    [
        ("iAg", int),
        ("iPauseAt", int),
        ("iPauseFor", int)
    ])


class DeterministicMalfunctionGen:
    def __init__(self, env, lAP:List[AgentPause]):
        self.env:RailEnv = env
        self.lAP:List[AgentPause] = lAP

    def generate(self,
        agent:EnvAgent=None,
        np_random:RandomState=None,
        reset=False) -> Optional[mfg.Malfunction]:

        if reset:
            return mfg.Malfunction(0)

        # find this agent in the env's list - bit risky
        iAg = self.env.agents.index(agent)

        if agent.malfunction_data["malfunction"] < 1:
            for oAP in self.lAP:
                if (oAP.iPauseAt == self.env._elapsed_steps) and (oAP.iAg == iAg):
                    return mfg.Malfunction(oAP.iPauseFor)
        
        return mfg.Malfunction(0)

class ForwardWithPause(Behaviour):
    def __init__(self, env, lPauses:List[AgentPause]):
        """ An agent with a list of pauses
        """

        self.env = env
        self.nAg = len(env.agents)
        self.lPauses = lPauses
        self.dAgPaused = {}

    def getActions(self):
        iStep = self.env._elapsed_steps + 1  # add one because this is called before step()

        # new pauses starting this step
        lNewPauses = [ tPause for tPause in self.lPauses if tPause.iPauseAt == iStep ]

        # copy across the agent index and pause length
        for pause in lNewPauses:
            self.dAgPaused[pause.iAg] = pause.iPauseFor

        # default action is move forward
        dAction = { i:RailEnvActions.MOVE_FORWARD for i in range(self.nAg) }

        # overwrite paused agents with stop
        for iAg in self.dAgPaused:
            dAction[iAg] = RailEnvActions.STOP_MOVING
        
        # decrement the counters for each pause, and remove any expired pauses.
        lFinished = []
        for iAg in self.dAgPaused:
            self.dAgPaused[iAg] -= 1
            if self.dAgPaused[iAg] <= 0:
                lFinished.append(iAg)
        
        for iAg in lFinished:
            self.dAgPaused.pop(iAg, None)
        
        return dAction

class Deterministic(Behaviour):
    def __init__(self, env, dAg_lActions):
        super().__init__(env)
        self.dAg_lActions = dAg_lActions
    
    def getActions(self):
        iStep = self.env._elapsed_steps
        
        dAg_Action = {}
        for iAg, lActions in self.dAg_lActions.items():
            if iStep < len(lActions):
                iAct = lActions[iStep]
            else:
                iAct = RailEnvActions.DO_NOTHING
            dAg_Action[iAg] = iAct
        #print(iStep, dAg_Action[0])
        return dAg_Action

class JuDisplay(fes.Display):
    def __init__(self, envCanvas):
        super().__init__()
        self.envCanvas = envCanvas
        self.action = None
        self.shown = False

    def set_env(self, env):
        super().set_env(env)
        self.envCanvas.setEnv(env)
        if not self.shown:
            self.envCanvas.show()
            self.shown = True

    def step(self, action):
        self.action = action
        self.envCanvas.render()


class RedisBehaviour(Behaviour):
    def __init__(self, env, oSrv):
        super().__init__(env)
        self.oSrv = oSrv

    def getActions(self):
        pass



class EnvCanvas():

    def __init__(self, env=None, behaviour:Behaviour=None, size=(600,300)):
        self.size = size
        self.oCan = canvas.Canvas(size=size)
        if env is not None:
            self.setEnv(env, behaviour)

    def setEnv(self, env, behaviour:Behaviour=None):
        self.env = env
        self.iStep = 0
        if behaviour is None:
            behaviour = AlwaysForward(env)
        self.behaviour = behaviour
        self.oRT = RenderTool(env, show_debug=True, 
            screen_width=self.size[0], screen_height=self.size[1])

        self.render()

    def render(self):
        self.oRT.render_env(show_rowcols=True,  show_inactive_agents=False, show_observations=False)
        self.oCan.put_image_data(self.oRT.get_image())

    def get_image(self):
        return self.oRT.get_image()

    def step(self):
        dAction = self.behaviour.getActions()
        self.env.step(dAction)
        return dAction

    def show(self):
        self.render()
        display.display(self.oCan)


