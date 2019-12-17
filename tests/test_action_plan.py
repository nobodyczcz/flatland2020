from flatland.action_plan.action_plan import TrainrunWaypoint, ActionPlanElement, \
    ControllerFromTrainruns, MalfunctionInActionPlan
from flatland.action_plan.action_plan_player import ControllerFromTrainrunsReplayer
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.malfunction_generators import single_malfunction_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map, sparse_rail_generator
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.schedule_generators import random_schedule_generator, sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.utils.simple_rail import make_simple_rail
from test_utils import create_flatland_environment_with_malfunction


def test_action_plan(rendering: bool = False):
    """Tests ActionPlanReplayer: does action plan generation and replay work as expected."""
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=77),
                  number_of_agents=2,
                  obs_builder_object=GlobalObsForRailEnv(),
                  remove_agents_at_target=True
                  )
    env.reset()
    env.agents[0].initial_position = (3, 0)
    env.agents[0].target = (3, 8)
    env.agents[0].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].initial_position = (3, 8)
    env.agents[1].initial_direction = Grid4TransitionsEnum.WEST
    env.agents[1].target = (0, 3)
    env.agents[1].speed_data['speed'] = 0.5  # two
    env.reset(False, False, False)
    for handle, agent in enumerate(env.agents):
        print("[{}] {} -> {}".format(handle, agent.initial_position, agent.target))

    chosen_path_dict = {0: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(3, 0), direction=3)),
                            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(3, 1), direction=1)),
                            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(3, 2), direction=1)),
                            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(3, 3), direction=1)),
                            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(3, 4), direction=1)),
                            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(3, 5), direction=1)),
                            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(3, 6), direction=1)),
                            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(3, 7), direction=1)),
                            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(3, 8), direction=1)),
                            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(3, 8), direction=5))],
                        1: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(3, 8), direction=3)),
                            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(3, 7), direction=3)),
                            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(3, 6), direction=3)),
                            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(3, 5), direction=3)),
                            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(3, 4), direction=3)),
                            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(3, 3), direction=3)),
                            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(2, 3), direction=0)),
                            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(1, 3), direction=0)),
                            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(0, 3), direction=0))]}
    expected_action_plan = [[
        # take action to enter the grid
        ActionPlanElement(0, RailEnvActions.MOVE_FORWARD),
        # take action to enter the cell properly
        ActionPlanElement(1, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(2, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(3, RailEnvActions.STOP_MOVING),
        ActionPlanElement(13, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(14, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(15, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(16, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(17, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(18, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(19, RailEnvActions.STOP_MOVING)

    ], [
        ActionPlanElement(0, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(1, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(3, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(5, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(7, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(9, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(11, RailEnvActions.MOVE_RIGHT),
        ActionPlanElement(13, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(15, RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(17, RailEnvActions.STOP_MOVING),

    ]]

    deterministic_controller = ControllerFromTrainruns(env, chosen_path_dict)
    deterministic_controller.print_action_plan()
    ControllerFromTrainruns.assert_actions_plans_equal(expected_action_plan, deterministic_controller.action_plan)
    if rendering:
        renderer = RenderTool(env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=True,
                              clear_debug_text=True,
                              screen_height=1000,
                              screen_width=1000)

    def render(*argv):
        if rendering:
            renderer.render_env(show=True, show_observations=False, show_predictions=False)

    ControllerFromTrainrunsReplayer.replay_verify(deterministic_controller, env, call_back=render)


def test_action_plan_malfunction(rendering=False):
    env_malfunction = create_flatland_environment_with_malfunction(number_of_agents=2,
                                                                   width=30,
                                                                   height=30,
                                                                   seed_value=12,
                                                                   max_num_cities=20,
                                                                   grid_mode=True,
                                                                   max_rails_between_cities=2,
                                                                   max_rails_in_city=6,
                                                                   malfunction_duration=20,
                                                                   earliest_malfunction=20,
                                                                   speed_data={1.0: 0.3333333333333333,
                                                                               0.5: 0.3333333333333333,
                                                                               0.3333333333333333: 0.0,
                                                                               0.25: 0.3333333333333333})
    env_malfunction.reset(random_seed=12)
    chosen_path_dict = {0: [
        TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(8, 23), direction=1)),
        TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(8, 24), direction=1)),
        TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(8, 25), direction=1)),
        TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(8, 26), direction=1)),
        TrainrunWaypoint(scheduled_at=53, waypoint=Waypoint(position=(8, 27), direction=1)),
        TrainrunWaypoint(scheduled_at=57, waypoint=Waypoint(position=(8, 28), direction=1)),
        TrainrunWaypoint(scheduled_at=61, waypoint=Waypoint(position=(8, 29), direction=1)),
        TrainrunWaypoint(scheduled_at=65, waypoint=Waypoint(position=(9, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=69, waypoint=Waypoint(position=(10, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=73, waypoint=Waypoint(position=(11, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=77, waypoint=Waypoint(position=(12, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=81, waypoint=Waypoint(position=(13, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=85, waypoint=Waypoint(position=(14, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=89, waypoint=Waypoint(position=(15, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=93, waypoint=Waypoint(position=(16, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=97, waypoint=Waypoint(position=(17, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=101, waypoint=Waypoint(position=(18, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=105, waypoint=Waypoint(position=(19, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=109, waypoint=Waypoint(position=(20, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=113, waypoint=Waypoint(position=(21, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=117, waypoint=Waypoint(position=(22, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=121, waypoint=Waypoint(position=(23, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=125, waypoint=Waypoint(position=(24, 29), direction=2)),
        TrainrunWaypoint(scheduled_at=129, waypoint=Waypoint(position=(24, 28), direction=3)),
        TrainrunWaypoint(scheduled_at=133, waypoint=Waypoint(position=(24, 27), direction=3)),
        TrainrunWaypoint(scheduled_at=137, waypoint=Waypoint(position=(24, 26), direction=3)),
        TrainrunWaypoint(scheduled_at=141, waypoint=Waypoint(position=(24, 25), direction=3)),
        TrainrunWaypoint(scheduled_at=145, waypoint=Waypoint(position=(24, 24), direction=3)),
        TrainrunWaypoint(scheduled_at=149, waypoint=Waypoint(position=(24, 23), direction=3))],
        1: [TrainrunWaypoint(scheduled_at=109, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=111, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=112, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=113, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=114, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=115, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=126, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=127, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=128, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=129, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=130, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=131, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=132, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=133, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=134, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=135, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=136, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=137, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=138, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=139, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=140, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=141, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=142, waypoint=Waypoint(position=(8, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=143, waypoint=Waypoint(position=(8, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=144, waypoint=Waypoint(position=(8, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=145, waypoint=Waypoint(position=(7, 26), direction=0)),
            TrainrunWaypoint(scheduled_at=146, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=147, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=148, waypoint=Waypoint(position=(7, 23), direction=3))]}

    expected_action_plan = [[
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
        ActionPlanElement(scheduled_at=16, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=17, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=41, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=45, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=49, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=53, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=57, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=61, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=65, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=69, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=73, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=77, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=81, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=85, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=89, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=93, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=97, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=101, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=105, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=109, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=113, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=117, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=121, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=125, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=129, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=133, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=137, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=141, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=145, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=149, action=RailEnvActions.STOP_MOVING)
    ], [
        ActionPlanElement(scheduled_at=0, action=RailEnvActions.DO_NOTHING),
        ActionPlanElement(scheduled_at=109, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=110, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=111, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=112, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=113, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=114, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=115, action=RailEnvActions.STOP_MOVING),
        ActionPlanElement(scheduled_at=125, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=126, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=127, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=128, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=129, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=130, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=131, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=132, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=133, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=134, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=135, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=136, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=137, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=138, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=139, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=140, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=141, action=RailEnvActions.MOVE_LEFT),
        ActionPlanElement(scheduled_at=142, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=143, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=144, action=RailEnvActions.MOVE_RIGHT),
        ActionPlanElement(scheduled_at=145, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=146, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=147, action=RailEnvActions.MOVE_FORWARD),
        ActionPlanElement(scheduled_at=148, action=RailEnvActions.STOP_MOVING)
    ]]

    deterministic_controller = ControllerFromTrainruns(
        env_malfunction, chosen_path_dict,
        malfunctions=[MalfunctionInActionPlan(time_step=20, agent_id=0, malfunction_duration=20)])
    deterministic_controller.print_action_plan()
    ControllerFromTrainruns.assert_actions_plans_equal(
        expected_action_plan,
        deterministic_controller.action_plan)
    if rendering:
        renderer = RenderTool(env_malfunction, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=True,
                              clear_debug_text=True,
                              screen_height=1000,
                              screen_width=1000)

    def render(*argv):
        if rendering:
            renderer.render_env(show=True, show_observations=False, show_predictions=False)

    ControllerFromTrainrunsReplayer.replay_verify(deterministic_controller, env_malfunction, call_back=render)
