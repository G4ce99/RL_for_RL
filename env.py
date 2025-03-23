# TODO: ADD SUPPORT FOR TEAMSPIRIT FACTOR AND RANDOM PROBABILITY PROBABILITY
    
def build_custom_env(team_spirit_factor, random_start_prob):
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_tools.rocket_league.state_mutators.weighted_sample_mutator import WeightedSampleMutator
    from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import RandomPhysicsMutator
    import numpy as np

    from reward import (SpeedTowardBallReward, InAirReward, VelocityBallToGoalReward, GoalReward, TouchReward,
                        ChallengeBallReward, BallGoalDistanceReward, FaceBallReward, HitBallInAirTowardsGoalReward, AdvancedTouchReward, 
                        BoostChangeReward, BoostKeepReward, TeamSpiritRewardWrapper)
    
    spawn_opponents = True
    team_size = 2
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds), TimeoutCondition(timeout_seconds=game_timeout_seconds))

    reward_fn = TeamSpiritRewardWrapper(CombinedReward(
        (GoalReward(), 40), 
        (TouchReward(), 2),
        (SpeedTowardBallReward(), 0.5), 
        (InAirReward(), 0.01),
        (VelocityBallToGoalReward(), 2),
        (ChallengeBallReward(), 0.5),
        (BallGoalDistanceReward(), 1),
        # (HitBallInAirTowardsGoalReward(), 0.2),
        (FaceBallReward(), 0.005),
        (BoostChangeReward(), 0.6),
        (BoostKeepReward(), 0.4)
    ), team_spirit_factor)

    obs_builder = DefaultObs(zero_padding=None,
                             pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
                             ang_coef=1 / np.pi,
                             lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                             boost_coef=1 / 100.0,)

    
    state_mutator = MutatorSequence(FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
                                    WeightedSampleMutator([KickoffMutator(), RandomPhysicsMutator()], 
                                                          [1-random_start_prob, random_start_prob]))
    
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()) #I added the last one. Lets see what happens now.
    return rlgym_env
