from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
from utils import AdaptiveLearnerWrapper
import numpy as np

# Custom Reward Functions that I wrote
class AdapativeLearningRateReward(RewardFunction[AgentID, GameState, float]):
    """
    A reward that dynamically adjusts the model learning rate (DONE BUT NOT TESTED)
    Learning rate calculated by max_lr * gamma ^ (alpha*cumulative_ts)
    """
    def __init__(self, alpha=1e-8, gamma=0.91):
        super().__init__()
        self.max_policy_lr = None
        self.max_critic_lr = None
        self.alpha = alpha
        self.gamma = gamma
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        if AdaptiveLearnerWrapper.learner is not None:
            if self.max_policy_lr is None:
                self.max_policy_lr = AdaptiveLearnerWrapper.learner.policy_lr
                self.max_critic_lr = AdaptiveLearnerWrapper.learner.critic_lr
            scaling_factor = (self.gamma ** (self.alpha * AdaptiveLearnerWrapper.learner.agent.cumulative_timesteps))
            policy_lr = self.max_policy_lr * scaling_factor
            critic_lr = self.max_critic_lr * scaling_factor
            self.learner_wrapper.learner.update_learning_rate(new_policy_lr=policy_lr, new_critic_lr=critic_lr)
        return {agent: 0. for agent in agents}

class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for facing the ball (DONE NOT TESTED)
    """
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            vector_to_ball = ball_physics.position - car_physics.position
            dir_to_ball = vector_to_ball * (1/np.linalg.norm(vector_to_ball))
            forward_vector = car_physics.forward()
            forward_dir = forward_vector * (1/np.linalg.norm(forward_vector))
            rewards[agent] = np.dot(dir_to_ball, forward_dir)
        
        return rewards

class ChallengeBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for challenging the Ball when closest (DONE NOT TESTED)
    Epsilon is supposed to encourage increase aggretion even if not exactly the closest (gives some leeway)
    """
    def __init__(self, epsilon=0.9):
        super().__init__()
        self.epsilon = epsilon
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        
        rewards = {}
        vectors_to_ball = {}
        min_dist = float('inf')
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            vectors_to_ball[agent] = (ball_physics.position - car_physics.position)
            
            min_dist = min(min_dist, np.linalg.norm(vectors_to_ball[agent]))
        
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            player_vel = car_physics.linear_velocity

            dist_to_ball = np.linalg.norm(vectors_to_ball[agent])
            rewards[agent] = 0.0
            if self.epsilon * dist_to_ball < min_dist:
                dir_to_ball = vectors_to_ball[agent] / dist_to_ball
                speed_toward_ball = np.dot(player_vel, dir_to_ball)
                rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
        
        return rewards
    
class DefensePositionReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for being well positioned when opponent has the ball(NOT DONE)
    """
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}
    
class BallFarOppsReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for being in the air (NOT DONE)
    """
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}
    
class BallGoalDistanceReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent when Ball is closer to opponents goal (DONE BUT NOT TESTED)
       Notes: Formula tanh(1.5 * log(x / (1-x))) was used for distribution 
       with 1.5 as a scaling factor I thought was appropriate for this use case"""
    def __init__(self, scaling_factor=1.5):
        super().__init__()
        self.tanh_scaling_factor = scaling_factor

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball = state.ball
        orange_goal_dist = np.linalg.norm(np.array([0, common_values.BACK_NET_Y, 0]) - ball.position)
        blue_goal_dist = np.linalg.norm(np.array([0, -common_values.BACK_NET_Y, 0]) - ball.position)
        
        # Ensure no division by 0
        if orange_goal_dist == 0:
            orange_reward = -1
        elif blue_goal_dist == 0:
            orange_reward = 1
        else:
            orange_reward = np.tanh(self.tanh_scaling_factor * np.log(orange_goal_dist / blue_goal_dist))
        
        return {agent: orange_reward if state.cars[agent].is_orange else -orange_reward for agent in agents}
    
class HitBallInAirTowardsGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent when Ball is hit in the air towards correct Goal (DONE BUT NOT TESTED)
    Epsilon is attempting to ensure that the reward is only given when hit with sufficient power
    Notes: Maybe want this to only happen when moving toward other goals
    """
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsion = epsilon

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        if car.is_orange:
            ball = state.ball
            goal_y = common_values.BACK_NET_Y
        else:
            ball = state.inverted_ball
            goal_y = -common_values.BACK_NET_Y
        if car.ball_touches > 0 and np.abs(np.dot(np.array([0, 0, 1]), ball.linear_velocity)) > self.epsilon*common_values.BALL_MAX_SPEED:
            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist
            
            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            return vel_toward_goal / common_values.BALL_MAX_SPEED
        return 0.

# --- Below this point are reward functions that I did not implement myself and cited where I found them ---
# Note: I may have modified some of these reward functions slightly for this bot

# Some nice base reward functions to have from RLGym's guide/repo
class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for hitting the ball toward the opponent's goal
    My change: I made this reward zero sum 
    """
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.is_orange:
                ball = state.ball
                goal_y = common_values.BACK_NET_Y
            else:
                ball = state.inverted_ball
                goal_y = -common_values.BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist
            
            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            # rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0)
            rewards[agent] = vel_toward_goal / common_values.BALL_MAX_SPEED
        return rewards

class TouchReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward of 1 if the agent touches the ball, 0 otherwise.
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        return 1. if state.cars[agent].ball_touches > 0 else 0.

class GoalReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward of 1 if the agent's team scored a goal, -1 if the opposing team scored a goal,
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        if state.goal_scored:
            return 1 if state.scoring_team == state.cars[agent].team_num else -1
        else:
            return 0

class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for moving quickly toward the ball
    """
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            player_vel = car_physics.linear_velocity
            pos_diff = (ball_physics.position - car_physics.position)
            dist_to_ball = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(player_vel, dir_to_ball)

            rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
        return rewards

class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}

# See https://github.com/RLGym/rlgym-tools/tree/main/rlgym_tools/rocket_league/reward_functions for more information
from rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward
from rlgym_tools.rocket_league.reward_functions.boost_change_reward import BoostChangeReward
from rlgym_tools.rocket_league.reward_functions.boost_keep_reward import BoostKeepReward
from rlgym_tools.rocket_league.reward_functions.team_spirit_reward_wrapper import TeamSpiritRewardWrapper
from rlgym_tools.rocket_league.reward_functions.ball_travel_reward import BallTravelReward

