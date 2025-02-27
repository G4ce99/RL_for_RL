from rlgym_ppo.learner import Learner
from rlgym_ppo.util.metrics_logger import MetricsLogger
import numpy as np

from my_env import build_rlgym_v2_env

class AdaptiveLearnerWrapper(MetricsLogger):
    """
    Notes to self: 
    
    Learning rate calculated by max_lr * gamma ^ (alpha*cumulative_ts)
    Alpha and gamma set up for what I think is good distribution for 
    """
    def __init__(self, alpha=1e-8, gamma=0.91, *args, **kwargs):
        self.learner = Learner(build_rlgym_v2_env, metrics_logger=self, **kwargs)
        self.learner.policy_lr =  kwargs["policy_lr"]
        self.learner.critic_lr =  kwargs["critic_lr"]
        
        self.max_policy_lr = self.learner.policy_lr
        self.max_critic_lr = self.learner.critic_lr
        self.alpha = alpha
        self.gamma = gamma
        self.timestep_cnt = 0

    def _collect_metrics(self, game_state):
        return np.array([])

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        metrics = {"Policy Learning Rate": self.learner.policy_lr, 
                   "Critic Learning Rate": self.learner.critic_lr}
        wandb_run.log(metrics, step=self.timestep_cnt)
        self.timestep_cnt+=1
        
        print("--------Next Learning Rate--------")
        scaling_factor = (self.gamma ** (self.alpha * self.learner.agent.cumulative_timesteps))
        policy_lr = self.max_policy_lr * scaling_factor
        critic_lr = self.max_critic_lr * scaling_factor
        self.learner.update_learning_rate(new_policy_lr=policy_lr, new_critic_lr=critic_lr)
        return