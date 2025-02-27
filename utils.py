from rlgym_ppo.learner import Learner
from rlgym_ppo.util.metrics_logger import MetricsLogger
import numpy as np

from my_env import build_rlgym_v2_env

class AdaptiveLearnerWrapper(MetricsLogger):
    """
    Notes to self: 
    (This approach was described by someone on the RLGym discord that was gatekeeping the adaptive_lr reward implementation)
    To implement adaptive learning rate, the simplest way was to make it a reward that dynamically adjusts it.
    Such as reward needs access to the Learner object itself, thus I used a wrapper and changed the definition 
    to build rl_gum_v2_env. I made this class inherit MetricsLogger to enable visualization of the learning rate
    to wandb. (DONE BUT NOT TESTED)
    """
    learner = None
    def __init__(self, *args, **kwargs):
        AdaptiveLearnerWrapper.learner = Learner(build_rlgym_v2_env, metrics_logger=self, **kwargs)
        AdaptiveLearnerWrapper.learner.policy_lr = kwargs["policy_lr"]
        AdaptiveLearnerWrapper.learner.critic_lr = kwargs["critic_lr"]
        self.timestep_cnt = 0

    def _collect_metrics(self, game_state):
        return np.array([])

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        metrics = {"Policy Learning Rate": AdaptiveLearnerWrapper.learner.policy_lr, 
                   "Critic Learning Rate": AdaptiveLearnerWrapper.learner.critic_lr}
        wandb_run.log(metrics, step=self.timestep_cnt)
        self.timestep_cnt+=1
        return
