from rlgym_ppo.learner import Learner
from rlgym_ppo.util.metrics_logger import MetricsLogger
import numpy as np
import pandas as pd
import io
import wandb
from contextlib import redirect_stdout

from my_env import build_rlgym_v2_env

class AdaptiveLearnerWrapper(MetricsLogger):
    """
    Notes to self: Someone on discord suggested to make a reward function. I found this way to be far simpler and more flexible.
    It instead updates learning rates during reporting phase of current iteration. 
    
    Learning rate calculated by max_lr * gamma ^ (alpha*cumulative_ts)
    Alpha and gamma set up for what I think is good distribution for 
    """
    def __init__(self, alpha=1e-8, gamma=0.91, *args, **kwargs):
        self.max_policy_lr = kwargs["policy_lr"]
        self.max_critic_lr = kwargs["critic_lr"]
        self.alpha = alpha
        self.gamma = gamma

        self.learner = Learner(build_rlgym_v2_env, metrics_logger=self, **kwargs)
        self.learner.policy_lr, self.learner.critic_lr, self.timestep_cnt = self.initialize_from_run(kwargs["wandb_run"])
        
    def initialize_from_run(self, wandb_run):
        api = wandb.Api()
        run = api.run(f"{wandb_run.entity}/{wandb_run.project}/{wandb_run.id}")

        history = pd.DataFrame(run.history(keys=["_step", "Cumulative Timesteps"]))
        if not history.empty:
            timestep_cnt = history["_step"].dropna().max()+1
            cumulative_ts = history["Cumulative Timesteps"].dropna().max()
        else:
            timestep_cnt = 0
            cumulative_ts = 0
        policy_lr, critic_lr = self.calculate_learning_rates(cumulative_ts)
        return policy_lr, critic_lr, timestep_cnt
    
    def calculate_learning_rates(self, cumulative_ts):
        scaling_factor = (self.gamma ** (self.alpha * cumulative_ts))
        policy_lr = self.max_policy_lr * scaling_factor
        critic_lr = self.max_critic_lr * scaling_factor
        return policy_lr, critic_lr

    def _collect_metrics(self, game_state):
        return np.array([])

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        metrics = {"Policy Learning Rate": self.learner.policy_lr, 
                   "Critic Learning Rate": self.learner.critic_lr}
        wandb_run.log(metrics, step=self.timestep_cnt)
        self.timestep_cnt+=1

        print("--------CURRENT LEARNING RATE--------")
        print(f"Policy learning rate: {self.learner.policy_lr}")
        print(f"Critic learning rate: {self.learner.critic_lr}")
        
        policy_lr, critic_lr = self.calculate_learning_rates(cumulative_timesteps)
        
        # Suppressing print statements since making my own. 
        with io.StringIO() as buffer, redirect_stdout(buffer):
            self.learner.update_learning_rate(new_policy_lr=policy_lr, new_critic_lr=critic_lr)
        return
