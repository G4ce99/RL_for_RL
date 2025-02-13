if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from my_env import build_rlgym_v2_env
    from rlgym_ppo import Learner
    import wandb

    if wandb.run is not None:
        wandb.finish()
    wandb_run = wandb.init(project="RL4RL", name="Example_Visual")
    
    # 32 processes
    n_proc = 16 # Trying this as well

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None,
                      ppo_batch_size=50000, # batch size - set this number to as large as your GPU can handle
                      policy_layer_sizes=[2048, 2048, 1024, 1024], # policy network
                      critic_layer_sizes=[2048, 2048, 1024, 1024], # value network
                      ts_per_iteration=50000, # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=150000, # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50000, # minibatch size - set this less than or equal to the batch size
                      ppo_ent_coef=0.01, # entropy coefficient - this determines the impact of exploration on the policy
                      policy_lr=5e-5, # policy learning rate
                      critic_lr=5e-5, # value function learning rate
                      ppo_epochs=1,   # number of PPO epochs
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=1_000_000, # save every 1M steps
                      timestep_limit=100_000_000, # Train for 1B steps
                      log_to_wandb=True,
                      wandb_run=wandb_run,
                      load_wandb=False, 
                      render=False)
    learner.learn()
    wandb_run.finish()