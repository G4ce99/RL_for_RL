if __name__ == "__main__":
    from rlgym_ppo import Learner
    from utils import AdaptiveLearnerWrapper
    import torch
    import wandb
    from my_env import build_rlgym_v2_env

    load_run = False
    run_name = "Test3_adaptive_lr_test"
    run_id = None
    use_adaptive_lr = True
    load_weights_manually = False
    weights_dir_path = "./data/FILL_ME_IN"

    if not load_run:
        if wandb.run is not None:
            wandb.finish()
        wandb_run = wandb.init(project="RL4RL", name=run_name)
    else:
        wandb_run = wandb.init(project="RL4RL", name=run_name, id=run_id, resume="must")
    
    # 32 processes
    n_proc = 24 # Trying this as well

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner_params = {
        "n_proc": n_proc,
        "min_inference_size": min_inference_size,
        "ppo_batch_size": 100_000, # batch size - set this number to as large as your GPU can handle
        "policy_layer_sizes": [2048, 2048, 1024, 1024], # policy network
        "critic_layer_sizes": [2048, 2048, 1024, 1024], # value network
        "ts_per_iteration": 100_000, # timesteps per training iteration - set this equal to the batch size
        "exp_buffer_size": 300_000, # size of experience buffer - keep this 2 - 3x the batch size
        "ppo_minibatch_size": 50_000, # minibatch size - set this less than or equal to the batch size
        "ppo_ent_coef": 0.01, # entropy coefficient - this determines the impact of exploration on the policy
        "policy_lr": 2e-4, # policy learning rate
        "critic_lr": 2e-4, # value function learning rate
        "ppo_epochs": 2,   # number of PPO epochs
        "standardize_returns": True,
        "standardize_obs": False,
        "save_every_ts": 10_000_000, # save every 10M steps
        "timestep_limit": 1_000_000_000, # Train for 1B steps
        "log_to_wandb": True,
        "wandb_run": wandb_run,
        "load_wandb": False,
        "render": False, 
        "device": ("cuda" if torch.cuda.is_available() else "cpu")
    }

    if not use_adaptive_lr:
        learner = Learner(build_rlgym_v2_env, **learner_params)
    else: 
        learner_wrapper = AdaptiveLearnerWrapper(**learner_params)
        learner = learner_wrapper.learner

    if load_weights_manually:
        learner.ppo_learner.load_from(weights_dir_path)
    
    learner.learn()
    wandb_run.finish()