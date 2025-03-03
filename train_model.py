# TODO's: 
### MAKE THIS INTO A CLASS CALLED Model for multistage training
### ALLOW SOME PARAMETERS TO BE LOADED FROM THE BOOK_KEEPING_VARS.json FILE INSTEAD OF BEING MANUALLY SET

if __name__ == "__main__":
    from rlgym_ppo import Learner
    from adapt_lr import AdaptiveLearnerWrapper
    import torch
    import wandb
    from my_env import build_rlgym_v2_env
    from util import create_trimmed_run

    load_run = True
    proj_name = "RL4RL"
    run_name = "Test3_adaptive_lr"
    old_run_id = "1uu4xfdd"
    use_adaptive_lr = True
    load_weights_manually = True
    weights_dir_path = "./data/Test3_adaptive_lr"
    last_step = 9599

    if wandb.run is not None:
        wandb.finish()

    if not load_run:
        #Creates new run from scratch
        wandb_run = wandb.init(project=proj_name, name=run_name)
    elif load_weights_manually:
        #Creates a clean copy of an old run with ending removed (used when a run was killed due to GPU constraints)
        wandb_run = create_trimmed_run(proj_name, run_name, old_run_id, last_step)
    else:
        #Uses RLGym PPO's automatic load from checkpoint to continue from most recent checkpoint saved locally
        #Note some training may be lost and inaccurate timesteps reported with this method
        wandb_run = wandb.init(project=proj_name, name=run_name, id=old_run_id, resume="must")
    
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