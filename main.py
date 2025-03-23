from model import PPO_Model
import wandb
from util import create_trimmed_run, most_recent_checkpoint_metadata

def main():
    proj_name = "RL4RL"
    run_name = "Test3_adaptive_lr"
    use_adaptive_lr = False
    recovering_from_crash = True
    last_cumulative_step = 0

    if wandb.run is not None:
        wandb.finish()

    if not recovering_from_crash:
        #Creates new run from scratch
        wandb_run = wandb.init(project=proj_name, name=run_name)
    elif recovering_from_crash:
        #Creates a clean copy of an old run with ending removed (used when a run was killed due to GPU constraints)
        old_run_id, last_step, last_cumulative_step = most_recent_checkpoint_metadata(run_name)
        wandb_run = create_trimmed_run(proj_name, run_name, old_run_id, last_step)

    n_proc = 24
    model_stages = 4
    rand_spawn_probs = [0.9, 0.5, 0.1, 0.5]
    team_spirits = [0.0, 0.4, 0.8, 0.5]

    # NOTE: THESE MUST BE CUMULATIVE
    timestep_limits = [300_000_000, 700_000_000, 990_500_000, 1_000_000_000] 
    for i in range(model_stages):
        print(f"----------STARTING STAGE: {i}----------")
        if timestep_limits[i] < last_cumulative_step:
            continue
        
        model = PPO_Model(wandb_run=wandb_run,
                          cumulative_timesteps=timestep_limits[i],
                          num_process=n_proc, 
                          team_spirit=team_spirits[i],
                          rand_spawn_prob=rand_spawn_probs[i],
                          learner_params_path=None,
                          learning_stage=i,
                          use_adaptive_lr=use_adaptive_lr,
                          recovering_from_crash=recovering_from_crash)
        model.train()
        recovering_from_crash = False
        
    wandb_run.finish()

if __name__ == "__main__":
    main()