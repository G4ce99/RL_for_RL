import wandb
import pandas as pd
import os
import re
import json

# Returns run_id and timestep of last found checkpoint, throws errors if necessary files/directories not available
def most_recent_checkpoint_metadata(run_name):
    stages_dir = os.path.join("data", "checkpoints", run_name)
    most_recent_stage = None
    for d in os.listdir(stages_dir):
        match = re.match(r"Stage_(\d+)", d)
        if match and os.path.isdir(os.path.join(stages_dir, d)):
            if most_recent_stage is None or most_recent_stage < int(match.group(1)):
                most_recent_stage = int(match.group(1))
            
    if most_recent_stage is None:
        raise AssertionError("RL4RL: No prior stages to recover from")
    
    checkpoints_dir = os.path.join(stages_dir, f"Stage_{most_recent_stage}")
    subdirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.isdigit()]

    if not subdirs:
        raise AssertionError("RL4RL: No prior checkpoints to recover from")
    
    latest_timestep = max(int(d) for d in subdirs)
    checkpoint_json = os.path.join(checkpoints_dir, str(latest_timestep), "BOOK_KEEPING_VARS.json")

    if not os.path.isfile(checkpoint_json):
        raise AssertionError("RL4RL: No BOOK_KEEPING_VARS.json found in latest checkpoint")
    
    with open(checkpoint_json, 'r') as f:
        data = json.load(f)

    return data["wandb_run_id"], data["epoch"], data["cumulative_timesteps"]

# Creates a trimmed copy of a crashed run to correctly enable crash recovery
def create_trimmed_run(project, new_run_name, old_run_id, last_step):
    api = wandb.Api()
    run = api.run(f"{project}/{old_run_id}")
    wandb_run = wandb.init(project=project, name=new_run_name)
    
    for row_data in run.scan_history():
        if row_data["_step"] <= last_step:
            wandb.log(row_data, step=int(row_data["_step"]))

    return wandb_run
