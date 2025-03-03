import wandb
import pandas as pd

def create_trimmed_run(project, new_run_name, old_run_id, last_step):
    api = wandb.Api()
    run = api.run(f"{project}/{old_run_id}")
    wandb_run = wandb.init(project=project, name=new_run_name)
    
    history = run.history()
    history = history.sort_values(by="_step")
    
    for index, row in history.iterrows():
        row_data = row.to_dict()
        if row_data["_step"] <= last_step:
            wandb.log(row_data, step=row_data["_step"])

    return wandb_run
