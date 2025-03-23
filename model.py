# TODO's: 
### MAKE THIS INTO A CLASS CALLED Model for multistage training
### ALLOW SOME PARAMETERS TO BE LOADED FROM THE BOOK_KEEPING_VARS.json FILE INSTEAD OF BEING MANUALLY SET
from rlgym_ppo import Learner
from adapt_lr import AdaptiveLearnerWrapper
import torch
import os

from env import build_custom_env

class PPO_Model:
    def __init__(self, 
                 wandb_run,
                 cumulative_timesteps,
                 num_process=8, 
                 team_spirit=0.0,
                 rand_spawn_prob=0.0,
                 learner_params_path=None, 
                 learning_stage = 0, 
                 recovering_from_crash = False, #NOTE: Only turn on when current stage did not run to completion
                 use_adaptive_lr = False):
        assert wandb_run is not None, "Please provide a valid weights and biases run"
        self.run = wandb_run
        self.team_spirit = team_spirit
        self.rand_spawn_prob = rand_spawn_prob
        self.cumulative_timesteps = cumulative_timesteps
        self.n_proc = num_process
        self.save_weight_ckpt_dir = os.path.join("data", "checkpoints", self.run.name, f"Stage_{learning_stage}")
        os.makedirs(self.save_weight_ckpt_dir, exist_ok=recovering_from_crash)

        if learner_params_path is None:
            self.learner_params = self.default_learner_params()
        else:
            self.learner_params = self.load_learner_params_json(learner_params_path)

        self.model_stage = 0 # In multi-stage learning this is used to recover weights_path
        self.learning_stage = learning_stage
        self.use_adaptive_lr = use_adaptive_lr

        self.load_weights = learning_stage != 0 or recovering_from_crash
        if self.load_weights:
            self.load_weights_dir = self.latest_checkpoint_path(recovering_from_crash)

        
    def train(self):
        from rlgym_ppo.batched_agents import BatchedAgentManager
        import io
        from contextlib import redirect_stdout

        # TODO: SUPPRESS REDUNDANT LOAD BY PASSING IN WEIGHTS DIR AS checkpoint_load_path (ENSURE THIS FIX WORKS WITH OTHER CODE)
        # For some reason it is not double loading on non-recovery runs. Figure out why
        if not self.use_adaptive_lr:
            learner = Learner(self.build_model_env, **self.learner_params)
            if self.load_weights:
                # learner.ppo_learner.load_from(self.load_weights_dir)
                learner.load(self.load_weights_dir, load_wandb=False)
        else: 
            learner_wrapper = AdaptiveLearnerWrapper(self.build_model_env, **self.learner_params)
            learner = learner_wrapper.learner
            if self.load_weights:
                # learner.ppo_learner.load_from(self.load_weights_dir)
                policy_lr, critic_lr, _ = learner_wrapper.initialize_from_run(self.run)
                learner.load(folder_path=self.load_weights_dir, load_wandb=False, new_policy_lr=policy_lr, new_critic_lr=critic_lr)

        # Modified code from AechPro's rlgym-ppo repo's learn/cleanup function to prevent cleanup from finishing the run
        try:
            learner._learn()
            learner.save(learner.agent.cumulative_timesteps)
        except Exception:
            import traceback

            print("\n\nLEARNING LOOP ENCOUNTERED AN ERROR\n")
            traceback.print_exc()

            try:
                learner.save(learner.agent.cumulative_timesteps)
            except:
                print("FAILED TO SAVE ON EXIT")
            finally:
                # We finish the run on any exception
                self.run.finish() 

        finally:
            if type(learner.agent) == BatchedAgentManager:
                learner.agent.cleanup()
            learner.experience_buffer.clear()
        # AechPro's code ends here


    def inference(self):
        # TODO: Maybe find a more elegant way to load weights back in for testing (seems overly complicated at the moment)
        pass

    def build_model_env(self):
        from rlgym_ppo.util import RLGymV2GymWrapper

        return RLGymV2GymWrapper(build_custom_env(self.team_spirit, self.rand_spawn_prob))


    def latest_checkpoint_path(self, recovering_from_crash):
        load_stage = self.learning_stage-1
        if recovering_from_crash:
            load_stage = self.learning_stage
        checkpoints_dir = os.path.join("data", "checkpoints", self.run.name, f"Stage_{load_stage}")
        subdirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.isdigit()]
        
        if not subdirs:
            if recovering_from_crash and load_stage > 0:
                # Attempting recovery from crash between stages
                checkpoints_dir = os.path.join("data", "checkpoints", self.run.name, f"Stage_{load_stage-1}")
                subdirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.isdigit()]
                if not subdirs:
                    raise AssertionError("RL4RL: No prior checkpoints found in recovery folder")
            else:
                raise AssertionError("RL4RL: No prior checkpoints found in necessary folder")
        
        latest_timestep = max(int(d) for d in subdirs)
        latest_checkpoint_path = os.path.join(checkpoints_dir, str(latest_timestep))
        
        return latest_checkpoint_path


    def default_learner_params(self):
        # educated guess - could be slightly higher or lower
        min_inference_size = max(1, int(round(self.n_proc * 0.9)))

        learner_params = {
            "n_proc": self.n_proc,
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
            "save_every_ts": 10_000_000, # save every 10M steps TODO: CHANGE BACK TO 10M
            "timestep_limit": self.cumulative_timesteps, # Train for X steps
            "log_to_wandb": True,
            "wandb_run": self.run,
            "load_wandb": False,
            "render": False, 
            "checkpoints_save_folder": self.save_weight_ckpt_dir, # I JUST ADDED THIS TO TEST
            "add_unix_timestamp": False, # I JUST ADDED THIS TO TEST
            "device": ("cuda" if torch.cuda.is_available() else "cpu")
        }

        return learner_params
    

    def load_learner_params_json(self, learner_params_path):
        # TODO: FIX ME

        # educated guess - could be slightly higher or lower
        min_inference_size = max(1, int(round(self.n_proc * 0.9)))
        
        # LOAD IN SHIT
        learner_params = {} # REPLACE ME

        # Non-predetermined parameters
        learner_params["n_proc"] = self.n_proc
        learner_params["min_inference_size"] = min_inference_size
        learner_params["wandb_run"] = self.run
        learner_params["checkpoints_save_folder"] = self.save_weight_ckpt_dir
        learner_params["device"] = ("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: SET NUMBER OF TIMESTEPS

        return learner_params