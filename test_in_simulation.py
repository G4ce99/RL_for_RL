if __name__ == "__main__":
    import time
    from itertools import chain

    from my_env import build_rlgym_v2_env, build_env
    from rlgym_ppo import Learner
    from rlgym_ppo.ppo import PPOLearner

    # 32 processes
    n_proc = 1 # Trying this as well
    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    agent_model = Learner(build_rlgym_v2_env,
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
                        log_to_wandb=False, 
                        load_wandb=False).ppo_learner

    agent_model.load_from("./data/Test2_IncreaseScale")

    env = build_env()

    render = True

    while True:
        obs_dict = env.reset()
        steps = 0
        ep_reward = {agent_id: 0 for agent_id in env.agents}
        t0 = time.time()
        while True:
            if render:
                env.render()
                time.sleep(6/120)
            obs = env.obs_builder.build_obs(env.agents, env.state, env.shared_info)
            actions = {}
            for agent_id, action_space in env.action_spaces.items():
                # agent.act(obs) | Your agent should go here
                actions[agent_id], action_prob = agent_model.policy.get_action(obs[agent_id])

            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)

            steps += 1
            for agent_id, reward in reward_dict.items():
                ep_reward[agent_id] += reward

            if any(chain(terminated_dict.values(), truncated_dict.values())):
                break

        ep_time = time.time() - t0
        print("Steps per second: {:.0f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(
            steps / ep_time, ep_time, max(ep_reward.values())))