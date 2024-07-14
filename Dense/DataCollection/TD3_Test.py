import sys
import gymnasium  #as gym
import torch
import numpy as np
import random
import pickle
import TD3

# import mujoco_py

#
env_name = 'Ant'
# Load environment
env = gymnasium.make('Ant-v4')
env_eval = gymnasium.make('Ant-v4')
# env = gymnasium.make('HalfCheetah-v4')
# env_eval = gymnasium.make('HalfCheetah-v4')
# Set seeds
seed = 5
offset = 100
env.reset(seed=seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Network and hyperparameters
device = "cuda:0"
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

max_steps = 1e6
memory_size = 1e6  # 5e5 why memory_size is larger than max_step?
step_eval = 1000

batch_size = 256
learning_starts = 10 * batch_size

replay_buffer = []
score_history = []
steps = 0  # Tracks the number of steps taken in the environment.
episodes = 0  # Tracks the number of episodes.
episodes_eval = 10  # The number of episodes to run for evaluation.

agent = TD3.Agent(state_dim, action_dim, max_action, hidden_dim=(256, 256), lr=(1e-3, 1e-3), batch_size=batch_size,
                  policy_noise=0.2, device=device)

while steps < max_steps + 1:
    # Training #
    done = False  # Reset the done flag for the new episode.
    done_trunc = False
    state = env.reset()[0]  # Reset the environment to get the initial state.
    step_env = 0  # Reset the environment step counter for the new episode.
    while not (done or done_trunc):  # Continue the episode until it ends.
        action = agent.choose_action(state)
        noise = np.random.normal(0, max_action * 0.1, size=action_dim)
        action = np.clip(action + noise, -max_action, max_action)
        next_state, reward, done, done_trunc, info = env.step(action)
        steps += 1  # Increment the step counter.
        step_env += 1  # Increment the episode step counter.
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state  # Update the current state.

        # Maintain the replay buffer size by removing the oldest transition if it exceeds the memory size.
        if len(replay_buffer) > memory_size:
            replay_buffer.pop(0)

        if steps >= learning_starts:
            ### Train dynamics model  ###
            agent.train(replay_buffer)

        # Evaluation (every step_eval steps)
        # change the seed to evaluate (evaluate not on the training data) Stability
        # By resetting env_eval with a specific seed, ensure each evaluation episode starts from a consistent state.
        # This makes the evaluation results more reliable and comparable over time.
        # The evaluation environment is not affected by exploration noise or other stochastic elements introduced during
        # training, leading to a more accurate measure of the agent's performance.
        env_eval.reset(seed=seed + offset)
        env_eval.action_space.seed(seed + offset)
        # Evaluate the agent every step_eval steps. ???
        if steps % step_eval == 0:
            score_temp = []
            for e in range(episodes_eval):  # Run evaluation episodes.
                done_eval = False
                done_trunc = False
                state_eval = env_eval.reset()[0]
                score_eval = 0
                while not (done_eval or done_trunc):
                    with torch.no_grad():
                        action_eval = agent.choose_action(state_eval)
                        state_eval, reward_eval, done_eval, done_trunc, info_eval = env_eval.step(action_eval)
                        score_eval += reward_eval  # sum of rewards within the episode
                score_temp.append(score_eval)
            score_eval = np.mean(score_temp)
            score_history.append(score_eval)  # calculate the mean of 10 episodes for each 1000 steps

            print("Episode", episodes, "Env Steps", steps, "Score %.2f" % score_eval)

    episodes += 1

np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/Ant/TD3_S{seed}",score_history)
torch.save(agent.actor.state_dict(),
                                   f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_expert_actor")
torch.save(agent.actor_target.state_dict(),
                                   f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_expert_actortarget")
torch.save(agent.critic.state_dict(),
                                   f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_expert_critic")
torch.save(agent.critic_target.state_dict(),
                                   f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_expert_critictarget")
