import sys
import gymnasium
import torch
import numpy as np
import random
import pickle

import TD3_Qfilter

env_name = 'Ant'

# Load environment
env = gymnasium.make('Ant-v4')
env_eval = gymnasium.make('Ant-v4')

# Set seeds
seed =5
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
batch_size = 256

max_steps = 1e6
memory_size = 1e6
step_eval = 1000

file_name = f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Data/{env_name}/DemoData_RanNoise0.1.pkl"
open_file = open(file_name, "rb")
dataset = pickle.load(open_file)
open_file.close()
# device = "mps:0"
# Convert D4RL to replay buffer

demos = []
for i in range(len(dataset)):
    demos.append((dataset[i][0], dataset[i][1], dataset[i][2], dataset[i][3], dataset[i][4]))


batch_size = 256
learning_starts = batch_size

replay_buffer = []
score_history = []
percent_accept_demos = []
average_accept_demos=[]
steps = 0
episodes = 0
episodes_eval = 10

# Record wandb metrics
agent = TD3_Qfilter.Agent(state_dim, action_dim, max_action, hidden_dim=(400,300),lmbda1=0.001,
                          lmbda2=1 / 128, lr=(1e-3,1e-3),batch_size_buffer=batch_size,
                          batch_size_demo=100,policy_noise = 0.2, device=device)
# agent2 = TD3_TS_Qfilter.Agent(state_dim, action_dim, max_action, latent_dim = 2*action_dim, hidden_dim=(400,300),lr=(1e-3,1e-3),
#                               batch_size_buffer=batch_size, batch_size_demo=100,policy_noise = 0.2, device=device)


while steps < max_steps + 1:
    # Training #
    done = False
    done_trunc = False
    state = env.reset()[0]
    step_env = 0  # step_env is reset at the start of each trajectory
    while not (done or done_trunc):
        action = agent.choose_action(state)
        noise = np.random.normal(0, max_action*0.1, size=action_dim)
        action = np.clip(action + noise, -max_action, max_action)
        next_state, reward, done, done_trunc, info = env.step(action)
        steps += 1
        step_env += 1
        if step_env == env._max_episode_steps:
            done_rb = False
            print("Max env steps reached")
        else:
            done_rb = done
        replay_buffer.append((state, action, reward, next_state, done_rb))
        state = next_state

        if len(replay_buffer) > memory_size:
            replay_buffer.pop(0)

        if steps >= learning_starts:
            ### Train dynamics model  ###
            # agent.train_buffer(replay_buffer)
            num_accept, num_total = agent.train_demos(replay_buffer, demos)
            percent_accept_demos.append(num_accept / num_total)

        # Evaluation (every step_eval steps)
        env_eval.reset(seed=seed+offset)
        env_eval.action_space.seed(seed+offset)
        if steps % step_eval == 0:
            score_temp = []
            for e in range(episodes_eval):
                done_eval = False
                done_trunc = False
                state_eval = env_eval.reset()[0]
                score_eval = 0
                while not (done_eval or done_trunc):
                    with torch.no_grad():
                        action_eval = agent.choose_action(state_eval)
                        state_eval, reward_eval, done_eval, done_trunc, info_eval = env_eval.step(action_eval)
                        score_eval += reward_eval
                score_temp.append(score_eval)
            score_eval = np.mean(score_temp)
            score_history.append(score_eval)

            print("Episode", episodes, "Env Steps", steps, "Score %.2f" % score_eval)
            if len(percent_accept_demos) == 0:
                print("Acceptance Rate of Demos = 0 ")
            else:
                last_ten_percent_demos = percent_accept_demos[-10:] if len(
                    percent_accept_demos) > 10 else percent_accept_demos
                average_accept_demos.append(np.mean(last_ten_percent_demos))
                print("Acceptance Rate of Demos = %.2f " % (np.mean(last_ten_percent_demos)))

    episodes += 1
np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/QFilter_Rand0.1_S{seed}_score", score_history)
np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/QFilter_Rand0.1_S{seed}_demoaccept", average_accept_demos)
