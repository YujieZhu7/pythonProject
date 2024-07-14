import gymnasium as gym
import torch
import numpy as np
import random
import pickle

import TD3

# Load environment
env_name ="Ant"
env = gym.make('Ant-v4')

# Set seeds
seed = 42
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
# batch_size = 100

# memory_size = 1e6
replay_buffer = []
score_history = []
score_sparse_history=[]


agent = TD3.Agent(state_dim, action_dim, max_action,batch_size = 256, hidden_dim=(256,256),lr=(1e-3,1e-3),policy_noise = 0.2, device=device)
agent.actor.load_state_dict(torch.load( f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_expert_actor", map_location=device))
agent.actor_target.load_state_dict(torch.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_expert_actortarget", map_location=device))
agent.critic.load_state_dict(torch.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_expert_critic", map_location=device))
agent.critic_target.load_state_dict(torch.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_expert_critictarget", map_location=device))


rand_prob = 0.1
traj = 0
dones = []
while traj < 10:
    done = False
    done_rb = False
    state = env.reset()[0]
    score = 0
    score_sparse=0
    while not (done or done_rb):
        with torch.no_grad():
            if random.random() <= rand_prob:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)
            next_state, reward, done_rb,done, info = env.step(action)
            if info["x_position"] > 20:
                reward_sparse = 1.0
            else:
                reward_sparse = 0.0
            dones.append(done_rb)
            replay_buffer.append((state, action, reward, next_state, done_rb))
            state = next_state
            score += reward
            score_sparse += reward_sparse
    traj +=1
    score_history.append(score)
    score_sparse_history.append(score_sparse)

    print("Score", score, "Sparse Score", score_sparse,"Replay buffer length", len(replay_buffer))

print("Average score of demonstrations = ", np.mean(score_history), np.mean(score_sparse_history))
file_name = f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Data/{env_name}/DemoData_RanNoise0.1.pkl"
open_file = open(file_name, "wb")
pickle.dump(replay_buffer, open_file)
open_file.close()
print(np.array(dones).sum()) 