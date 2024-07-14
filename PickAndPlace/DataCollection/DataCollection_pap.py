import gymnasium as gym
import torch
import numpy as np
import random
import pickle

import TD3_HER as TD3

# do we need to standardise the inputs?
clip_range = 5
clip_obs = 200
clip_return = 50
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, ax=0):
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std + 1e-6), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std + 1e-6), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm], axis=ax)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


# In order to change the mean and variance when a new input is added.
### running mean implementation according to Welford's algorithm
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return (mean, np.ones_like(mean), np.ones_like(mean))
    # np.ones_like: An array of ones with the same shape as the mean, used as a placeholder for the variance.
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)

# Load environment
env_name = 'FetchPickAndPlace'
env = gym.make('FetchPickAndPlace-v2')

# Set seeds
seed = 42
env.reset(seed=seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Network and hyperparameters
device = "cuda:0"
state_dim = env.observation_space['observation'].shape[0]
goal_dim = env.observation_space['desired_goal'].shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

meanvar = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S5_update.npz")
states_agg = (meanvar['state_count'], meanvar['state_mean2'], meanvar['state_M2'])  # (count, mean, M2)
goals_agg = (meanvar['goal_count'], meanvar['goal_mean2'], meanvar['goal_M2'])

# memory_size = 1e6
replay_buffer = []
score_history = []
success_history = []

agent = TD3.Agent(state_dim, goal_dim, action_dim, max_action, hidden_dim=(256,256),
                  gamma=0.98, tau=0.005, lr=(1e-3,1e-3), batch_size=1024,
                  policy_noise = 0.2, noise_clip=0.5, policy_freq=2, device=device)
agent.actor.load_state_dict(torch.load( f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_HER_expert_actor", map_location=device))
agent.actor_target.load_state_dict(torch.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_HER_expert_actortarget", map_location=device))
agent.critic.load_state_dict(torch.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_HER_expert_critic", map_location=device))
agent.critic_target.load_state_dict(torch.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/SaveModels/{env_name}/TD3_HER_expert_critictarget", map_location=device))


rand_prob = 0.1
traj = 0
# dones = []

# store 100 episodes as demonstration data? evaluate each episode
# length of replay buffer is 50 for each trajectory.
while traj < 100:
    done = False
    obs = env.reset()[0]
    state = obs['observation']
    desired_goal = obs['desired_goal']
    # update the initial state and goal
    states_agg = update(states_agg, np.array(state))
    goals_agg = update(goals_agg, np.array(desired_goal))

    score = 0
    while not (done):
        with torch.no_grad():
            # add some random noise to the action
            if random.random() <= rand_prob:
                action = env.action_space.sample()
            else:
                state_stats = finalize(states_agg)  # return (mean, variance, sampleVariance)
                goal_stats = finalize(goals_agg)
                inputs = process_inputs(state, desired_goal, o_mean=state_stats[0], o_std=np.sqrt(state_stats[1]),
                                            g_mean=goal_stats[0], g_std=np.sqrt(goal_stats[1]))
                action = agent.choose_action(inputs)
                var = 0
                noise = np.random.normal(0, max_action * var, size=action_dim)
                action = np.clip(action + noise, -max_action, max_action)
            next_obs, reward, done_rb, done, info = env.step(action)
            next_state = next_obs['observation']
            next_desired_goal = next_obs['desired_goal']
            states_agg = update(states_agg, np.array(next_state))
            goals_agg = update(goals_agg, np.array(desired_goal))
            # dones.append(done_rb)
            replay_buffer.append((state, action, reward, next_state, desired_goal, done_rb))
            state = next_state
            score += reward
    success = info['is_success']
    traj +=1
    score_history.append(score)
    success_history.append(success)

    print("Score", score, "Replay buffer length", len(replay_buffer), "Success rate %.2f" % success)

print("Average score of demonstrations = ", np.mean(score_history))
print("Average success of demonstrations = ", np.mean(success_history))
file_name = f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Data/{env_name}/DemoData_RanNoise0.1.pkl"
open_file = open(file_name, "wb")
pickle.dump(replay_buffer, open_file)
open_file.close()
# print(np.array(dones).sum())

# average score = -8.87
# average success = 0.99