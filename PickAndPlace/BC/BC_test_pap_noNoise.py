import sys
import gymnasium as gym
import torch
import numpy as np
import random
import pickle
import BC

# use the ensemble method first, then consider MC dropout

# process the inputs
clip_range = 5
clip_obs = 200
clip_return = 50

# normalise the states and goals, but why we need to clip twice?
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
# why we need sampleVariance here?

########################################################################

# Load environment
env_name = 'FetchPickAndPlace'
env = gym.make('FetchPickAndPlace-v2')
env_train = gym.make('FetchPickAndPlace-v2')
env_eval = gym.make('FetchPickAndPlace-v2')
# method = "First"
# if method == "MCDropout":
#     drop_rate = 0.1
# else:
#     drop_rate = 0.0
# ensemble_size = 10
# Set seeds
seed = 5
offset = 100
env.reset(seed=seed)
env.action_space.seed(seed)
env_train.reset(seed=seed)
env_train.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Network and hyperparameters
device = "cuda:0"
state_dim = env.observation_space['observation'].shape[0]
goal_dim = env.observation_space['desired_goal'].shape[0]
obs_dim = state_dim + goal_dim
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
# var=0.5
open_file = open(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Data/{env_name}/DemoData_noNoise.pkl", "rb")
dataset = pickle.load(open_file)
open_file.close()

demos = []
states_agg = (0, np.zeros(state_dim), np.zeros(state_dim))  # (count, mean, M2)
goals_agg = (0, np.zeros(goal_dim), np.zeros(goal_dim))
for i in range(len(dataset)):
    demos.append((dataset[i][0], dataset[i][1], dataset[i][2], dataset[i][3], dataset[i][4], dataset[i][5]))
    states_agg = update(states_agg, np.array(dataset[i][0]))
    goals_agg = update(goals_agg, np.array(dataset[i][4]))

max_steps = 5e5

score_history = []
success_history = []

steps = 0
episodes = 0
episodes_eval = 25 # take the average score of 25 episodes
# eps_eval = 10  # Evaluate every 10 episodes
step_eval = 500

agent = BC.Agent(state_dim, goal_dim, action_dim, max_action, hidden_dim=(256, 256), batch_size_demo=128,
                  lr=1e-3, policy_noise=0.2, noise_clip=0.5, policy_freq=2, device=device)

while steps < max_steps + 1:
    # Training #
    if steps % 50 == 0:  # in order to have the same learning rate as TD3 online
        state_stats = finalize(states_agg)  # return (mean, variance, sampleVariance) unchanged for BC
        goal_stats = finalize(goals_agg)
        agent.BC(demos, normalizers=(state_stats[0], np.sqrt(state_stats[1]), goal_stats[0],
                                     np.sqrt(goal_stats[1])), iterations=2)
    steps += 1

    # Evaluation (every step_eval steps)
    # do the same thing as evaluate per 10 episodes
    env_eval.reset(seed=seed + offset)
    env_eval.action_space.seed(seed + offset)
    if steps % step_eval == 0:
        score_temp = []
        fin_temp = []
        for e in range(episodes_eval):
            done_eval = False
            obs_eval = env_eval.reset()[0]
            state_eval = obs_eval['observation']
            desired_goal_eval = obs_eval['desired_goal']
            score_eval = 0
            while not done_eval:
                with torch.no_grad():
                    state_stats = finalize(states_agg)
                    goal_stats = finalize(goals_agg)
                    inputs = process_inputs(state_eval, desired_goal_eval, o_mean=state_stats[0],
                                            o_std=np.sqrt(state_stats[1]),
                                            g_mean=goal_stats[0], g_std=np.sqrt(goal_stats[1]))
                    action_eval = agent.choose_action(inputs)
                    obs_eval, reward_eval, terminated_eval, done_eval, info_eval = env_eval.step(action_eval)
                    state_eval = obs_eval['observation']
                    desired_goal_eval = obs_eval['desired_goal']
                    score_eval += reward_eval
            fin_eval = info_eval['is_success']
            score_temp.append(score_eval)
            fin_temp.append(fin_eval)
        score_eval = np.mean(score_temp)
        fin_eval = np.mean(fin_temp)
        score_history.append(score_eval)
        success_history.append(fin_eval)
        print("Episode", episodes, "Env Steps", steps, "Score %.2f" % score_eval, "Success rate %.2f" % fin_eval)

    episodes += 1
np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/noNoise/S{seed}_score_5e5", score_history)
np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/noNoise/S{seed}_success_5e5", success_history)
np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/BC/noNoise/S{seed}_loss_5e5", agent.actor_loss_history)

