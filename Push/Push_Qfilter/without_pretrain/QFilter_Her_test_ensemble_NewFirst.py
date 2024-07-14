import sys
import gymnasium as gym
import torch
import numpy as np
import random
import pickle
import TD3_HER_Qfilter_ensemble_NEW as TD3

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
env_name = 'FetchPush'
env = gym.make('FetchPush-v2')
env_train = gym.make('FetchPush-v2')
env_eval = gym.make('FetchPush-v2')
method = "LCB+Mean"
if method == "MCDropout":
    drop_rate = 0.1
else:
    drop_rate = 0.0
# steps_accept = 0
ensemble_size = 10
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
# var=0.2
open_file = open(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Data/{env_name}/DemoData_test0.5+1.pkl", "rb")
dataset = pickle.load(open_file)
open_file.close()

demos = []
states_agg = (0, np.zeros(state_dim), np.zeros(state_dim))  # (count, mean, M2)
goals_agg = (0, np.zeros(goal_dim), np.zeros(goal_dim))
for i in range(len(dataset)):
    demos.append((dataset[i][0], dataset[i][1], dataset[i][2], dataset[i][3], dataset[i][4], dataset[i][5]))
    states_agg = update(states_agg, np.array(dataset[i][0]))
    goals_agg = update(goals_agg, np.array(dataset[i][4]))

max_steps = 2e6
memory_size = 1e6

batch_size = 1024
learning_starts = 10*batch_size

replay_buffer = []
score_history = []
success_history = []
percent_accept_demos = []
average_accept_demos = []

steps = 0
episodes = 0
episodes_eval = 25 # take the average score of 25 episodes
eps_eval = 10  # Evaluate every 10 episodes

# model_iters = 5
# lambda_BC = 1
# lmbda1, lmbda2 according to the paper
# tau is the rate of target updates
# didn't use L2 regularization with coefficient 0.005 in the critic and actor networks
agent = TD3.Agent(state_dim, goal_dim, action_dim, max_action, hidden_dim=(256, 256), drop_rate=drop_rate,
                  method=method, ensemble_size=ensemble_size, lmbda1=0.001,
                  lmbda2=1 / 128, batch_size_buffer=1024, batch_size_demo=128,
                  gamma=0.98, tau=0.005, lr=(1e-3, 1e-3),
                  policy_noise=0.2, noise_clip=0.5, policy_freq=2, device=device)
# updated during training to monitor the utilization of the demonstration data.

while steps < max_steps + 1:
    # Training #
    done = False
    obs_ep = []
    obs = env_train.reset()[0]
    state = obs['observation']
    desired_goal = obs['desired_goal']
    # update the initial state and goal
    states_agg = update(states_agg, np.array(state))
    goals_agg = update(goals_agg, np.array(desired_goal))

    # interact with the environment to get transitions of one trajectory and save them to the replay buffer
    while not done:
        state_stats = finalize(states_agg)  # return (mean, variance, sampleVariance)
        goal_stats = finalize(goals_agg)
        # goal remains the same during a trajectory. Is it normalised across trajectories?
        # state is normalised both within and between trajectories?
        inputs = process_inputs(state, desired_goal, o_mean=state_stats[0], o_std=np.sqrt(state_stats[1]),
                                g_mean=goal_stats[0], g_std=np.sqrt(goal_stats[1]))
        action = agent.choose_action(inputs)
        noise = np.random.normal(0, max_action * 0.1, size=action_dim)
        action = np.clip(action + noise, -max_action, max_action)
        next_obs, reward, terminated, done, info = env_train.step(action)
        next_state = next_obs['observation']
        next_desired_goal = next_obs['desired_goal']

        steps += 1

        # Update again since we start from finalize
        states_agg = update(states_agg, np.array(next_state))
        goals_agg = update(goals_agg, np.array(desired_goal))
        # save to the replay buffer
        replay_buffer.append((state, action, reward, next_state, desired_goal, terminated))

        if len(replay_buffer) > memory_size:
            replay_buffer.pop(0)

        # what is this? Save these values to get HER
        obs_ep.append((obs, action, next_obs, info))

        obs = next_obs
        state = next_state

    # HER
    # save modified transitions of a trajectory to the replay buffer
    substitute_goal = obs["achieved_goal"].copy()
    for i in range(len(obs_ep)):
        observation, action, next_observation, info = obs_ep[i]
        state = observation['observation']
        states_agg = update(states_agg, np.array(state))
        # replace the desired goal with the substitute goal. goal is unchanged within each trajectory
        goals_agg = update(goals_agg, np.array(substitute_goal))
        obs = np.concatenate([state, substitute_goal])
        next_state = next_observation['observation']
        next_obs = np.concatenate([next_state, substitute_goal])

        substitute_reward = env.unwrapped.compute_reward(observation["achieved_goal"], substitute_goal, info)
        substitute_terminated = env.unwrapped.compute_terminated(observation["achieved_goal"], substitute_goal, info)
        substitute_truncated = env.unwrapped.compute_truncated(observation["achieved_goal"], substitute_goal, info)
        # done_HER = False  # done flag is always false ?what does this do?
        replay_buffer.append((state, action, substitute_reward, next_state, substitute_goal, substitute_terminated))
        if len(replay_buffer) > memory_size:
            replay_buffer.pop(0)

    if len(replay_buffer) > learning_starts:
        state_stats = finalize(states_agg)
        goal_stats = finalize(goals_agg)
        num_accept, num_total = agent.train(replay_buffer, demos,
                                            normalizers=(state_stats[0], np.sqrt(state_stats[1]), goal_stats[0],
                                                         np.sqrt(goal_stats[1])), iterations=2)
        percent_accept_demos.append(num_accept / num_total)

    # Evaluation (every step_eval steps)
    env_eval.reset(seed=seed + offset)
    env_eval.action_space.seed(seed + offset)
    if episodes % eps_eval == 0:
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
        if len(percent_accept_demos) == 0:
            print("Acceptance Rate of Demos = 0 ")
        else:
            last_ten_percent_demos = percent_accept_demos[-10:] if len(
                percent_accept_demos) > 10 else percent_accept_demos
            average_accept_demos.append(np.mean(last_ten_percent_demos))
            print("Acceptance Rate of Demos = %.2f " % (np.mean(last_ten_percent_demos)))

    episodes += 1
np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/{method}/EnsSize_{ensemble_size}_S{seed}_score", score_history)
np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/{method}/EnsSize_{ensemble_size}_S{seed}_success", success_history)
np.save(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/{method}/EnsSize_{ensemble_size}_S{seed}_demoaccept",
        average_accept_demos)

