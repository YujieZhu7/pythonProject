import copy
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# process the inputs
clip_range = 5
clip_obs = 200
clip_return = 50
# normalise observations and goals
def process_inputs(o, g, o_mean, o_std, g_mean, g_std):
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std + 1e-6), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std + 1e-6), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm], axis=1)
    return inputs

# Define critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256)):  # no learning rate
        super(Critic, self).__init__()
        # The first critic
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        # The second critic
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], dim=-1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return torch.squeeze(q1, dim=-1), torch.squeeze(q2, dim=-1)

    # why we need to squeeze here? the output is a single value (scalar)

    def Q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return torch.squeeze(q1, dim=-1)
    # why we define Q1 separately?


# Define actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=(256, 256)):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))  # scale the actions by max_action

        return a


class Agent(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, hidden_dim=(256, 256),
                 batch_size=1024, gamma=0.99, tau=0.005, lr=(3e-4, 3e-4),
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cuda:0"):

        self.actor = Actor(state_dim + goal_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        # why we need copy this target? don't need to train again separately to get the target action?
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr[0])

        self.critic = Critic(state_dim + goal_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr[1])

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.device = device
        self.batch_size = batch_size

        self.critic_loss_history = []  # why we need to store the loss history?
        self.actor_loss_history = []

        self.total_it = 0

    def choose_action(self, state):
        with torch.no_grad():  # without keeping track of the gradients
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)  # why reshape to a row vector
            action = self.actor(state)

        return action.cpu().numpy().flatten()  # .cpu().detach().numpy() why need flatten?

    def train(self, replay_buffer, normalizers=(0, 1, 0, 1), iterations=2):  # iteration controls the time of training
        for it in range(iterations):
            self.total_it += 1
            minibatch = random.sample(replay_buffer, self.batch_size)
            state = torch.Tensor(np.array([d[0] for d in minibatch]))
            action = torch.Tensor(np.array([d[1] for d in minibatch])).to(self.device)
            reward = torch.Tensor(np.array([d[2] for d in minibatch])).to(self.device)
            next_state = torch.Tensor(np.array([d[3] for d in minibatch]))
            goal = torch.Tensor(np.array([d[4] for d in minibatch]))
            done = torch.Tensor(np.array([d[5] for d in minibatch])).to(self.device)
            # don't send states and goals to tensor since we will process them and then send to tensor
            input = process_inputs(state, goal, o_mean=normalizers[0], o_std=normalizers[1],
                                   g_mean=normalizers[2], g_std=normalizers[3])
            input = torch.Tensor(input).to(self.device)
            next_input = process_inputs(next_state, goal, o_mean=normalizers[0], o_std=normalizers[1],
                                        g_mean=normalizers[2], g_std=normalizers[3])
            next_input = torch.Tensor(next_input).to(self.device)

            # Critic #
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_input) + noise).clamp(-self.max_action, self.max_action)
                targetQ1, targetQ2 = self.critic_target(next_input, next_action)
                targetQ = reward + (1 - done) * self.gamma * torch.min(targetQ1, targetQ2)

            currentQ1, currentQ2 = self.critic(input, action)

            critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)
            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor #
            if self.total_it % self.policy_freq == 0:
                policy_actions = self.actor(input)
                Q = self.critic.Q1(input, policy_actions)

                actor_loss = -Q.mean()
                self.actor_loss_history.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
