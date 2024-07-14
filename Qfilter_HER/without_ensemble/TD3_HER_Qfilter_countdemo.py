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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256)):
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

    def Q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return torch.squeeze(q1, dim=-1)


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
    def __init__(self, state_dim, goal_dim, action_dim, max_action, hidden_dim=(256, 256), lmbda1=1, lmbda2=1,
                 batch_size_buffer=1024, batch_size_demo=128, gamma=0.99, tau=0.005, lr=(3e-4, 3e-4),
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cuda:0"):

        self.actor = Actor(state_dim + goal_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr[0])

        self.critic = Critic(state_dim + goal_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr[1])

        self.state_dim = state_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim
        self.device = device
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2  # how to define lamda1 and lamda2
        self.batch_size_buffer = batch_size_buffer
        self.batch_size_demo = batch_size_demo
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.value_loss_history = []
        self.BC_buffer_loss_history = []
        self.BC_demos_loss_history = []

        self.total_it = 0

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def train(self, replay_buffer, demos, normalizers=(0, 1, 0, 1), iterations=2):
        for it in range(iterations):
            self.total_it += 1
            # minibatch for the replay buffer
            minibatch = random.sample(replay_buffer, self.batch_size_buffer)
            state = tuple(d[0] for d in minibatch)
            action = torch.Tensor(tuple(d[1] for d in minibatch)).to(self.device)
            reward = torch.Tensor(tuple(d[2] for d in minibatch)).to(self.device)
            next_state = tuple(d[3] for d in minibatch)
            goal = tuple(d[4] for d in minibatch)
            done = torch.Tensor(tuple(d[5] for d in minibatch)).to(self.device)

            input = process_inputs(state, goal, o_mean=normalizers[0], o_std=normalizers[1],
                                   g_mean=normalizers[2], g_std=normalizers[3])
            input = torch.Tensor(input).to(self.device)
            next_input = process_inputs(next_state, goal, o_mean=normalizers[0], o_std=normalizers[1],
                                        g_mean=normalizers[2], g_std=normalizers[3])
            next_input = torch.Tensor(next_input).to(self.device)

            # minibatch for demonstrations
            demos_minibatch = random.sample(demos, self.batch_size_demo)
            demos_state = tuple(d[0] for d in demos_minibatch)
            demos_action = torch.Tensor(tuple(d[1] for d in demos_minibatch)).to(self.device)
            demos_reward = torch.Tensor(tuple(d[2] for d in demos_minibatch)).to(self.device)
            demos_next_state = tuple(d[3] for d in demos_minibatch)
            demos_goal = tuple(d[4] for d in demos_minibatch)
            demos_done = torch.Tensor(tuple(d[5] for d in demos_minibatch)).to(self.device)

            demos_input = process_inputs(demos_state, demos_goal, o_mean=normalizers[0], o_std=normalizers[1],
                                   g_mean=normalizers[2], g_std=normalizers[3])
            demos_input = torch.Tensor(demos_input).to(self.device)
            demos_next_input = process_inputs(demos_next_state, demos_goal, o_mean=normalizers[0], o_std=normalizers[1],
                                        g_mean=normalizers[2], g_std=normalizers[3])
            demos_next_input = torch.Tensor(demos_next_input).to(self.device)

            # Critic #
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_input) + noise).clamp(-self.max_action, self.max_action)
                target_q1, target_q2 = self.critic_target(next_input, next_action)
                target_q = reward + (1 - done) * self.gamma * torch.min(target_q1, target_q2)

            current_q1, current_q2 = self.critic(input, action)

            with torch.no_grad():
                demos_noise = (torch.randn_like(demos_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                demos_next_action = (self.actor_target(demos_next_input) + demos_noise).clamp(-self.max_action, self.max_action)
                demos_target_q1, demos_target_q2 = self.critic_target(demos_next_input, demos_next_action)
                demos_target_q = demos_reward + (1 - demos_done) * self.gamma * torch.min(demos_target_q1, demos_target_q2)

            demos_current_q1, demos_current_q2 = self.critic(demos_input, demos_action)

            critic_loss = (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                           + F.mse_loss(demos_current_q1, demos_target_q) + F.mse_loss(demos_current_q2, demos_target_q))
            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor #
            num_accept = 0
            num_total = 0
            if self.total_it % self.policy_freq == 0:
                policy_actions = self.actor(input)
                Q = self.critic.Q1(input, policy_actions)

                demos_policy_actions = self.actor(demos_input)
                demos_Q = self.critic.Q1(demos_input, demos_policy_actions)
                Q_dem = self.critic.Q1(demos_input, demos_action)
                mask = torch.ge(Q_dem, demos_Q).reshape(self.batch_size_demo, 1).repeat(1, self.action_dim)
                num_accept = mask.sum(dim=0)[0].detach().cpu().item()
                num_total = self.batch_size_demo
                BC_loss = F.mse_loss(torch.masked_select(demos_policy_actions, mask), torch.masked_select(demos_action, mask))
                # BC_loss = ((torch.ge(Q_dem, demos_Q) * torch.mean((demos_policy_actions - demos_action) ** 2, 1)).sum()
                #            / (torch.ge(Q_dem, demos_Q).sum() + 1e-6))
                # self.BC_demos_loss_history.append(BC_loss.item())
                actor_loss = -self.lmbda1 * Q.mean() + self.lmbda2 * BC_loss
                self.actor_loss_history.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return num_accept, num_total
