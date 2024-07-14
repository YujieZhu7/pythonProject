import copy
import numpy as np
import random
import math
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

#Define VectorizedLinear instead of original nn.Linear
class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias

# Ensemble_size = num_critics
class VectorizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_critics=2, hidden_dim=(256,256), drop_rate = 0.0):
        super(VectorizedCritic, self).__init__()

        self.l1 = VectorizedLinear(state_dim+action_dim, hidden_dim[0], num_critics)
        self.l2 = VectorizedLinear(hidden_dim[0], hidden_dim[1], num_critics)
        self.l3 = VectorizedLinear(hidden_dim[1], 1, num_critics)
        self.dropout = nn.Dropout(drop_rate)
        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        q_values = F.relu(self.l1(state_action))
        q_values = self.dropout(q_values)
        q_values = F.relu(self.l2(q_values))
        q_values = self.dropout(q_values)
        q_values = self.l3(q_values)

        return q_values.squeeze(-1)

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
                 method="First", ensemble_size = 2, drop_rate = 0.0,
                 lmbda1=1, lmbda2=1, batch_size_buffer=1024, batch_size_demo=128, gamma=0.99, tau=0.005, lr=(3e-4, 3e-4),
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cuda:0"):

        self.actor = Actor(state_dim + goal_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr[0])

        self.critic = VectorizedCritic(state_dim + goal_dim, action_dim, ensemble_size, hidden_dim, drop_rate).to(device)
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
        self.lmbda2 = lmbda2
        self.batch_size_buffer = batch_size_buffer
        self.batch_size_demo = batch_size_demo
        self.critic_loss_history = []
        self.actor_loss_history = []
        # self.value_loss_history = []
        self.BC_buffer_loss_history = []
        self.BC_demos_loss_history = []

        self.method = method
        self.drop_rate = drop_rate

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
            state = np.array([d[0] for d in minibatch])
            action = torch.Tensor(np.array([d[1] for d in minibatch])).to(self.device)
            reward = torch.Tensor(np.array([d[2] for d in minibatch])).to(self.device)
            next_state = np.array([d[3] for d in minibatch])
            goal = np.array([d[4] for d in minibatch])
            done = torch.Tensor(np.array([d[5] for d in minibatch])).to(self.device)

            input = process_inputs(state, goal, o_mean=normalizers[0], o_std=normalizers[1],
                                   g_mean=normalizers[2], g_std=normalizers[3])
            input = torch.Tensor(input).to(self.device)
            next_input = process_inputs(next_state, goal, o_mean=normalizers[0], o_std=normalizers[1],
                                        g_mean=normalizers[2], g_std=normalizers[3])
            next_input = torch.Tensor(next_input).to(self.device)

            # minibatch for demonstrations
            demos_minibatch = random.sample(demos, self.batch_size_demo)
            demos_state = np.array([d[0] for d in demos_minibatch])
            demos_action = torch.Tensor(np.array([d[1] for d in demos_minibatch])).to(self.device)
            demos_reward = torch.Tensor(np.array([d[2] for d in demos_minibatch])).to(self.device)
            demos_next_state = np.array([d[3] for d in demos_minibatch])
            demos_goal = np.array([d[4] for d in demos_minibatch])
            demos_done = torch.Tensor(np.array([d[5] for d in demos_minibatch])).to(self.device)

            demos_input = process_inputs(demos_state, demos_goal, o_mean=normalizers[0], o_std=normalizers[1],
                                   g_mean=normalizers[2], g_std=normalizers[3])
            demos_input = torch.Tensor(demos_input).to(self.device)
            demos_next_input = process_inputs(demos_next_state, demos_goal, o_mean=normalizers[0], o_std=normalizers[1],
                                        g_mean=normalizers[2], g_std=normalizers[3])
            demos_next_input = torch.Tensor(demos_next_input).to(self.device)

            # Critic #
            if self.method == "MCDropout":
                self.critic = self.critic.train()  # allow stochastic forward passes with dropout
                self.critic_target = self.critic_target.train()
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_input) + noise).clamp(-self.max_action, self.max_action)
                target_q = self.critic_target(next_input, next_action)
                # need to sample 2 from target_q, then take the minimum
                # target_q has dim (num_critics, batch_size)
                indices = torch.randperm(target_q.size(0))
                target_q = target_q[indices[:2]]
                target_q = target_q.min(0)[0]
                target_q = reward + (1 - done) * self.gamma * target_q

            current_q = self.critic(input, action)

            with torch.no_grad():
                demos_noise = (torch.randn_like(demos_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                demos_next_action = (self.actor_target(demos_next_input) + demos_noise).clamp(-self.max_action, self.max_action)
                demos_target_q = self.critic_target(demos_next_input, demos_next_action)
                demos_indices = torch.randperm(demos_target_q.size(0))
                demos_target_q = demos_target_q[demos_indices[:2]]
                demos_target_q = demos_target_q.min(0)[0]
                demos_target_q = demos_reward + (1 - demos_done) * self.gamma * demos_target_q

            demos_current_q = self.critic(demos_input, demos_action)

            critic_loss = (F.mse_loss(current_q, target_q)
                           + F.mse_loss(demos_current_q, demos_target_q))
            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor #
            num_accept = 0
            num_total = 0
            if self.total_it % self.policy_freq == 0:
                if self.method == "MCDropout":
                    self.critic = self.critic.eval()  # disable dropout layers
                    self.critic_target = self.critic_target.eval()
                policy_actions = self.actor(input)
                Q = self.critic(input, policy_actions)[0] # select one critic network randomly to evaluate the policy action

                demos_policy_actions = self.actor(demos_input)
                if self.method == "First":  # usual Q-Filter method
                    # take only one pair of Q values to compare from the ensembles
                    demos_Q = self.critic(demos_input, demos_policy_actions)[0]
                    Q_dem = self.critic(demos_input, demos_action)[0]
                if self.method == "Mean":
                    demos_Q_set = self.critic(demos_input, demos_policy_actions)
                    Q_dem_set = self.critic(demos_input, demos_action)
                    demos_Q = torch.mean(demos_Q_set, dim=0)
                    Q_dem = torch.mean(Q_dem_set, dim=0)
                if self.method == "Minimum":
                    demos_Q = self.critic(demos_input, demos_policy_actions).min(0)[0]
                    Q_dem = self.critic(demos_input, demos_action).min(0)[0]
                if self.method == "LCB":
                    demos_Q_set = self.critic(demos_input, demos_policy_actions)
                    Q_dem_set = self.critic(demos_input, demos_action)
                    demos_Q_std = torch.std(demos_Q_set, dim=0)
                    Q_dem_std = torch.std(Q_dem_set, dim=0)
                    demos_Q = torch.mean(demos_Q_set, dim=0) - 2*demos_Q_std
                    Q_dem = torch.mean(Q_dem_set, dim=0) - 2 * Q_dem_std
                if self.method == "ModifiedLCB":
                    demos_Q_set = self.critic(demos_input, demos_policy_actions)
                    Q_dem_set = self.critic(demos_input, demos_action)
                    demos_Q_std = torch.std(demos_Q_set, dim=0)
                    demos_Q = torch.mean(demos_Q_set, dim=0) - 2*demos_Q_std
                    Q_dem = torch.mean(Q_dem_set, dim=0)

                mask = torch.ge(Q_dem, demos_Q).reshape(self.batch_size_demo, 1).repeat(1, self.action_dim)
                num_accept = mask.sum(dim=0)[0].detach().cpu().item()
                num_total = self.batch_size_demo
                BC_loss = F.mse_loss(torch.masked_select(demos_policy_actions, mask), torch.masked_select(demos_action, mask))

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
