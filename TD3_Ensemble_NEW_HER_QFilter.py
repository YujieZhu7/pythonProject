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
def process_inputs(o, g, o_mean, o_std, g_mean, g_std):
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std+1e-6), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std+1e-6), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm], axis = 1)
    return inputs


# VectorizedLinear taken from - https://github.com/tinkoff-ai/CORL/blob/main/algorithms/sac_n.py
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

class VectorizedCritic(nn.Module):
    def __init__(self, state_dim,action_dim, num_critics=2, hidden_dim=(256,256,256,256), drop_rate = 0.0):
        super(VectorizedCritic, self).__init__()

        self.l1 = VectorizedLinear(state_dim+action_dim, hidden_dim[0], num_critics)
        self.l2 = VectorizedLinear(hidden_dim[0], hidden_dim[1], num_critics)
        self.l3 = VectorizedLinear(hidden_dim[1], hidden_dim[2], num_critics)
        self.l4 = VectorizedLinear(hidden_dim[2], hidden_dim[3], num_critics)
        self.qs = VectorizedLinear(hidden_dim[3], 1, num_critics)
        self.dropout = nn.Dropout(drop_rate)
        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        q_values = F.relu(self.l1(state_action))
        q_values = self.dropout(q_values)
        q_values = F.relu(self.l2(q_values))
        q_values = self.dropout(q_values)
        q_values = F.relu(self.l3(q_values))
        q_values = self.dropout(q_values)
        q_values = F.relu(self.l4(q_values))
        q_values = self.dropout(q_values)
        q_values = self.qs(q_values)

        return q_values.squeeze(-1)


class VectorizedValue(nn.Module):
    def __init__(self, state_dim, num_critics=2, hidden_dim=(256,256,256,256), drop_rate = 0.0):
        super(VectorizedValue, self).__init__()

        self.l1 = VectorizedLinear(state_dim, hidden_dim[0], num_critics)
        self.l2 = VectorizedLinear(hidden_dim[0], hidden_dim[1], num_critics)
        self.l3 = VectorizedLinear(hidden_dim[1], hidden_dim[2], num_critics)
        self.l4 = VectorizedLinear(hidden_dim[2], hidden_dim[3], num_critics)
        self.vs = VectorizedLinear(hidden_dim[3], 1, num_critics)
        self.dropout = nn.Dropout(drop_rate)
        self.num_critics = num_critics

    def forward(self, state):

        state = state.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        val = F.relu(self.l1(state))
        val = self.dropout(val)
        val = F.relu(self.l2(val))
        val = self.dropout(val)
        val = F.relu(self.l3(val))
        val = self.dropout(val)
        val = F.relu(self.l4(val))
        val = self.dropout(val)
        val = self.vs(val)

        return val.squeeze(-1)

# Define actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=(256,256,256,256)):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.l4 = nn.Linear(hidden_dim[2], hidden_dim[3])
        self.l5 = nn.Linear(hidden_dim[3], action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = F.relu(self.l4(a))
        a = self.max_action * torch.tanh(self.l5(a))

        return a
class Agent(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action,hidden_dim=(256,256,256,256),method = "First" ,ensemble_size = 2,
                 lmbda1=1, lmbda2 = 1, batch_size_buffer=1024, batch_size_demo = 128, gamma=0.99, tau=0.005, lr=(3e-4,3e-4),
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cpu"):

        self.actor = Actor(state_dim+goal_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr[0])

        self.critic = VectorizedCritic(state_dim+goal_dim, action_dim, hidden_dim = hidden_dim, num_critics = ensemble_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr[1])

        self.value = VectorizedValue(state_dim+goal_dim, hidden_dim=hidden_dim, num_critics=ensemble_size).to(device)
        self.value_target = copy.deepcopy(self.value)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr[1])

        self.state_dim = state_dim
        self.input_dim = state_dim + goal_dim
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
        self.value_loss_history = []
        self.actor_loss_history = []
        # self.value_loss_history = []
        self.BC_buffer_loss_history = []
        self.BC_demos_loss_history = []
        #
        # self.critic_loss_history = []
        # self.actor_loss_history = []

        self.method = method
        self.total_it = 0

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def train_buffer(self, replay_buffer, normalizers =(0,1,0,1), iterations=2):
        for it in range(iterations):
            self.total_it += 1
            minibatch = random.sample(replay_buffer, self.batch_size_buffer)
            state = tuple(d[0] for d in minibatch)
            action = torch.Tensor(tuple(d[1] for d in minibatch)).to(self.device)
            reward = torch.Tensor(tuple(d[2] for d in minibatch)).to(self.device)
            next_state = tuple(d[3] for d in minibatch)
            goal = tuple(d[4] for d in minibatch)
            done = torch.Tensor(tuple(d[5] for d in minibatch)).to(self.device)

            input = process_inputs(state, goal, o_mean= normalizers[0], o_std=normalizers[1],
                                   g_mean= normalizers[2], g_std= normalizers[3])
            input = torch.Tensor(input).to(self.device)
            next_input = process_inputs(next_state, goal, o_mean=normalizers[0], o_std=normalizers[1],
                                   g_mean=normalizers[2], g_std=normalizers[3])
            next_input = torch.Tensor(next_input).to(self.device)

            # Critic #
            if self.method == "MCDropout":
                self.critic = self.critic.train()  # allow stochastic forward passes with dropout
                self.critic_target = self.critic_target.train()
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_input) + noise).clamp(-self.max_action, self.max_action)
                target_Q = self.critic_target(next_input, next_action)
                target_Q = target_Q.min(0)[0]
                target_Q = reward + (1 - done) * self.gamma * target_Q

                target_V = self.value_target(next_input)
                target_V = target_V.min(0)[0]
                target_V = reward + (1 - done) * self.gamma * target_V

            Q_vals_all = self.critic(input, action)
            critic_loss = F.mse_loss(Q_vals_all, target_Q)

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            V_vals_all = self.value(input)
            value_loss = F.mse_loss(V_vals_all, target_V)

            self.value_loss_history.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Actor #
            if self.total_it % self.policy_freq == 0:
                if self.method == "MCDropout":
                    self.critic = self.critic.eval()
                    self.critic_target = self.critic_target.eval()
                policy_actions = self.actor(input)
                Q = self.critic(input, policy_actions)[0]
                actor_loss = -Q.mean()
                self.actor_loss_history.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_demos(self, demos, normalizers =(0,1,0,1), iterations=2, BC_only = False):
        for it in range(iterations):
            self.total_it += 1
            minibatch = random.sample(demos, self.batch_size_demo)
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
            if self.method == "TrueDiscounted":
                True_Disc_Ret = torch.Tensor(tuple(d[6] for d in minibatch)).to(self.device)


            # Critic #
            with torch.no_grad():
                if self.method == "MCDropout":
                    self.critic = self.critic.train()  # allow stochastic forward passes with dropout
                    self.critic_target = self.critic_target.train()
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_input) + noise).clamp(-self.max_action, self.max_action)
                target_Q = self.critic_target(next_input, next_action)
                target_Q = target_Q.min(0)[0]
                target_Q = reward + (1 - done) * self.gamma * target_Q

            Q_vals_all = self.critic(input, action)

            critic_loss = F.mse_loss(Q_vals_all, target_Q)
            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            num_accept = 0
            num_total = 0
            # Actor #
            if self.total_it % self.policy_freq == 0:
                policy_actions = self.actor(input)
                if not BC_only: # BC_only = False
                    if self.method == "First":  # usual Q-Filter method
                        Q_dem = self.critic(input, action)[0]  # pickkthe first Q-value (Q-Filter)
                        V = self.value(input)[0]
                        # Q = self.critic(input, policy_actions)[0]  # pick the first Q-value (Q-Filter)
                    elif self.method == "Minimum":
                        Q_dem = self.critic(input, action).min(0)[0]
                        V = self.value(input).min(0)[0]
                        # Q = self.critic(input, policy_actions).min(0)[0]
                    elif self.method == "TrueDiscounted":
                        Q_dem = True_Disc_Ret
                        V = self.value(input).min(0)[0]
                    elif self.method == "MCDropout": #Need to change
                        self.critic = self.critic.train()  # allow stochastic forward passes with dropout
                        # state_repeat = state.reshape(batch_size, self.state_dim, 1).repeat(1,1, 10000)
                        input_repeat = input.repeat(1000, 1)
                        action_repeat = action.repeat(1000, 1)
                        policy_actions_repeat = policy_actions.repeat(1000,1)
                        Q_dem_hat = self.critic(input_repeat, action_repeat)[0].reshape(1000, self.batch_size_demo)
                        Q_dem_hat_std = torch.std(Q_dem_hat, dim=0)
                        Q_hat = self.critic(input_repeat, policy_actions_repeat)[0].reshape(1000, self.batch_size_demo)
                        Q_hat_std = torch.std(Q_hat, dim=0)
                        # Q_dem_hat = np.array([self.critic(state, next_state)[0].detach().cpu().numpy() for _ in range(10000)])
                        # Q_hat = np.array([self.critic(state, cycle_next_state)[0].detach().cpu().numpy() for _ in range(10000)])
                        self.critic = self.critic.eval()  # evaluate all nodes without dropout
                        Q_dem = self.critic(input, action)[0] - 2 * Q_dem_hat_std
                        V = self.critic(input, policy_actions)[0] - 2 * Q_hat_std
                    elif self.method == "Ensemble_unc":
                        Q_dem_hat = self.critic(input, action)
                        Q_dem_hat_std = torch.std(Q_dem_hat, dim=0)

                        Q_hat = self.critic(input, policy_actions)
                        Q_hat_std = torch.std(Q_hat, dim=0)

                        Q_dem = self.critic(input, action)[0] - 2 * Q_dem_hat_std
                        V = self.critic(input, policy_actions)[0] - 2 * Q_hat_std


                    mask = Q_dem.ge(V).reshape(self.batch_size_demo, 1).repeat(1, self.action_dim)
                    num_accept = mask.sum(dim=0)[0].detach().cpu().item()
                    BC_loss = F.mse_loss(torch.masked_select(policy_actions, mask), torch.masked_select(action, mask))
                elif BC_only:
                    num_accept = self.batch_size_demo
                    BC_loss = F.mse_loss(policy_actions, action)
                # BC_loss = (torch.ge(Q_dem, Q) * torch.mean((policy_actions - action) ** 2, 1)).sum() / (
                #             torch.ge(Q_dem, Q).sum() + 1e-6)
                # self.BC_demos_loss_history.append(BC_loss.item())
                Q = self.critic(input, policy_actions)[0]
                actor_loss = -self.lmbda1*Q.mean() + self.lmbda2*BC_loss
                self.actor_loss_history.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                num_total = self.batch_size_demo
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return num_accept, num_total