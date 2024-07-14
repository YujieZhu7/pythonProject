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
# class VectorizedLinear(nn.Module):
#     def __init__(self, in_features, out_features, ensemble_size):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.ensemble_size = ensemble_size
#
#         self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
#         self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # default pytorch init for nn.Linear module
#         for layer in range(self.ensemble_size):
#             nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))
#
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # input: [ensemble_size, batch_size, input_size]
#         # weight: [ensemble_size, input_size, out_size]
#         # out: [ensemble_size, batch_size, out_size]
#         return x @ self.weight + self.bias

# Ensemble_size = num_critics
# class VectorizedCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, num_critics=2, hidden_dim=(256,256), drop_rate = 0.0):
#         super(VectorizedCritic, self).__init__()
#
#         self.l1 = VectorizedLinear(state_dim+action_dim, hidden_dim[0], num_critics)
#         self.l2 = VectorizedLinear(hidden_dim[0], hidden_dim[1], num_critics)
#         self.l3 = VectorizedLinear(hidden_dim[1], 1, num_critics)
#         self.dropout = nn.Dropout(drop_rate)
#         self.num_critics = num_critics
#
#     def forward(self, state, action):
#         state_action = torch.cat([state, action], dim=-1)
#         state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
#
#         q_values = F.relu(self.l1(state_action))
#         q_values = self.dropout(q_values)
#         q_values = F.relu(self.l2(q_values))
#         q_values = self.dropout(q_values)
#         q_values = self.l3(q_values)
#
#         return q_values.squeeze(-1)

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
    def __init__(self, state_dim, goal_dim, action_dim, max_action, hidden_dim=(256, 256), batch_size_demo=128, lr=3e-4,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cuda:0"):

        self.actor = Actor(state_dim + goal_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # self.critic = VectorizedCritic(state_dim + goal_dim, action_dim, ensemble_size, hidden_dim, drop_rate).to(device)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr[1])

        self.state_dim = state_dim
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim
        self.device = device
        self.batch_size_demo = batch_size_demo
        self.critic_loss_history = []
        self.actor_loss_history = []
        # self.BC_buffer_loss_history = []
        # self.BC_demos_loss_history = []
        #
        # self.method = method
        # self.drop_rate = drop_rate

        self.total_it = 0

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def BC(self, demos, normalizers=(0, 1, 0, 1), iterations=2):
        for it in range(iterations):
            self.total_it += 1
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

            # Actor #
            if self.total_it % self.policy_freq == 0:
                # if self.method == "MCDropout":
                #     self.critic = self.critic.eval()  # disable dropout layers
                #     self.critic_target = self.critic_target.eval()

                demos_policy_actions = self.actor(demos_input)
                BC_loss = F.mse_loss(demos_policy_actions, demos_action)
                self.actor_loss_history.append(BC_loss.item())
                self.actor_optimizer.zero_grad()
                BC_loss.backward()
                self.actor_optimizer.step()
