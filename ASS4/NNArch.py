import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, is_continuous=True):
        super(PolicyNetwork, self).__init__()
        self.is_continuous = is_continuous
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if self.is_continuous:
            # For SAC: Output Mean and Log Std
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)
        else:
            # For A2C/PPO Discrete: Output Logits (Softmax applied later)
            self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.is_continuous:
            mean = self.mean(x)
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, -20, 2) # Stability clamping
            return mean, log_std
        else:
            return F.softmax(self.action_head(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        # For SAC (Q-Network), input is State + Action
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture (Double Q-learning for stability)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        # Action rescaling - register as buffers so they move to device automatically
        if action_space is None:
            action_scale = torch.tensor(1.)
            action_bias = torch.tensor(0.)
        else:
            action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
        
        self.register_buffer('action_scale', action_scale)
        self.register_buffer('action_bias', action_bias)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick (mean + std * epsilon)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Enforcing Action Bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean