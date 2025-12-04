
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical 
import torch.nn.functional as F

class A2CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, is_continuous, hidden_dim=256):
        super().__init__()
        self.is_continuous = is_continuous

        # Shared encoder
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))

        # Actor & Critic separate heads
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim)
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.actor_fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.critic_fc.weight, gain=nn.init.calculate_gain('relu'))

        # Actor heads
        if is_continuous:
            self.mu_head = nn.Linear(hidden_dim, action_dim)
            self.log_sigma_head = nn.Linear(hidden_dim, action_dim)
            nn.init.zeros_(self.log_sigma_head.bias)
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        a = F.relu(self.actor_fc(x))
        v = F.relu(self.critic_fc(x))
        value = self.value_head(v).squeeze(-1)

        if self.is_continuous:
            mu = self.mu_head(a)
            log_sigma = torch.clamp(self.log_sigma_head(a), min=-20, max=2)
            sigma = log_sigma.exp()
            dist = Normal(mu, sigma)
        else:
            logits = self.action_head(a)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

        return dist, value
