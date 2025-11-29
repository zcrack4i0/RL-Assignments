import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

class A2CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, is_continuous, hidden_dim=256):
        super(A2CNetwork, self).__init__()
        self.is_continuous = is_continuous
        
        # Shared Feature Extractor
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Actor Head (Policy)
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim)
        if is_continuous:
            self.mu_head = nn.Linear(hidden_dim, action_dim)
            self.sigma_log_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)
            
        # Critic Head (Value)
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        # Critic Stream
        v = F.relu(self.critic_fc(x))
        state_value = self.value_head(v)
        
        # Actor Stream
        a = F.relu(self.actor_fc(x))
        
        if self.is_continuous:
            mu = torch.tanh(self.mu_head(a))
            log_sigma = self.sigma_log_head(a)
            log_sigma = torch.clamp(log_sigma, min=-20, max=2)
            sigma = log_sigma.exp()
            dist = Normal(mu, sigma)
        else:
            logits = self.action_head(a)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
        return dist, state_value
class A2CAgent:
    def __init__(self, state_dim, action_dim, config, is_continuous):
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('learning_rate', 1e-3)
        self.entropy_beta = config.get('entropy_beta', 0.01) # Hyperparam to prevent premature convergence
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = A2CNetwork(state_dim, action_dim, is_continuous).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Temporary buffer for the current batch/episode
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.masks = []
    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        dist, value = self.network(state)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Store for update
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        
        if self.network.is_continuous:
            return action.detach().cpu().numpy()[0] # Return array
        else:
            return action.item() # Return int

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.masks.append(1 - done)

    def update(self):
        # Calculate Returns (Discounted Cumulative Rewards)
        # We process rewards in reverse to calculate the return G_t at each step
        returns = []
        R = 0
        
        # If the episode didn't end, we could bootstrap with the last value, 
        # but for this assignment, a simple Monte Carlo return per batch/episode is usually sufficient.
        
        for r, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = r + self.gamma * R * mask
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.device)
        
        # Normalize returns for stability (optional but recommended)
        if len(returns) > 1:
             returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Convert lists to tensors
        # Note: Depending on action shape, we might need to squeeze/unsqueeze
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()
        entropies = torch.stack(self.entropies).mean() # Mean entropy for the loss
        
        # Advantage = Return - Value (Baseline)
        # We detach returns because they are fixed targets
        advantage = returns - values
        
        # Calculate Losses
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        # Total Loss = Actor + Critic - Entropy * beta
        # We subtract entropy to Maximize it (minimize negative entropy)
        loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropies
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Clear buffers
        self.clear_memory()
        
        return loss.item()

    def clear_memory(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.masks = []