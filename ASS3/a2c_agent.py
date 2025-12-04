# a2c_agent.py  (replace existing file)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from neural_networks import A2CNetwork

class A2CAgent:
    def __init__(self, state_dim, action_dim, config, is_continuous,
                 action_space=None, device=None):
        """
        config keys:
          - learning_rate
          - gamma
          - entropy_coef
          - value_coef
          - max_grad_norm
          - n_steps
          - hidden_dim (optional)
          - normalize_adv (optional, default True)
        """
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 3e-4)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.n_steps = config.get("n_steps", 5)
        self.normalize_adv = config.get("normalize_adv", True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        

        self.network = A2CNetwork(state_dim, action_dim, is_continuous,
                                    hidden_dim=config.get("hidden_dim", 256)).to(self.device)

        self.is_continuous = is_continuous
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # action scaling (for Box action spaces, e.g. Pendulum)
        if action_space is not None and hasattr(action_space, "high"):
            self.action_scale = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)
        else:
            self.action_scale = None

        # rollout buffers (store up to n_steps until update)
        self.clear_memory()

    # -----------------------
    # Select action
    # -----------------------
    def select_action(self, state, deterministic=False):
        """
        state: numpy array
        deterministic: eval mode (use mean)
        returns: action in env format (float, array, or int)
        Also stores state, action, log_prob, value, entropy
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, S)
        dist, value = self.network(state_t)  # dist: torch.distribution, value: (1,)

        if self.is_continuous:
            # dist.mean and dist.rsample shapes: (1, action_dim)
            if deterministic:
                raw_action = dist.mean
            else:
                raw_action = dist.rsample()

            action_t = torch.tanh(raw_action)  # squashed to (-1,1)
            if self.action_scale is not None:
                env_action_t = action_t * self.action_scale  # scale to env bounds
            else:
                env_action_t = action_t

            # log_prob of raw_action (before tanh). For simple approach we use dist.log_prob(raw_action)
            # sum over dims to get single scalar per sample
            log_prob = dist.log_prob(raw_action).sum(dim=-1)  # shape (1,)
            entropy = dist.entropy().sum(dim=-1)  # shape (1,)

            action_out = env_action_t.squeeze(0).detach().cpu().numpy()
            # store action as tensor on device (same shape used for potential continuous training)
            store_action = env_action_t.detach()  # shape (1, A)
        else:
            # discrete
            if deterministic:
                probs = F.softmax(dist.logits, dim=-1)
                action_t = torch.argmax(probs, dim=-1)  # shape (1,)
            else:
                action_t = dist.sample()  # shape (1,)
            log_prob = dist.log_prob(action_t).squeeze(-1)  # shape (1,)
            entropy = dist.entropy().squeeze(-1)  # shape (1,)

            action_out = int(action_t.item())
            store_action = action_t.detach()

        # store rollout elements
        self.states.append(state_t)            # list of (1, S)
        self.actions.append(store_action)      # list of tensors (1, A) or (1,)
        self.log_probs.append(log_prob)       # list of (1,)
        self.values.append(value)             # list of (1,)
        self.entropies.append(entropy)        # list of (1,)

        return action_out

    # -----------------------
    # store reward
    # -----------------------
    def store_reward(self, reward, done):
        # reward: scalar from env
        self.rewards.append(float(reward))
        self.masks.append(0.0 if done else 1.0)

    # -----------------------
    # compute n-step returns
    # -----------------------
    def compute_nstep_returns(self, last_value):
        """
        Compute bootstrapped returns for stored rewards:
            returns[t] = r_t + gamma * r_{t+1} + ... + gamma^{T-t-1} * last_value
        where last_value = V(s_{t+T}) if mask==1 else 0
        Works for T = len(self.rewards) (normally T == n_steps)
        """
        T = len(self.rewards)
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        running_return = last_value.detach() if isinstance(last_value, torch.Tensor) else torch.tensor(float(last_value), device=self.device)

        # iterate backwards
        for t in reversed(range(T)):
            r = torch.tensor(self.rewards[t], dtype=torch.float32, device=self.device)
            mask = torch.tensor(self.masks[t], dtype=torch.float32, device=self.device)
            running_return = r + self.gamma * running_return * mask
            returns[t] = running_return

        values = torch.stack(self.values).squeeze(-1)  # shape (T,)
        advantages = returns - values.detach()

        if self.normalize_adv and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    # -----------------------
    # update (called when you have >= n_steps samples)
    # -----------------------
    def update(self):
        """
        Performs update using the current buffers.
        Note: your training loop should call update() every environment step and only when
        len(self.rewards) >= self.n_steps (or at episode end to flush residual).
        """
        if len(self.rewards) == 0:
            return 0.0

        # If not enough samples, do nothing
        if len(self.rewards) < self.n_steps:
            return 0.0

        # Bootstrapping: compute last_value (value of next state after last stored step)
        with torch.no_grad():
            if self.masks[-1] == 0.0:
                last_value = torch.tensor(0.0, device=self.device)
            else:
                # evaluate last stored state to get bootstrap value
                _, last_value = self.network(self.states[-1])
                last_value = last_value.squeeze(-1)

        returns, advantages = self.compute_nstep_returns(last_value)

        # Stack tensors: each was shape (1,), after stacking we get (T,1) or (T,)
        log_probs = torch.cat(self.log_probs).squeeze(-1)    # (T,)
        values = torch.cat(self.values).squeeze(-1)          # (T,)
        entropies = torch.cat(self.entropies).squeeze(-1)   # (T,)

        # actor/critic/entropy losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # clear memory after update
        self.clear_memory()
        return loss.item()

    # -----------------------
    # clear buffers
    # -----------------------
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.entropies = []

    # -----------------------
    # save/load
    # -----------------------
    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename, map_location=None):
        self.network.load_state_dict(torch.load(filename, map_location=map_location))
