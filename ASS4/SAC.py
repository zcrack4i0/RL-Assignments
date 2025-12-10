import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from ReplayBuffer import ReplayBuffer
from ImageReplayBuffer import ImageReplayBuffer
from NNArch import QNetwork, GaussianPolicy, CNNQNetwork, CNNGaussianPolicy

class SACAgent:
    def __init__(self, state_dim, action_dim, action_space, config, use_cnn=False, input_channels=3, image_shape=None):
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005) # Soft update parameter
        self.alpha = config.get('alpha', 0.2) # Fixed entropy temperature (or initial)
        self.lr = config.get('learning_rate', 3e-4)
        self.use_cnn = use_cnn
        
        # Explicit GPU device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU")

        # Initialize Critic and Actor based on observation type
        if use_cnn:
            print(f"Using CNN-based networks for image observations ({input_channels} channels)")
            self.critic = CNNQNetwork(input_channels, action_dim, hidden_dim=256).to(self.device)
            self.critic_target = CNNQNetwork(input_channels, action_dim, hidden_dim=256).to(self.device)
            self.policy = CNNGaussianPolicy(input_channels, action_dim, hidden_dim=256, action_space=action_space).to(self.device)
        else:
            print(f"Using MLP-based networks for vector observations ({state_dim} dims)")
            self.critic = QNetwork(state_dim, action_dim, hidden_dim=256).to(self.device)
            self.critic_target = QNetwork(state_dim, action_dim, hidden_dim=256).to(self.device)
            self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim=256, action_space=action_space).to(self.device)
        
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Automatic Entropy Tuning (Optional but standard in modern SAC)
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        
        # Initialize Replay Buffer based on observation type
        if use_cnn and image_shape is not None:
            print(f"Using ImageReplayBuffer with shape {image_shape}")
            self.memory = ImageReplayBuffer(config.get('buffer_size', 100000), image_shape, action_dim)
        else:
            self.memory = ReplayBuffer(config.get('buffer_size', 100000), state_dim, action_dim)
    def save(self, filename):
        # Save both Actor and Critic for SAC
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha
        }, filename)
    def select_action(self, state, evaluate=False):
        # Handle both vector and image observations
        if self.use_cnn:
            # State should be (H, W, C) - convert to (1, C, H, W)
            if isinstance(state, np.ndarray):
                # Convert from uint8 [0, 255] to float32 [0, 1]
                state_normalized = state.astype(np.float32) / 255.0
                # Rearrange from (H, W, C) to (C, H, W) and add batch dimension
                state_tensor = torch.FloatTensor(state_normalized).to(self.device).permute(2, 0, 1).unsqueeze(0)
            else:
                state_tensor = state.unsqueeze(0) if len(state.shape) == 3 else state
        else:
            # Vector observation
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            _, _, mean = self.policy.sample(state_tensor)
            return mean.detach().cpu().numpy()[0]
        else:
            action, _, _ = self.policy.sample(state_tensor)
            return action.detach().cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        # Helper to push to memory
        self.memory.add(state, action, reward, next_state, done)

    def update(self, batch_size):
        if self.memory.size < batch_size:
            return 0  # Not enough samples yet

        # 1. Sample from buffer
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size)

        # 2. Update Critic (Q-functions)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)  
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value) 
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # 3. Update Actor (Policy)
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # 4. Update Alpha (Temperature)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        # 5. Soft Update Target Networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return qf_loss.item()