import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from dqn_network import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    """
    DQN Agent that learns to play using Deep Q-Learning.
    
    This agent supports both:
    - Standard DQN: Uses target network for both action selection and evaluation
    - Double DQN (DDQN): Uses policy network for action selection, target network for evaluation
    
    Other features:
    - Experience Replay for breaking correlations
    - Target Network for stability
    - Epsilon-greedy exploration
    """
    
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (dict): Configuration dictionary containing hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)  # Discount factor
        self.epsilon = config.get('epsilon_start', 1.0)  # Exploration rate
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update_freq', 10)
        self.use_double_dqn = config.get('use_double_dqn', False)  # Use Double DQN
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim, config.get('hidden_dim', 128)).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.get('buffer_capacity', 10000))
        
        # Training statistics
        self.step_count = 0
        self.episode_count = 0
        
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training (bool): Whether the agent is in training mode
            
        Returns:
            int: Selected action
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Exploit: choose best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(self, state, action, next_state, reward, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether the episode ended
        """
        self.replay_buffer.push(state, action, next_state, reward, done)
    
    def train(self):
        """
        Train the agent using a batch of experiences from the replay buffer.
        
        Returns:
            float: Loss value (or None if not enough samples)
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample a batch from replay buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy network to select actions, target network to evaluate
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_net(next_states).max(1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Increment step count
        self.step_count += 1
        
        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """
        Save the agent's state.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, filepath)
    
    def load(self, filepath):
        """
        Load the agent's state.
        
        Args:
            filepath (str): Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']

