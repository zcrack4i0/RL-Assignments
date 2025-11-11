import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture.
    
    This network takes the state as input and outputs Q-values for each action.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the DQN network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(DQN, self).__init__()
        
        # Define the neural network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): State input
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

