"""
Wrapper to discretize continuous action spaces for DQN compatibility.
This allows DQN (which requires discrete actions) to work with continuous action environments.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math


class DiscretizedActionWrapper(gym.ActionWrapper):
    """
    Wrapper that discretizes a continuous action space into discrete actions.
    
    For Pendulum-v1, the action space is continuous in range [-2, 2].
    This wrapper creates n_bins evenly spaced discrete actions.
    """
    
    def __init__(self, env, n_bins=11, use_fine_control=False):
        """
        Args:
            env: The gymnasium environment to wrap
            n_bins: Number of discrete actions to create (default: 11 for Pendulum)
            use_fine_control: If True, use non-uniform bins with more precision near zero
                             (better for tasks requiring fine control like Pendulum)
        """
        super().__init__(env)
        
        # Get the original action space bounds
        if isinstance(env.action_space, spaces.Box):
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_dim = env.action_space.shape[0]
        else:
            raise ValueError("Environment must have a Box action space to discretize")
        
        # Create discrete action space
        self.n_bins = n_bins
        self.action_space = spaces.Discrete(n_bins)
        
        # Create action bins - evenly spaced values between low and high
        # For multi-dimensional actions, create bins for each dimension
        if self.action_dim == 1:
            if use_fine_control:
                # Non-uniform bins: more precision near zero for fine control
                # Use a tanh-like distribution to concentrate bins near zero
                # This is better for Pendulum where small corrections are crucial
                half_bins = n_bins // 2
                if n_bins % 2 == 1:
                    # Odd number: include zero exactly
                    # Create symmetric bins around zero
                    # Use arctanh to create non-uniform spacing
                    x = np.linspace(-0.99, 0.99, half_bins)
                    # Map to action range with more density near zero
                    positive_bins = np.tanh(x) * self.action_high[0]
                    negative_bins = -positive_bins[::-1]
                    self.action_bins = np.concatenate([negative_bins, [0.0], positive_bins])
                else:
                    # Even number: symmetric without exact zero
                    x = np.linspace(-0.99, 0.99, n_bins)
                    self.action_bins = np.tanh(x) * self.action_high[0]
                # Sort to ensure proper ordering
                self.action_bins = np.sort(self.action_bins)
            else:
                # Single dimension: create linear bins (evenly spaced)
                self.action_bins = np.linspace(
                    self.action_low[0], 
                    self.action_high[0], 
                    n_bins
                )
        else:
            # Multi-dimensional: create bins for each dimension
            self.action_bins = []
            for i in range(self.action_dim):
                bins = np.linspace(
                    self.action_low[i],
                    self.action_high[i],
                    n_bins
                )
                self.action_bins.append(bins)
            # For multi-dim, we'd need to create a cartesian product
            # For now, we'll handle 1D case (like Pendulum)
            if self.action_dim > 1:
                raise NotImplementedError(
                    f"Multi-dimensional action discretization not yet implemented. "
                    f"Got action_dim={self.action_dim}"
                )
    
    def action(self, act):
        """
        Convert discrete action to continuous action.
        
        Args:
            act: Discrete action index (0 to n_bins-1)
            
        Returns:
            Continuous action value (as array for compatibility)
        """
        if self.action_dim == 1:
            # Return as array with single element for Pendulum compatibility
            action_value = self.action_bins[act]
            return np.array([action_value], dtype=np.float32)
        else:
            # For multi-dim, would need to map index to combination
            raise NotImplementedError("Multi-dimensional actions not implemented")
    
    def reverse_action(self, action):
        """
        Convert continuous action back to discrete (for debugging).
        Not used by DQN but useful for analysis.
        """
        if self.action_dim == 1:
            # Find closest bin
            idx = np.argmin(np.abs(self.action_bins - action[0]))
            return idx
        else:
            raise NotImplementedError("Multi-dimensional actions not implemented")

