"""
Wrapper for Pendulum-v1 to discretize continuous action space
This is necessary because DQN/DDQN only handle discrete actions
"""

import gymnasium as gym
import numpy as np


class DiscretePendulumWrapper(gym.ActionWrapper):
    """
    Wraps Pendulum-v1 to use discrete actions instead of continuous
    
    Pendulum-v1 normally takes continuous actions in [-2, 2]
    This wrapper converts discrete action indices to continuous values
    """
    
    def __init__(self, env, num_actions=5):
        """
        Args:
            env: Pendulum-v1 environment
            num_actions: Number of discrete actions (default: 5)
        """
        super().__init__(env)
        
        self.num_actions = num_actions
        
        # Create discrete action space
        self.action_space = gym.spaces.Discrete(num_actions)
        
        # Map discrete actions to continuous values
        # For 5 actions: [-2, -1, 0, 1, 2]
        self.action_map = np.linspace(-2, 2, num_actions)
        
        print(f"Pendulum wrapped with {num_actions} discrete actions:")
        print(f"Action mapping: {self.action_map}")
    
    def action(self, action):
        """
        Convert discrete action to continuous
        
        Args:
            action: Discrete action index (0 to num_actions-1)
        
        Returns:
            Continuous action value in [-2, 2]
        """
        continuous_action = self.action_map[action]
        return np.array([continuous_action], dtype=np.float32)
    
    def reverse_action(self, action):
        """Convert continuous action back to discrete (for compatibility)"""
        # Find closest discrete action
        idx = np.argmin(np.abs(self.action_map - action[0]))
        return idx


def make_pendulum(num_discrete_actions=5, render_mode=None):
    """
    Create Pendulum-v1 environment with discrete actions
    
    Args:
        num_discrete_actions: Number of discrete actions to use
        render_mode: Render mode for visualization
    
    Returns:
        Wrapped Pendulum environment with discrete actions
    """
    env = gym.make('Pendulum-v1', render_mode=render_mode)
    env = DiscretePendulumWrapper(env, num_actions=num_discrete_actions)
    return env


def test_wrapper():
    """Test the discretization wrapper"""
    print("\n" + "="*60)
    print("Testing Pendulum Discretization Wrapper")
    print("="*60 + "\n")
    
    # Create wrapped environment
    env = make_pendulum(num_discrete_actions=5)
    
    print(f"Original action space: Continuous in [-2, 2]")
    print(f"Wrapped action space: Discrete with {env.action_space.n} actions")
    print(f"Observation space: {env.observation_space}")
    print()
    
    # Test episode
    state, _ = env.reset()
    total_reward = 0
    
    print("Running test episode with random actions...")
    for step in range(50):
        # Sample random discrete action
        action = env.action_space.sample()
        
        # Get continuous action that will be applied
        continuous_action = env.action_map[action]
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step < 5:  # Print first few steps
            print(f"Step {step}: Discrete action={action}, "
                  f"Continuous action={continuous_action:.2f}, Reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nTest episode completed!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Episode length: {step + 1} steps")
    
    env.close()
    
    print("\n" + "="*60)
    print("Wrapper test successful!")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_wrapper()
