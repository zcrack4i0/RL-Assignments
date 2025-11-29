import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DiscreteToContinuousWrapper(gym.ActionWrapper):
    """
    Wraps a discrete environment to accept continuous actions.
    Useful for running SAC on discrete environments like CartPole, Acrobot, etc.
    """
    def __init__(self, env):
        super().__init__(env)
        # Ensure the original env has a Discrete action space
        assert isinstance(env.action_space, spaces.Discrete), "Env must have Discrete action space"
        
        self.n_actions = env.action_space.n
        
        # Define a new continuous action space (Box) between -1 and 1
        # We use shape (1,) assuming a single dimension of control
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        """
        Maps continuous action [-1, 1] to discrete action {0, ..., n-1}
        """
        # 1. Clip action to ensure it stays in range
        clipped_action = np.clip(action, -1.0, 1.0)[0]
        
        # 2. Map [-1, 1] to [0, n_actions]
        # Example for 2 actions: [-1, 0) -> 0, [0, 1] -> 1
        # Linear mapping formula: 
        raw_val = (clipped_action + 1) / 2 * self.n_actions
        
        # 3. Floor to get integer index and clip to valid range [0, n-1]
        discrete_action = int(np.clip(np.floor(raw_val), 0, self.n_actions - 1))
        
        return discrete_action