import numpy as np
import random
from collections import deque, namedtuple

# Define a named tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    
    This buffer stores past experiences and allows random sampling
    to break correlations between consecutive samples.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether the episode ended
        """
        self.buffer.append(Transition(state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, next_states, rewards, dones)
        """
        # Randomly sample transitions
        transitions = random.sample(self.buffer, batch_size)
        
        # Unzip the transitions
        batch = Transition(*zip(*transitions))
        
        # Convert to numpy arrays
        states = np.array(batch.state)
        actions = np.array(batch.action)
        next_states = np.array(batch.next_state)
        rewards = np.array(batch.reward)
        dones = np.array(batch.done)
        
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

