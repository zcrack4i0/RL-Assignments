import torch
import numpy as np

class ImageReplayBuffer:
    """
    Replay Buffer optimized for image observations.
    Stores images as uint8 to save memory and converts to float32 during sampling.
    """
    def __init__(self, capacity, image_shape, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Store images as uint8 (0-255) to save memory
        self.state = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        # State and next_state should be in range [0, 255] as uint8
        self.state[self.ptr] = state
        self.action[self.ptr] = np.asarray(action, dtype=np.float32)
        self.reward[self.ptr] = np.asarray(reward, dtype=np.float32)
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = np.asarray(done, dtype=np.float32)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # Convert from uint8 [0, 255] to float32 [0, 1] and rearrange to (batch, channels, height, width)
        states = torch.FloatTensor(self.state[ind]).to(self.device) / 255.0
        next_states = torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0
        
        # Rearrange from (batch, H, W, C) to (batch, C, H, W)
        if len(states.shape) == 4:  # Image observations
            states = states.permute(0, 3, 1, 2)
            next_states = next_states.permute(0, 3, 1, 2)
        
        return (
            states,
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            next_states,
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
