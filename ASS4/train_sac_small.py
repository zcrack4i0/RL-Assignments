"""
Small training file to test SAC agent implementation.
This is a quick test with minimal episodes to verify everything works.
"""
import os
import sys
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from SAC import SACAgent
from DiscreteToContinuousWrapper import DiscreteToContinuousWrapper
from datetime import datetime

# Ensure directories exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

def make_env(env_name, render_mode=None, record_video=False, continuous=None):
    """Create environment with optional video recording"""
    # Handle Box2D environments with continuous parameter
    if env_name in ["LunarLander-v3", "LunarLander-v2"]:
        if continuous is None:
            continuous = True  # Default to continuous for Box2D
        env = gym.make(env_name, render_mode=render_mode, continuous=continuous)
    elif env_name in ["CarRacing-v3", "CarRacing-v2"]:
        if continuous is None:
            continuous = True  # Default to continuous for Box2D
        env = gym.make(env_name, render_mode=render_mode, continuous=continuous)
    else:
        env = gym.make(env_name, render_mode=render_mode)
        # Apply wrapper for discrete environments (except continuous ones)
        if env_name not in ["Pendulum-v1"] and not isinstance(env.action_space, gym.spaces.Box):
            env = DiscreteToContinuousWrapper(env)
    
    # Record video during training
    if record_video:
        video_folder = f"./videos/sac_{env_name}_training"
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda x: x % 10 == 0  # Record every 10th episode
        )
    
    return env

def train_sac_small(env_name="CartPole-v1", num_episodes=50, record_video=True):
    """
    Small training run to test SAC implementation
    
    Args:
        env_name: Name of the gymnasium environment
        num_episodes: Number of training episodes (small for testing)
        record_video: Whether to record videos during training
    """
    print(f"\n{'='*60}")
    print(f"Training SAC on {env_name} (Small Test Run)")
    print(f"{'='*60}\n")
    
    # Configuration for small test
    config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'batch_size': 64,
        'buffer_size': 10000
    }
    
    # Create environment
    continuous = True if env_name in ["LunarLander-v3", "CarRacing-v3"] else None
    env = make_env(env_name, render_mode="rgb_array" if record_video else None, 
                   record_video=record_video, continuous=continuous)
    
    # Get environment dimensions
    # Handle different observation space types (vector vs image)
    if len(env.observation_space.shape) == 1:
        # Vector observation
        state_dim = env.observation_space.shape[0]
    elif len(env.observation_space.shape) == 3:
        # Image observation (e.g., CarRacing-v3)
        # Flatten the image for now (or use CNN - but for simplicity, flatten)
        state_dim = np.prod(env.observation_space.shape)
        print(f"Warning: Image observation space detected. Flattening to {state_dim} dimensions.")
    else:
        raise ValueError(f"Unsupported observation space shape: {env.observation_space.shape}")
    
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize agent
    agent = SACAgent(state_dim, action_dim, env.action_space, config)
    print(f"Agent initialized on device: {agent.device}\n")
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    
    print("Starting training...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Select action
            # Flatten state if it's an image
            if len(state.shape) > 1:
                state_flat = state.flatten()
            else:
                state_flat = state
            
            action = agent.select_action(state_flat)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Flatten next_state if it's an image
            if len(next_state.shape) > 1:
                next_state_flat = next_state.flatten()
            else:
                next_state_flat = next_state
            
            # Store transition
            agent.store_transition(state_flat, action, reward, next_state_flat, done)
            
            state = next_state_flat
            
            # Update agent if we have enough samples
            if agent.memory.size > config['batch_size']:
                loss = agent.update(config['batch_size'])
            
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1:3d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Length: {steps:4d} | "
                  f"Avg Reward (last 10): {avg_reward:7.2f} | "
                  f"Buffer Size: {agent.memory.size}")
    
    # Final statistics
    print("-" * 60)
    print(f"\nTraining Complete!")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
    
    # Save model
    model_path = f"saved_models/sac_{env_name}_small_test.pth"
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Test the trained agent
    print("\n" + "="*60)
    print("Testing trained agent (5 episodes)...")
    print("="*60)
    
    # Use same continuous parameter for test environment
    continuous = True if env_name in ["LunarLander-v3", "CarRacing-v3"] else None
    test_env = make_env(env_name, render_mode="rgb_array", record_video=True, continuous=continuous)
    test_rewards = []
    
    for test_ep in range(5):
        state, _ = test_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Flatten state if it's an image
            if len(state.shape) > 1:
                state_flat = state.flatten()
            else:
                state_flat = state
            
            action = agent.select_action(state_flat, evaluate=True)  # Deterministic
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        test_rewards.append(total_reward)
        print(f"Test Episode {test_ep + 1}: Reward = {total_reward:.2f}")
    
    print(f"\nTest Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    
    env.close()
    test_env.close()
    
    print("\n" + "="*60)
    print("Small test complete! Check videos/ folder for recorded episodes.")
    print("="*60 + "\n")
    
    return agent, episode_rewards

class Tee:
    """Class to write to both console and file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

if __name__ == "__main__":
    # Default to CartPole for quick test, but allow command line argument
    if len(sys.argv) > 1:
        env_name = sys.argv[1]
    else:
        env_name = "CartPole-v1"
    
    # Create log file with timestamp
    log_filename = f"training_logs/training_output_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("training_logs", exist_ok=True)
    
    # Open log file and redirect output
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    try:
        print(f"Output is being saved to: {log_filename}\n")
        # Test on specified environment
        agent, rewards = train_sac_small(
            env_name=env_name,
            num_episodes=50,
            record_video=True
        )
    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print(f"\nTraining output saved to: {log_filename}")

