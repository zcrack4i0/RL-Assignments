"""
Full training file for SAC agent with various hyperparameters.
Includes video recording and comprehensive training/testing.
"""
import os
import sys
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from SAC import SACAgent
from DiscreteToContinuousWrapper import DiscreteToContinuousWrapper
import json
from datetime import datetime

# Ensure directories exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("training_logs", exist_ok=True)

def make_env(env_name, render_mode=None, record_video=False, video_folder=None, continuous=None):
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
        if video_folder is None:
            video_folder = f"./videos/sac_{env_name}_training"
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda x: x % 20 == 0  # Record every 20th episode
        )
    
    return env

def train_sac(env_name, config, num_episodes=500, record_video=True, save_best=True):
    """
    Train SAC agent on specified environment
    
    Args:
        env_name: Name of the gymnasium environment
        config: Dictionary with hyperparameters
        num_episodes: Number of training episodes
        record_video: Whether to record videos during training
        save_best: Whether to save the best model
    """
    print(f"\n{'='*70}")
    print(f"Training SAC on {env_name}")
    print(f"{'='*70}\n")
    
    # Create environment
    continuous = True if env_name in ["LunarLander-v3", "CarRacing-v3"] else None
    video_folder = f"./videos/sac_{env_name}_lr{config['learning_rate']}_bs{config['batch_size']}"
    env = make_env(env_name, render_mode="rgb_array" if record_video else None, 
                   record_video=record_video, video_folder=video_folder, continuous=continuous)
    
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
    print(f"\nHyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize agent
    agent = SACAgent(state_dim, action_dim, env.action_space, config)
    print(f"Agent initialized on device: {agent.device}\n")
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    best_reward = -np.inf
    best_episode = 0
    
    print("Starting training...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_losses = []
        
        while not done:
            # Flatten state if it's an image
            if len(state.shape) > 1:
                state_flat = state.flatten()
            else:
                state_flat = state
            
            # Select action
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
            
            # Update agent if we have enough samples
            if agent.memory.size > config['batch_size']:
                loss = agent.update(config['batch_size'])
                if loss > 0:
                    episode_losses.append(loss)
            
            state = next_state_flat
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if episode_losses:
            training_losses.append(np.mean(episode_losses))
        
        # Track best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode + 1
            if save_best:
                model_path = f"saved_models/sac_{env_name}_best.pth"
                agent.save(model_path)
        
        # Print progress
        if (episode + 1) % 50 == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
            avg_loss = np.mean(training_losses[-50:]) if training_losses else 0
            print(f"Episode {episode + 1:4d}/{num_episodes} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Length: {steps:4d} | "
                  f"Avg Reward (last 50): {avg_reward:7.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Buffer: {agent.memory.size}")
    
    # Final statistics
    print("-" * 70)
    print(f"\nTraining Complete!")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Best Reward: {best_reward:.2f} (Episode {best_episode})")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
    print(f"Final Buffer Size: {agent.memory.size}")
    
    # Save final model
    final_model_path = f"saved_models/sac_{env_name}_final.pth"
    agent.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    if save_best:
        print(f"Best model saved to: saved_models/sac_{env_name}_best.pth")
    
    # Save training statistics
    stats = {
        'env_name': env_name,
        'config': config,
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_losses': training_losses,
        'best_reward': float(best_reward),
        'best_episode': best_episode,
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards))
    }
    
    stats_path = f"training_logs/sac_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Training statistics saved to: {stats_path}")
    
    env.close()
    
    return agent, stats

def test_agent(env_name, agent, num_episodes=10, record_video=True):
    """Test trained agent"""
    print(f"\n{'='*70}")
    print(f"Testing SAC agent on {env_name}")
    print(f"{'='*70}\n")
    
    continuous = True if env_name in ["LunarLander-v3", "CarRacing-v3"] else None
    test_env = make_env(env_name, render_mode="rgb_array", 
                       record_video=record_video,
                       video_folder=f"./videos/sac_{env_name}_testing",
                       continuous=continuous)
    
    test_rewards = []
    test_lengths = []
    
    for test_ep in range(num_episodes):
        state, _ = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
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
            steps += 1
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        print(f"Test Episode {test_ep + 1:3d}: Reward = {total_reward:7.2f}, Length = {steps:4d}")
    
    print("-" * 70)
    print(f"Test Results:")
    print(f"  Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Best Reward: {np.max(test_rewards):.2f}")
    print(f"  Average Length: {np.mean(test_lengths):.2f}")
    
    test_env.close()
    
    return test_rewards

def main():
    """Main training function with various hyperparameter configurations"""
    
    # Define environments to train on
    # Box2D environments: LunarLander-v3 and CarRacing-v3 (with continuous=True)
    environments = ["LunarLander-v3", "CarRacing-v3"]
    
    # Define hyperparameter configurations to test
    hyperparameter_configs = [
        {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 64,
            'buffer_size': 100000
        },
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 128,
            'buffer_size': 100000
        },
        {
            'learning_rate': 3e-4,
            'gamma': 0.95,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 64,
            'buffer_size': 50000
        },
        {
            'learning_rate': 1e-3,
            'gamma': 0.95,
            'tau': 0.01,
            'alpha': 0.2,
            'batch_size': 128,
            'buffer_size': 100000
        }
    ]
    
    # Training parameters
    num_episodes = 500
    record_video = True
    
    # Results storage
    all_results = {}
    
    print("\n" + "="*70)
    print("SAC Training with Multiple Hyperparameter Configurations")
    print("="*70)
    
    # Train on each environment with each configuration
    for env_name in environments:
        print(f"\n\n{'#'*70}")
        print(f"# Environment: {env_name}")
        print(f"{'#'*70}\n")
        
        env_results = {}
        
        for config_idx, config in enumerate(hyperparameter_configs):
            print(f"\n{'='*70}")
            print(f"Configuration {config_idx + 1}/{len(hyperparameter_configs)}")
            print(f"{'='*70}")
            
            try:
                # Train agent
                agent, stats = train_sac(
                    env_name=env_name,
                    config=config,
                    num_episodes=num_episodes,
                    record_video=record_video,
                    save_best=True
                )
                
                # Test agent
                test_rewards = test_agent(
                    env_name=env_name,
                    agent=agent,
                    num_episodes=10,
                    record_video=record_video
                )
                
                # Store results
                config_key = f"config_{config_idx + 1}"
                env_results[config_key] = {
                    'config': config,
                    'training_stats': stats,
                    'test_rewards': test_rewards,
                    'test_avg_reward': float(np.mean(test_rewards)),
                    'test_std_reward': float(np.std(test_rewards))
                }
                
                print(f"\n✓ Configuration {config_idx + 1} completed successfully!")
                
            except Exception as e:
                print(f"\n✗ Error with configuration {config_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results[env_name] = env_results
    
    # Save all results
    results_path = f"training_logs/all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for env_name, env_results in all_results.items():
        print(f"\n{env_name}:")
        for config_key, result in env_results.items():
            print(f"  {config_key}: Test Avg Reward = {result['test_avg_reward']:.2f} ± {result['test_std_reward']:.2f}")
    
    print("\n" + "="*70)
    print("All training complete! Check videos/ folder for recorded episodes.")
    print("="*70 + "\n")

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
    # Create log file with timestamp
    log_filename = f"training_logs/training_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("training_logs", exist_ok=True)
    
    # Open log file and redirect output
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    try:
        print(f"Output is being saved to: {log_filename}\n")
        main()
    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print(f"\nTraining output saved to: {log_filename}")

