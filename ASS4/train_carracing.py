"""
Optimized training script for CarRacing-v3 with CNN-based SAC
"""
import os
import sys
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from SAC import SACAgent
import json
from datetime import datetime
import wandb

# Ensure directories exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("training_logs", exist_ok=True)

class CarRacingWrapper(gym.Wrapper):
    """
    Wrapper for CarRacing-v3 to handle episode termination better
    and add frame skipping for efficiency
    """
    def __init__(self, env, frame_skip=4, neg_reward_threshold=-10):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.neg_reward_threshold = neg_reward_threshold
        self.neg_reward_counter = 0
        
    def reset(self, **kwargs):
        self.neg_reward_counter = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        total_reward = 0
        done = False
        truncated = False
        
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Early stopping if car goes off-track for too long
            if reward < 0:
                self.neg_reward_counter += 1
            else:
                self.neg_reward_counter = 0
            
            if self.neg_reward_counter > self.neg_reward_threshold:
                terminated = True
            
            done = terminated or truncated
            if done:
                break
        
        return obs, total_reward, terminated, truncated, info

def make_env(env_name="CarRacing-v3", render_mode=None, record_video=False, video_folder=None, frame_skip=4):
    """Create CarRacing environment with optimizations"""
    env = gym.make(env_name, render_mode=render_mode, continuous=True, lap_complete_percent=0.95)
    
    # Apply wrapper for frame skipping and better termination
    env = CarRacingWrapper(env, frame_skip=frame_skip)
    
    # Record video during training
    if record_video:
        if video_folder is None:
            video_folder = f"./videos/sac_{env_name}_training"
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda x: x % 50 == 0  # Record every 50th episode
        )
    
    return env

def train_carracing(config, num_episodes=2000, record_video=True, save_best=True):
    """
    Train SAC agent on CarRacing-v3 with CNN
    
    Args:
        config: Dictionary with hyperparameters
        num_episodes: Number of training episodes
        record_video: Whether to record videos during training
        save_best: Whether to save the best model
    """
    env_name = "CarRacing-v3"
    print(f"\n{'='*70}")
    print(f"Training SAC with CNN on {env_name}")
    print(f"{'='*70}\n")
    
    # Create environment
    video_folder = f"./videos/sac_{env_name}_cnn_lr{config['learning_rate']}_bs{config['batch_size']}"
    env = make_env(env_name, render_mode="rgb_array" if record_video else None, 
                   record_video=record_video, video_folder=video_folder, 
                   frame_skip=config.get('frame_skip', 4))
    
    # Get environment dimensions
    image_shape = env.observation_space.shape  # (96, 96, 3)
    action_dim = env.action_space.shape[0]  # 3 (steering, gas, brake)
    
    print(f"Image observation space: {image_shape}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")
    print(f"\nHyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize wandb
    run_name = f"SAC_CNN_{env_name}_lr{config['learning_rate']}_bs{config['batch_size']}"
    wandb_config = config.copy()
    wandb_config['env_name'] = env_name
    wandb_config['num_episodes'] = num_episodes
    wandb_config['image_shape'] = image_shape
    wandb_config['action_dim'] = action_dim
    
    use_wandb = True
    try:
        wandb.init(
            project="SAC-CarRacing-CNN",
            config=wandb_config,
            name=run_name,
            group=env_name,
            tags=["SAC", "CNN", "CarRacing"],
            reinit=True,
            mode="online"
        )
        print(f"WandB initialized successfully")
    except Exception as e:
        print(f"WandB initialization failed: {e}")
        print("Continuing without WandB logging")
        use_wandb = False
    
    # Initialize agent with CNN
    agent = SACAgent(
        state_dim=np.prod(image_shape),  # Not used for CNN but required parameter
        action_dim=action_dim,
        action_space=env.action_space,
        config=config,
        use_cnn=True,
        input_channels=3,
        image_shape=image_shape
    )
    
    print(f"\nAgent initialized on device: {agent.device}")
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"WandB run: {run_name}\n")
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    best_reward = -np.inf
    best_episode = 0
    
    # Warm-up period - collect random samples
    warmup_episodes = config.get('warmup_episodes', 10)
    
    print("Starting training...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_losses = []
        
        # Warm-up phase: use random actions
        use_random = episode < warmup_episodes
        
        while not done:
            # Select action
            if use_random:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # Clip action to valid range
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition (state is already in correct format for ImageReplayBuffer)
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent if we have enough samples and not in warmup
            if not use_random and agent.memory.size > config['batch_size']:
                loss = agent.update(config['batch_size'])
                if loss > 0:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Break if episode is too long
            if steps > config.get('max_steps', 1000):
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        if episode_losses:
            training_losses.append(avg_loss)
        
        # Log to wandb
        if use_wandb:
            try:
                log_dict = {
                    "episode": episode + 1,
                    "episode_reward": total_reward,
                    "episode_length": steps,
                    "avg_loss": avg_loss,
                    "buffer_size": agent.memory.size,
                    "alpha": agent.alpha.item() if torch.is_tensor(agent.alpha) else agent.alpha,
                }
                
                if episode > 0:
                    log_dict["avg_reward_10"] = np.mean(episode_rewards[-10:])
                    log_dict["avg_reward_100"] = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                
                wandb.log(log_dict)
            except Exception as e:
                pass
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode + 1
            if save_best:
                best_model_path = f"saved_models/sac_{env_name}_cnn_best.pth"
                agent.save(best_model_path)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            avg_length_10 = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Avg(10): {avg_reward_10:.2f} | "
                  f"Length: {steps} | "
                  f"Best: {best_reward:.2f} | "
                  f"Buffer: {agent.memory.size}")
    
    # Final statistics
    print("-" * 70)
    print(f"\nTraining Complete!")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Best Reward: {best_reward:.2f} (Episode {best_episode})")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
    print(f"Final Buffer Size: {agent.memory.size}")
    
    # Log final statistics to wandb
    if use_wandb:
        try:
            wandb.log({
                "final/avg_reward": np.mean(episode_rewards),
                "final/std_reward": np.std(episode_rewards),
                "final/best_reward": best_reward,
                "final/best_episode": best_episode,
                "final/avg_length": np.mean(episode_lengths),
                "final/buffer_size": agent.memory.size
            })
            wandb.finish()
        except Exception as e:
            pass
    
    # Save final model
    final_model_path = f"saved_models/sac_{env_name}_cnn_final.pth"
    agent.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    if save_best:
        print(f"Best model saved to: {best_model_path}")
    
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
    
    stats_path = f"training_logs/sac_{env_name}_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Training statistics saved to: {stats_path}")
    
    env.close()
    
    return agent, stats

def test_agent(agent, num_episodes=10, record_video=True):
    """Test trained agent"""
    env_name = "CarRacing-v3"
    print(f"\n{'='*70}")
    print(f"Testing SAC CNN agent on {env_name}")
    print(f"{'='*70}\n")
    
    test_env = make_env(env_name, render_mode="rgb_array", 
                       record_video=record_video,
                       video_folder=f"./videos/sac_{env_name}_cnn_testing",
                       frame_skip=4)
    
    test_rewards = []
    test_lengths = []
    
    for test_ep in range(num_episodes):
        state, _ = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps > 1000:
                break
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        print(f"Test Episode {test_ep + 1}/{num_episodes} | Reward: {total_reward:.2f} | Length: {steps}")
    
    print("-" * 70)
    print(f"Test Results:")
    print(f"  Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Best Reward: {np.max(test_rewards):.2f}")
    print(f"  Average Length: {np.mean(test_lengths):.2f}")
    
    test_env.close()
    
    return test_rewards

def main():
    """Main training function"""
    
    # Optimized hyperparameters for CarRacing-v3 with CNN
    configs = [
        {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 128,
            'buffer_size': 50000,
            'warmup_episodes': 10,
            'frame_skip': 4,
            'max_steps': 1000
        },
        {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.1,
            'batch_size': 256,
            'buffer_size': 100000,
            'warmup_episodes': 20,
            'frame_skip': 4,
            'max_steps': 1000
        },
        {
            'learning_rate': 5e-4,
            'gamma': 0.99,
            'tau': 0.01,
            'alpha': 0.2,
            'batch_size': 64,
            'buffer_size': 50000,
            'warmup_episodes': 15,
            'frame_skip': 4,
            'max_steps': 1000
        }
    ]
    
    print("\n" + "="*70)
    print("SAC Training on CarRacing-v3 with CNN")
    print("="*70)
    
    best_config = None
    best_test_reward = -np.inf
    
    for idx, config in enumerate(configs):
        print(f"\n\n{'#'*70}")
        print(f"# Configuration {idx + 1}/{len(configs)}")
        print(f"{'#'*70}\n")
        
        try:
            # Train agent
            agent, stats = train_carracing(
                config=config,
                num_episodes=2000,
                record_video=True,
                save_best=True
            )
            
            # Test agent
            test_rewards = test_agent(
                agent=agent,
                num_episodes=10,
                record_video=True
            )
            
            avg_test_reward = np.mean(test_rewards)
            
            if avg_test_reward > best_test_reward:
                best_test_reward = avg_test_reward
                best_config = config
                # Save best overall model
                agent.save("saved_models/sac_CarRacing-v3_cnn_best_overall.pth")
            
            print(f"\n✓ Configuration {idx + 1} completed!")
            print(f"  Test Average Reward: {avg_test_reward:.2f}")
            
        except Exception as e:
            print(f"\n✗ Error with configuration {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Best Test Reward: {best_test_reward:.2f}")
    print(f"Best Configuration: {best_config}")
    print("="*70 + "\n")

if __name__ == "__main__":
    import sys
    
    # Create log file with timestamp
    log_filename = f"training_logs/carracing_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Single config training mode
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        config = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 128,
            'buffer_size': 50000,
            'warmup_episodes': 10,
            'frame_skip': 4,
            'max_steps': 1000
        }
        
        print(f"Training with single config. Logs: {log_filename}\n")
        agent, stats = train_carracing(config, num_episodes=2000, record_video=True, save_best=True)
        test_rewards = test_agent(agent, num_episodes=10, record_video=True)
    else:
        print(f"Training with multiple configs. Logs: {log_filename}\n")
        main()
