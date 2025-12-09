"""
Training script specifically for Box2D environments:
- LunarLander-v3 (with continuous=True)
- CarRacing-v3 (with continuous=True)
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

def make_env(env_name, render_mode=None, record_video=False, video_folder=None):
    """Create Box2D environment with continuous actions"""
    # Box2D environments with continuous=True
    env = gym.make(env_name, render_mode=render_mode, continuous=True)
    
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

def train_sac_box2d(env_name, config, num_episodes=500, record_video=True, save_best=True):
    """
    Train SAC agent on Box2D environment
    
    Args:
        env_name: Name of the Box2D environment (LunarLander-v3 or CarRacing-v3)
        config: Dictionary with hyperparameters
        num_episodes: Number of training episodes
        record_video: Whether to record videos during training
        save_best: Whether to save the best model
    """
    print(f"\n{'='*70}")
    print(f"Training SAC on {env_name} (Box2D, Continuous Actions)")
    print(f"{'='*70}\n")
    
    # Create environment
    video_folder = f"./videos/sac_{env_name}_lr{config['learning_rate']}_bs{config['batch_size']}"
    env = make_env(env_name, render_mode="rgb_array" if record_video else None, 
                   record_video=record_video, video_folder=video_folder)
    
    # Get environment dimensions
    # Handle different observation space types (vector vs image)
    if len(env.observation_space.shape) == 1:
        # Vector observation (LunarLander-v3)
        state_dim = env.observation_space.shape[0]
        print(f"Vector observation space: {state_dim} dimensions")
    elif len(env.observation_space.shape) == 3:
        # Image observation (CarRacing-v3)
        # Flatten the image for now (or use CNN - but for simplicity, flatten)
        state_dim = np.prod(env.observation_space.shape)
        print(f"Image observation space: {env.observation_space.shape}")
        print(f"Flattened to {state_dim} dimensions")
    else:
        raise ValueError(f"Unsupported observation space shape: {env.observation_space.shape}")
    
    action_dim = env.action_space.shape[0]
    
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")
    print(f"\nHyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize wandb for this training run (with error handling)
    run_name = f"SAC_{env_name}_lr{config['learning_rate']}_bs{config['batch_size']}_gamma{config['gamma']}"
    wandb_config = config.copy()
    wandb_config['env_name'] = env_name
    wandb_config['num_episodes'] = num_episodes
    wandb_config['state_dim'] = state_dim
    wandb_config['action_dim'] = action_dim
    
    use_wandb = True
    try:
        wandb.init(
            project="SAC-Box2D-Training",
            config=wandb_config,
            name=run_name,
            group=f"{env_name}",
            tags=["SAC", "Box2D", env_name],
            reinit=True,
            mode="online"  # Try online first
        )
    except Exception as e:
        print(f"Warning: Failed to initialize wandb online: {e}")
        print("Attempting to use wandb in offline mode...")
        try:
            wandb.init(
                project="SAC-Box2D-Training",
                config=wandb_config,
                name=run_name,
                group=f"{env_name}",
                tags=["SAC", "Box2D", env_name],
                reinit=True,
                mode="offline"  # Fallback to offline mode
            )
            print("WandB initialized in offline mode. Run 'wandb sync' later to upload.")
        except Exception as e2:
            print(f"Warning: Failed to initialize wandb offline: {e2}")
            print("Continuing training without wandb logging...")
            use_wandb = False
    
    # Initialize agent
    agent = SACAgent(state_dim, action_dim, env.action_space, config)
    print(f"\nAgent initialized on device: {agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"WandB run: {run_name}")
    print(f"WandB project: SAC-Box2D-Training\n")
    
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
                # Normalize image to [0, 1] range (CarRacing images are uint8 [0, 255])
                state_normalized = state.astype(np.float32) / 255.0
                state_flat = state_normalized.flatten()
            else:
                state_flat = state
            
            # Select action
            action = agent.select_action(state_flat)
            
            # Clip action to valid range (important for CarRacing)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Flatten next_state if it's an image
            if len(next_state.shape) > 1:
                # Normalize image to [0, 1] range
                next_state_normalized = next_state.astype(np.float32) / 255.0
                next_state_flat = next_state_normalized.flatten()
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
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        if episode_losses:
            training_losses.append(avg_loss)
        
        # Log to wandb every episode
        # Log to wandb every episode (with error handling)
        if use_wandb:
            try:
                log_dict = {
                    "episode": episode + 1,
                    "episode_reward": total_reward,
                    "episode_length": steps,
                    "buffer_size": agent.memory.size,
                    "alpha": agent.alpha.item() if hasattr(agent.alpha, 'item') else agent.alpha
                }
                if avg_loss > 0:
                    log_dict["episode_loss"] = avg_loss
                wandb.log(log_dict)
            except Exception as e:
                if episode % 100 == 0:  # Only print warning occasionally
                    print(f"Warning: wandb.log() failed: {e}. Continuing training...")
                use_wandb = False
        
        # Track best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode + 1
            if save_best:
                model_path = f"saved_models/sac_{env_name}_best.pth"
                agent.save(model_path)
            if use_wandb:
                try:
                    wandb.log({"best_reward": best_reward, "best_episode": best_episode})
                except:
                    pass  # Silently continue if wandb fails
        
        # Print progress
        if (episode + 1) % 50 == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
            avg_loss_window = np.mean(training_losses[-50:]) if training_losses else 0
            
            # Show action info for debugging (especially for CarRacing)
            if env_name == "CarRacing-v3" and episode < 5:
                # Sample an action to see what's being generated
                sample_state = state_flat if 'state_flat' in locals() else np.zeros(state_dim, dtype=np.float32)
                sample_action = agent.select_action(sample_state)
                print(f"Episode {episode + 1:4d}/{num_episodes} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Length: {steps:4d} | "
                      f"Avg Reward (last 50): {avg_reward:7.2f} | "
                      f"Avg Loss: {avg_loss_window:.4f} | "
                      f"Buffer: {agent.memory.size}")
                print(f"  Sample Action: {sample_action} | Action Space: {env.action_space}")
            else:
                print(f"Episode {episode + 1:4d}/{num_episodes} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Length: {steps:4d} | "
                      f"Avg Reward (last 50): {avg_reward:7.2f} | "
                      f"Avg Loss: {avg_loss_window:.4f} | "
                      f"Buffer: {agent.memory.size}")
            
            # Log rolling averages to wandb
            if use_wandb:
                try:
                    wandb.log({
                        "avg_reward_50": avg_reward,
                        "avg_length_50": avg_length,
                        "avg_loss_50": avg_loss_window
                    })
                except:
                    pass  # Silently continue if wandb fails
    
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
                "final_avg_reward": np.mean(episode_rewards),
                "final_std_reward": np.std(episode_rewards),
                "final_best_reward": best_reward,
                "final_best_episode": best_episode,
                "final_avg_length": np.mean(episode_lengths),
                "final_buffer_size": agent.memory.size
            })
        except:
            pass  # Silently continue if wandb fails
    
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
    
    # Finish wandb run for training
    if use_wandb:
        try:
            wandb.finish()
        except:
            pass  # Silently continue if wandb fails
    
    return agent, stats

def test_agent(env_name, agent, num_episodes=10, record_video=True, config=None):
    """Test trained agent"""
    print(f"\n{'='*70}")
    print(f"Testing SAC agent on {env_name}")
    print(f"{'='*70}\n")
    
    # Initialize wandb for testing (with error handling)
    use_wandb_test = False
    if config is not None:
        test_run_name = f"TEST_{env_name}_lr{config['learning_rate']}_bs{config['batch_size']}"
        try:
            wandb.init(
                project="SAC-Box2D-Testing",
                config={**config, 'env_name': env_name, 'num_test_episodes': num_episodes},
                name=test_run_name,
                group=f"Test_{env_name}",
                tags=["SAC", "Box2D", env_name, "Testing"],
                reinit=True,
                mode="online"
            )
            use_wandb_test = True
        except Exception as e:
            print(f"Warning: Failed to initialize wandb for testing: {e}")
            try:
                wandb.init(
                    project="SAC-Box2D-Testing",
                    config={**config, 'env_name': env_name, 'num_test_episodes': num_episodes},
                    name=test_run_name,
                    group=f"Test_{env_name}",
                    tags=["SAC", "Box2D", env_name, "Testing"],
                    reinit=True,
                    mode="offline"
                )
                use_wandb_test = True
            except:
                print("Continuing testing without wandb logging...")
                use_wandb_test = False
    
    test_env = make_env(env_name, render_mode="rgb_array", 
                       record_video=record_video,
                       video_folder=f"./videos/sac_{env_name}_testing")
    
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
                # Normalize image to [0, 1] range
                state_normalized = state.astype(np.float32) / 255.0
                state_flat = state_normalized.flatten()
            else:
                state_flat = state
            
            action = agent.select_action(state_flat, evaluate=True)  # Deterministic
            # Clip action to valid range
            action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        print(f"Test Episode {test_ep + 1:3d}: Reward = {total_reward:7.2f}, Length = {steps:4d}")
        
        # Log each test episode to wandb
        if use_wandb_test:
            try:
                wandb.log({
                    "test_episode": test_ep + 1,
                    "test_reward": total_reward,
                    "test_length": steps
                })
            except:
                pass  # Silently continue if wandb fails
    
    print("-" * 70)
    print(f"Test Results:")
    print(f"  Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Best Reward: {np.max(test_rewards):.2f}")
    print(f"  Average Length: {np.mean(test_lengths):.2f}")
    
    # Log final test statistics to wandb
    if use_wandb_test:
        try:
            wandb.log({
                "test_avg_reward": np.mean(test_rewards),
                "test_std_reward": np.std(test_rewards),
                "test_best_reward": np.max(test_rewards),
                "test_avg_length": np.mean(test_lengths)
            })
            wandb.finish()
        except:
            try:
                wandb.finish()
            except:
                pass  # Silently continue if wandb fails
    
    test_env.close()
    
    return test_rewards

def main():
    """Main training function for Box2D environments"""
    
    # Box2D environments with continuous actions
    # environments = ["LunarLander-v3", "CarRacing-v3"]
    environments = ["CarRacing-v3"]  # Only run CarRacing-v3
    
    # Hyperparameter configurations optimized for Box2D environments
    # 5 different configurations to test
    # Note: Buffer sizes are reduced for CarRacing-v3 (image observations) to avoid memory issues
    hyperparameter_configs = [
        {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 256,  # Larger batch for image observations
            'buffer_size': 50000  # Reduced for image observations (CarRacing-v3 has 27K dimensions)
        },
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 128,
            'buffer_size': 25000
        },
        {
            'learning_rate': 3e-4,
            'gamma': 0.995,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 256,
            'buffer_size': 50000
        },
        {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'tau': 0.01,
            'alpha': 0.2,
            'batch_size': 512,
            'buffer_size': 100000  # Still reasonable with float32
        },
        {
            'learning_rate': 5e-4,
            'gamma': 0.98,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 128,
            'buffer_size': 50000
        }
    ]
    
    # Training parameters
    num_episodes = 1000  # More episodes for complex environments
    record_video = True
    
    # Results storage
    all_results = {}
    
    print("\n" + "="*70)
    print("SAC Training on Box2D Environments (Continuous Actions)")
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
                agent, stats = train_sac_box2d(
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
                    record_video=record_video,
                    config=config
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
    results_path = f"training_logs/box2d_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    import sys
    
    # Create log file with timestamp
    log_filename = f"training_logs/training_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("training_logs", exist_ok=True)
    
    # Open log file and redirect output
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    try:
        # Allow training on a single environment via command line
        if len(sys.argv) > 1:
            env_name = sys.argv[1]
            if env_name not in ["LunarLander-v3", "CarRacing-v3"]:
                print(f"Error: {env_name} is not a supported Box2D environment.")
                print("Supported environments: LunarLander-v3, CarRacing-v3")
                sys.exit(1)
            
            # Single environment training with default config
            config = {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'batch_size': 256,
                'buffer_size': 50000  # Reduced for image observations
            }
            
            print(f"Training on {env_name} only...")
            print(f"Output is being saved to: {log_filename}\n")
            agent, stats = train_sac_box2d(
                env_name=env_name,
                config=config,
                num_episodes=1000,
                record_video=True,
                save_best=True
            )
            
            test_rewards = test_agent(env_name, agent, num_episodes=10, record_video=True, config=config)
        else:
            # Train on all environments
            print(f"Output is being saved to: {log_filename}\n")
            main()
    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print(f"\nTraining output saved to: {log_filename}")

