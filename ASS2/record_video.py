import gymnasium as gym
from gymnasium import spaces
import torch
import argparse
import os
from gymnasium.wrappers import RecordVideo
from dqn_agent import DQNAgent
from discretize_wrapper import DiscretizedActionWrapper
import wandb

def record_agent_video(agent, env_name, num_episodes=5, video_folder='videos', wandb_log=False, n_bins=11):
    """
    Record videos of a trained agent playing.
    
    Args:
        agent: Trained DQN agent
        env_name (str): Name of the environment
        num_episodes (int): Number of episodes to record
        video_folder (str): Folder to save videos
        wandb_log (bool): Whether to log videos to wandb
        n_bins (int): Number of discrete action bins for continuous environments
    """
    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Create environment with video recording
    env = gym.make(env_name, render_mode='rgb_array')
    
    # Check if action space is continuous and needs discretization
    needs_discretization = isinstance(env.action_space, spaces.Box)
    if needs_discretization:
        env = DiscretizedActionWrapper(env, n_bins=n_bins)
        print(f"⚠️  Continuous action space detected. Using discretization with {n_bins} actions.")
    
    # Wrap environment to record video
    # Record every episode
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f"{env_name}_dqn"
    )
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Recording {num_episodes} episodes to {video_folder}...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (greedy, no exploration)
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward}, Length: {episode_length}")
    
    env.close()
    
    # Calculate statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    
    print(f"\nRecording complete!")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Videos saved to: {video_folder}")
    
    # Log to wandb if requested
    if wandb_log and wandb.run is not None:
        # Find all video files in the folder
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            wandb.log({
                "video": wandb.Video(video_path, fps=30, format="mp4"),
                "avg_reward": avg_reward,
                "avg_length": avg_length
            })
        
        print("Videos logged to wandb")
    
    return episode_rewards, episode_lengths

def main():
    parser = argparse.ArgumentParser(description='Record videos of trained DQN agent')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gymnasium environment name')
    parser.add_argument('--num-episodes', type=int, default=5, help='Number of episodes to record')
    parser.add_argument('--video-folder', type=str, default='videos', help='Folder to save videos')
    parser.add_argument('--wandb-log', action='store_true', help='Log videos to wandb')
    parser.add_argument('--wandb-project', type=str, default='dqn-rl-assignment', help='Wandb project name')
    parser.add_argument('--double-dqn', action='store_true', help='Model was trained with Double DQN')
    parser.add_argument('--n-bins', type=int, default=11,
                       help='Number of discrete action bins for continuous environments')
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            name='video_recording',
            tags=['evaluation', 'video']
        )
    
    # Create environment to get dimensions
    env = gym.make(args.env)
    
    # Check if action space is continuous and needs discretization
    needs_discretization = isinstance(env.action_space, spaces.Box)
    if needs_discretization:
        env = DiscretizedActionWrapper(env, n_bins=args.n_bins)
        print(f"⚠️  Continuous action space detected. Using discretization with {args.n_bins} actions.")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Configuration for agent (must match training config)
    config = {
        'gamma': 0.99,
        'epsilon_start': 0.0,  # No exploration during recording
        'epsilon_min': 0.0,
        'epsilon_decay': 1.0,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'target_update_freq': 10,
        'buffer_capacity': 10000,
        'hidden_dim': 128,
        'use_double_dqn': args.double_dqn  # Match training configuration
    }
    
    # Adjust hidden_dim for specific environments
    if args.env == 'MountainCar-v0':
        config['hidden_dim'] = 256
    elif args.env == 'Pendulum-v1':
        config['hidden_dim'] = 256  # Pendulum may benefit from larger network
    
    # Create agent and load model
    agent = DQNAgent(state_dim, action_dim, config)
    
    print(f"Using {'Double DQN (DDQN)' if args.double_dqn else 'Standard DQN'}")
    
    print(f"Loading model from {args.model_path}...")
    agent.load(args.model_path)
    print("Model loaded successfully!")
    
    # Record videos
    record_agent_video(
        agent, 
        args.env, 
        num_episodes=args.num_episodes,
        video_folder=args.video_folder,
        wandb_log=args.wandb_log,
        n_bins=args.n_bins
    )
    
    # Finish wandb run if active
    if args.wandb_log:
        wandb.finish()

if __name__ == '__main__':
    main()

