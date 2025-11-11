"""
Test trained agents on 100 episodes per environment and track episode durations.
This script evaluates the performance of trained DQN agents.
"""

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import time
import argparse
import os
from dqn_agent import DQNAgent
from discretize_wrapper import DiscretizedActionWrapper
import wandb

def test_agent(agent, env_name, num_episodes=100, render=False, wandb_log=False, n_bins=11):
    """
    Test a trained agent for specified number of episodes.
    
    Args:
        agent: Trained DQN agent
        env_name (str): Name of the environment
        num_episodes (int): Number of test episodes
        render (bool): Whether to render the environment
        wandb_log (bool): Whether to log to wandb
        n_bins (int): Number of discrete action bins for continuous environments
        
    Returns:
        dict: Test statistics including rewards, durations, and success rate
    """
    env = gym.make(env_name, render_mode='human' if render else None)
    
    # Check if action space is continuous and needs discretization
    needs_discretization = isinstance(env.action_space, spaces.Box)
    if needs_discretization:
        env = DiscretizedActionWrapper(env, n_bins=n_bins)
        print(f"⚠️  Continuous action space detected. Using discretization with {n_bins} actions.")
    
    episode_rewards = []
    episode_lengths = []
    episode_durations = []
    solved_count = 0
    
    print(f"\n{'='*70}")
    print(f"Testing {env_name} for {num_episodes} episodes")
    print(f"{'='*70}\n")
    
    # Environment-specific solved thresholds
    solved_thresholds = {
        'CartPole-v1': 195,
        'Acrobot-v1': -100,
        'MountainCar-v0': -110,
        'Pendulum-v1': -200,  # Pendulum rewards are negative, lower is better
    }
    threshold = solved_thresholds.get(env_name, None)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        # Start timing
        start_time = time.time()
        
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # End timing
        duration = time.time() - start_time
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_durations.append(duration)
        
        # Check if solved (for environments with thresholds)
        if threshold is not None:
            if env_name == 'CartPole-v1' and episode_reward >= threshold:
                solved_count += 1
            elif env_name in ['Acrobot-v1', 'MountainCar-v0'] and episode_reward >= threshold:
                solved_count += 1
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_duration = np.mean(episode_durations[-10:])
            print(f"Episodes {episode-8}-{episode+1}: Avg Reward = {avg_reward:.2f}, "
                  f"Avg Duration = {avg_duration:.3f}s")
    
    env.close()
    
    # Calculate statistics
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_durations': episode_durations,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_duration': np.mean(episode_durations),
        'std_duration': np.std(episode_durations),
        'min_duration': np.min(episode_durations),
        'max_duration': np.max(episode_durations),
        'total_time': np.sum(episode_durations),
        'solved_count': solved_count,
        'success_rate': solved_count / num_episodes if threshold else None,
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Test Results for {env_name}")
    print(f"{'='*70}")
    print(f"Episodes: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min:  {stats['min_reward']:.2f}")
    print(f"  Max:  {stats['max_reward']:.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean: {stats['mean_length']:.2f} ± {stats['std_length']:.2f} steps")
    print(f"\nEpisode Duration Statistics:")
    print(f"  Mean: {stats['mean_duration']:.3f} ± {stats['std_duration']:.3f} seconds")
    print(f"  Min:  {stats['min_duration']:.3f} seconds")
    print(f"  Max:  {stats['max_duration']:.3f} seconds")
    print(f"  Total: {stats['total_time']:.2f} seconds ({stats['total_time']/60:.2f} minutes)")
    if threshold:
        print(f"\nSuccess Rate: {stats['success_rate']*100:.1f}% ({solved_count}/{num_episodes} episodes ≥ threshold)")
    print(f"{'='*70}\n")
    
    # Log to wandb if requested
    if wandb_log and wandb.run is not None:
        wandb.log({
            'test_mean_reward': stats['mean_reward'],
            'test_std_reward': stats['std_reward'],
            'test_min_reward': stats['min_reward'],
            'test_max_reward': stats['max_reward'],
            'test_mean_length': stats['mean_length'],
            'test_mean_duration': stats['mean_duration'],
            'test_std_duration': stats['std_duration'],
            'test_total_time': stats['total_time'],
            'test_success_rate': stats['success_rate'] if stats['success_rate'] else 0,
        })
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Test trained DQN agents')
    parser.add_argument('--env', type=str, required=True, 
                       help='Environment name (CartPole-v1, Acrobot-v1, MountainCar-v0)')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--num-episodes', type=int, default=100,
                       help='Number of test episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during testing')
    parser.add_argument('--wandb', action='store_true',
                       help='Log results to wandb')
    parser.add_argument('--wandb-project', type=str, default='dqn-testing',
                       help='Wandb project name')
    parser.add_argument('--double-dqn', action='store_true',
                       help='Model was trained with Double DQN')
    parser.add_argument('--n-bins', type=int, default=11,
                       help='Number of discrete action bins for continuous environments')
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f'test_{args.env}',
            tags=['testing', args.env.lower().replace('-', '_')],
            config={
                'environment': args.env,
                'num_episodes': args.num_episodes,
                'model_path': args.model_path,
                'double_dqn': args.double_dqn,
            }
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
        'epsilon_start': 0.0,  # No exploration during testing
        'epsilon_min': 0.0,
        'epsilon_decay': 1.0,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'target_update_freq': 10,
        'buffer_capacity': 10000,
        'hidden_dim': 128,
        'use_double_dqn': args.double_dqn,
    }
    
    # Adjust hidden_dim for specific environments
    if args.env == 'MountainCar-v0':
        config['hidden_dim'] = 256
    elif args.env == 'Pendulum-v1':
        config['hidden_dim'] = 256  # Pendulum may benefit from larger network
    
    # Create agent and load model
    agent = DQNAgent(state_dim, action_dim, config)
    
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        return
    
    agent.load(args.model_path)
    print(f"✅ Model loaded successfully!")
    print(f"Using {'Double DQN' if args.double_dqn else 'Standard DQN'}\n")
    
    # Test the agent
    stats = test_agent(
        agent, 
        args.env, 
        num_episodes=args.num_episodes,
        render=args.render,
        wandb_log=args.wandb,
        n_bins=args.n_bins
    )
    
    # Finish wandb run if active
    if args.wandb:
        wandb.finish()
    
    return stats

if __name__ == '__main__':
    main()

