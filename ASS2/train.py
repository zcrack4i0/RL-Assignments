import gymnasium as gym
import numpy as np
import torch
import wandb
import argparse
from dqn_agent import DQNAgent
from gymnasium.wrappers import RecordVideo
from gymnasium import spaces
from discretize_wrapper import DiscretizedActionWrapper
import os

def train_dqn(config):
    """
    Train a DQN agent on a Gymnasium environment.
    
    Args:
        config (dict): Configuration dictionary with hyperparameters
    """
    # Initialize wandb
    tags = ['reinforcement-learning']
    tags.append('double-dqn' if config.get('use_double_dqn') else 'dqn')
    
    run = wandb.init(
        project=config.get('project_name', 'dqn-rl-assignment'),
        config=config,
        name=config.get('run_name', None),
        tags=tags
    )
    
    # Create environment
    env = gym.make(config['env_name'])
    
    # Check if action space is continuous and needs discretization
    needs_discretization = isinstance(env.action_space, spaces.Box)
    if needs_discretization:
        n_bins = config.get('n_bins', 11)  # Default 11 bins for Pendulum
        env = DiscretizedActionWrapper(env, n_bins=n_bins)
        print(f"⚠️  Continuous action space detected. Discretizing into {n_bins} actions.")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: {config['env_name']}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Algorithm: {'Double DQN (DDQN)' if config['use_double_dqn'] else 'Standard DQN'}")
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim, config)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    # Training loop
    for episode in range(config['num_episodes']):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, next_state, reward, done or truncated)
            
            # Train the agent
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        agent.episode_count += 1
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        # Log to wandb
        wandb.log({
            'episode': episode,
            'reward': episode_reward,
            'episode_length': episode_length,
            'epsilon': agent.epsilon,
            'avg_loss': avg_loss,
            'avg_reward_100': np.mean(episode_rewards[-100:]),
        })
        
        # Print progress
        if (episode + 1) % config.get('print_freq', 10) == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{config['num_episodes']} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (100): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # Save model periodically
        if (episode + 1) % config.get('save_freq', 100) == 0:
            save_path = os.path.join(config['save_dir'], f'dqn_episode_{episode + 1}.pth')
            os.makedirs(config['save_dir'], exist_ok=True)
            agent.save(save_path)
            print(f"Model saved to {save_path}")
        
        # Check if solved (for CartPole, avg reward > 195 over 100 episodes)
        if len(episode_rewards) >= 100:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            if avg_reward_100 >= config.get('solved_threshold', 195):
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                print(f"Average reward over last 100 episodes: {avg_reward_100:.2f}")
                break
    
    # Save final model
    final_save_path = os.path.join(config['save_dir'], 'dqn_final.pth')
    os.makedirs(config['save_dir'], exist_ok=True)
    agent.save(final_save_path)
    print(f"\nFinal model saved to {final_save_path}")
    
    # Close environment
    env.close()
    
    # Finish wandb run
    wandb.finish()
    
    return agent

def evaluate_agent(agent, env_name, num_episodes=10, render=False, n_bins=11):
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained DQN agent
        env_name (str): Name of the environment
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment
        n_bins (int): Number of discrete action bins for continuous environments
        
    Returns:
        float: Average reward over evaluation episodes
    """
    env = gym.make(env_name, render_mode='human' if render else None)
    
    # Check if action space is continuous and needs discretization
    needs_discretization = isinstance(env.action_space, spaces.Box)
    if needs_discretization:
        env = DiscretizedActionWrapper(env, n_bins=n_bins)
        print(f"⚠️  Continuous action space detected. Using discretization with {n_bins} actions.")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return avg_reward

def main():
    parser = argparse.ArgumentParser(description='Train DQN on Gymnasium environment')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gymnasium environment name')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--target-update', type=int, default=10, help='Target network update frequency')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--wandb-project', type=str, default='dqn-rl-assignment', help='Wandb project name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--double-dqn', action='store_true', help='Use Double DQN instead of standard DQN')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained agent')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model for evaluation')
    parser.add_argument('--n-bins', type=int, default=11, help='Number of discrete action bins for continuous environments')
    
    args = parser.parse_args()
    
    # Configuration dictionary
    config = {
        'env_name': args.env,
        'num_episodes': args.episodes,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'buffer_capacity': args.buffer_size,
        'hidden_dim': args.hidden_dim,
        'target_update_freq': args.target_update,
        'use_double_dqn': args.double_dqn,
        'save_dir': args.save_dir,
        'project_name': args.wandb_project,
        'print_freq': 10,
        'save_freq': 100,
        'solved_threshold': 195,  # For CartPole-v1
        'n_bins': args.n_bins,  # For continuous action space discretization
    }
    
    # Disable wandb if requested
    if args.no_wandb:
        # Create a mock wandb module that does nothing
        class MockWandb:
            @staticmethod
            def init(*args, **kwargs):
                return MockWandb()
            @staticmethod
            def log(data):
                pass
            @staticmethod
            def finish():
                pass
        
        wandb.init = MockWandb.init
        wandb.log = MockWandb.log
        wandb.finish = MockWandb.finish
    
    if args.evaluate and args.load_model:
        # Evaluation mode
        env = gym.make(config['env_name'])
        # Check if action space is continuous and needs discretization
        needs_discretization = isinstance(env.action_space, spaces.Box)
        if needs_discretization:
            n_bins = config.get('n_bins', 11)
            env = DiscretizedActionWrapper(env, n_bins=n_bins)
            print(f"⚠️  Continuous action space detected. Discretizing into {n_bins} actions.")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        
        agent = DQNAgent(state_dim, action_dim, config)
        agent.load(args.load_model)
        print(f"Loaded model from {args.load_model}")
        
        evaluate_agent(agent, config['env_name'], num_episodes=10, render=True)
    else:
        # Training mode
        print("Starting training...")
        agent = train_dqn(config)
        
        # Evaluate the trained agent
        print("\nEvaluating trained agent...")
        evaluate_agent(agent, config['env_name'], num_episodes=10, render=False, n_bins=config.get('n_bins', 11))

if __name__ == '__main__':
    main()

