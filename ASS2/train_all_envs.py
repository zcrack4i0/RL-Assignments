"""
Train DQN/DDQN on multiple Gymnasium environments
This script trains on CartPole, Acrobot, and MountainCar
"""

import subprocess
import sys
import os

# Environment configurations
ENVIRONMENTS = {
    'CartPole-v1': {
        'episodes': 500,
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'buffer_size': 10000,
        'hidden_dim': 128,
        'target_update': 10,
        'solved_threshold': 195,
    },
    'Acrobot-v1': {
        'episodes': 800,
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon_decay': 0.997,
        'batch_size': 64,
        'buffer_size': 15000,
        'hidden_dim': 128,
        'target_update': 10,
        'solved_threshold': -100,  # Lower is better (negative rewards)
    },
    'MountainCar-v0': {
        'episodes': 1500,
        'lr': 0.0005,
        'gamma': 0.99,
        'epsilon_decay': 0.999,
        'batch_size': 128,
        'buffer_size': 20000,
        'hidden_dim': 256,
        'target_update': 10,
        'solved_threshold': -110,  # Lower is better (negative rewards)
    },
}

def train_environment(env_name, config, use_double_dqn=True, use_wandb=False):
    """Train DQN on a specific environment"""
    
    print(f"\n{'='*70}")
    print(f"🎮 Training on {env_name}")
    print(f"{'='*70}")
    print(f"Episodes: {config['episodes']}")
    print(f"Algorithm: {'Double DQN' if use_double_dqn else 'Standard DQN'}")
    print(f"Solved threshold: {config['solved_threshold']}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [
        'python', 'train.py',
        '--env', env_name,
        '--episodes', str(config['episodes']),
        '--lr', str(config['lr']),
        '--gamma', str(config['gamma']),
        '--epsilon-decay', str(config['epsilon_decay']),
        '--batch-size', str(config['batch_size']),
        '--buffer-size', str(config['buffer_size']),
        '--hidden-dim', str(config['hidden_dim']),
        '--target-update', str(config['target_update']),
        '--save-dir', f'models/{env_name.lower().replace("-", "_")}',
    ]
    
    if use_double_dqn:
        cmd.append('--double-dqn')
    
    if not use_wandb:
        cmd.append('--no-wandb')
    else:
        cmd.extend(['--wandb-project', f'dqn-{env_name.lower()}'])
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Successfully trained on {env_name}!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error training on {env_name}: {e}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user on {env_name}\n")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          DQN/DDQN Multi-Environment Training Suite          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Parse command line arguments
    use_double_dqn = '--double-dqn' in sys.argv or '-d' in sys.argv
    use_wandb = '--wandb' in sys.argv or '-w' in sys.argv
    
    algorithm = "Double DQN" if use_double_dqn else "Standard DQN"
    print(f"Algorithm: {algorithm}")
    print(f"Wandb Logging: {'Enabled' if use_wandb else 'Disabled'}")
    print(f"\nEnvironments to train: {len(ENVIRONMENTS)}")
    
    # Train on each environment
    results = {}
    for env_name, config in ENVIRONMENTS.items():
        success = train_environment(env_name, config, use_double_dqn, use_wandb)
        results[env_name] = success
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    for env_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{env_name:20s} {status}")
    print("="*70 + "\n")
    
    # Print model locations
    print("💾 Trained models saved in:")
    for env_name in ENVIRONMENTS.keys():
        model_dir = f"models/{env_name.lower().replace('-', '_')}"
        if os.path.exists(model_dir):
            print(f"  - {model_dir}/")
    
    total_success = sum(results.values())
    print(f"\n🎉 Successfully trained on {total_success}/{len(ENVIRONMENTS)} environments!\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN on multiple environments')
    parser.add_argument('--double-dqn', '-d', action='store_true', 
                       help='Use Double DQN (recommended)')
    parser.add_argument('--wandb', '-w', action='store_true', 
                       help='Enable Weights & Biases logging')
    parser.add_argument('--env', type=str, default=None,
                       help='Train only on specific environment (CartPole-v1, Acrobot-v1, MountainCar-v0)')
    
    args = parser.parse_args()
    
    # If specific environment requested
    if args.env:
        if args.env in ENVIRONMENTS:
            config = ENVIRONMENTS[args.env]
            train_environment(args.env, config, args.double_dqn, args.wandb)
        else:
            print(f"❌ Unknown environment: {args.env}")
            print(f"Available: {', '.join(ENVIRONMENTS.keys())}")
    else:
        # Train on all environments
        main()

