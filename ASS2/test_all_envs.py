"""
Test all trained agents on 100 episodes each and record videos.
This script runs comprehensive testing and video recording for all environments.
"""

import subprocess
import sys
import os

# Environment configurations
ENVIRONMENTS = {
    'CartPole-v1': {
        'model_path': 'models/cartpole_v1/dqn_final.pth',
        'video_folder': 'videos/cartpole_v1',
    },
    'Acrobot-v1': {
        'model_path': 'models/acrobot_v1/dqn_final.pth',
        'video_folder': 'videos/acrobot_v1',
    },
    'MountainCar-v0': {
        'model_path': 'models/mountaincar_v0/dqn_final.pth',
        'video_folder': 'videos/mountaincar_v0',
    },
}

def test_environment(env_name, config, num_episodes=100, use_wandb=False, use_double_dqn=True):
    """Test a trained agent on an environment"""
    
    print(f"\n{'='*70}")
    print(f"🧪 Testing {env_name}")
    print(f"{'='*70}")
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"❌ Model not found: {config['model_path']}")
        print(f"   Skipping {env_name}...")
        return False
    
    # Build test command
    cmd = [
        'python', 'test_agents.py',
        '--env', env_name,
        '--model-path', config['model_path'],
        '--num-episodes', str(num_episodes),
    ]
    
    if use_double_dqn:
        cmd.append('--double-dqn')
    
    if use_wandb:
        cmd.extend(['--wandb', '--wandb-project', 'dqn-testing'])
    
    # Run testing
    try:
        print(f"Running 100 test episodes...")
        result = subprocess.run(cmd, check=True)
        print(f"✅ Testing complete for {env_name}!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error testing {env_name}: {e}\n")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ Testing interrupted by user on {env_name}\n")
        return False

def record_videos(env_name, config, num_episodes=5, use_wandb=False, use_double_dqn=True):
    """Record videos of a trained agent"""
    
    print(f"\n{'='*70}")
    print(f"🎥 Recording videos for {env_name}")
    print(f"{'='*70}")
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"❌ Model not found: {config['model_path']}")
        print(f"   Skipping video recording for {env_name}...")
        return False
    
    # Create video folder
    os.makedirs(config['video_folder'], exist_ok=True)
    
    # Build record command
    cmd = [
        'python', 'record_video.py',
        '--model-path', config['model_path'],
        '--env', env_name,
        '--num-episodes', str(num_episodes),
        '--video-folder', config['video_folder'],
    ]
    
    if use_double_dqn:
        cmd.append('--double-dqn')
    
    if use_wandb:
        cmd.extend(['--wandb-log', '--wandb-project', 'dqn-videos'])
    
    # Run video recording
    try:
        print(f"Recording {num_episodes} episodes...")
        result = subprocess.run(cmd, check=True)
        print(f"✅ Video recording complete for {env_name}!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error recording {env_name}: {e}\n")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ Recording interrupted by user on {env_name}\n")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     DQN Agent Testing & Video Recording Suite                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Parse arguments
    use_wandb = '--wandb' in sys.argv or '-w' in sys.argv
    use_double_dqn = '--double-dqn' in sys.argv or '-d' in sys.argv
    test_only = '--test-only' in sys.argv
    video_only = '--video-only' in sys.argv
    
    print(f"Configuration:")
    print(f"  Wandb Logging: {'Enabled' if use_wandb else 'Disabled'}")
    print(f"  Double DQN: {'Yes' if use_double_dqn else 'No'}")
    print(f"  Mode: {'Test Only' if test_only else 'Video Only' if video_only else 'Test + Video'}")
    print()
    
    test_results = {}
    video_results = {}
    
    # Test all environments
    if not video_only:
        print("="*70)
        print("PHASE 1: Testing Agents (100 episodes each)")
        print("="*70)
        
        for env_name, config in ENVIRONMENTS.items():
            success = test_environment(env_name, config, num_episodes=100, 
                                     use_wandb=use_wandb, use_double_dqn=use_double_dqn)
            test_results[env_name] = success
    
    # Record videos for all environments
    if not test_only:
        print("="*70)
        print("PHASE 2: Recording Videos (5 episodes each)")
        print("="*70)
        
        for env_name, config in ENVIRONMENTS.items():
            success = record_videos(env_name, config, num_episodes=5,
                                  use_wandb=use_wandb, use_double_dqn=use_double_dqn)
            video_results[env_name] = success
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    if not video_only:
        print("\nTesting Results:")
        for env_name, success in test_results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {env_name:20s} {status}")
    
    if not test_only:
        print("\nVideo Recording Results:")
        for env_name, success in video_results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {env_name:20s} {status}")
    
    print("\n" + "="*70)
    
    # Print file locations
    print("\n💾 Results saved in:")
    if not video_only:
        print("  Test results logged to wandb (if enabled)")
    if not test_only:
        print("  Videos saved in:")
        for env_name, config in ENVIRONMENTS.items():
            print(f"    - {config['video_folder']}/")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and record videos for all environments')
    parser.add_argument('--wandb', '-w', action='store_true', 
                       help='Enable Weights & Biases logging')
    parser.add_argument('--double-dqn', '-d', action='store_true', 
                       help='Models were trained with Double DQN')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing, skip video recording')
    parser.add_argument('--video-only', action='store_true',
                       help='Only record videos, skip testing')
    
    args = parser.parse_args()
    main()

