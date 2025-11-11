"""
Complete Workflow: Train → Test → Record Videos for All 3 Environments
This Python script runs everything sequentially.
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} - Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ {description} - Interrupted by user")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     Complete DQN Workflow: Train → Test → Record             ║
    ║     Environments: CartPole, Acrobot, MountainCar, Pendulum   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # ========================================================================
    # PHASE 1: TRAINING
    # ========================================================================
    print("\n" + "="*70)
    print("  PHASE 1: TRAINING ALL ENVIRONMENTS")
    print("="*70 + "\n")
    
    # 1. CartPole-v1
    if os.path.exists("models/cartpole_v1/dqn_final.pth"):
        print("⚠️  CartPole model already exists. Skipping...")
    else:
        cmd = [
            sys.executable, 'train.py',
            '--env', 'CartPole-v1',
            '--double-dqn',
            '--episodes', '500',
            '--save-dir', 'models/cartpole_v1',
            '--wandb-project', 'dqn-cartpole'
        ]
        if not run_command(cmd, "[1/4] Training CartPole-v1"):
            return
    
    # 2. Acrobot-v1
    if os.path.exists("models/acrobot_v1/dqn_final.pth"):
        print("⚠️  Acrobot model already exists. Skipping...")
    else:
        cmd = [
            sys.executable, 'train.py',
            '--env', 'Acrobot-v1',
            '--double-dqn',
            '--episodes', '800',
            '--epsilon-decay', '0.997',
            '--buffer-size', '15000',
            '--save-dir', 'models/acrobot_v1',
            '--wandb-project', 'dqn-acrobot'
        ]
        if not run_command(cmd, "[2/4] Training Acrobot-v1"):
            return
    
    # 3. MountainCar-v0
    if os.path.exists("models/mountaincar_v0/dqn_final.pth"):
        print("⚠️  MountainCar model already exists. Skipping...")
    else:
        cmd = [
            sys.executable, 'train.py',
            '--env', 'MountainCar-v0',
            '--double-dqn',
            '--episodes', '1500',
            '--lr', '0.0005',
            '--epsilon-decay', '0.999',
            '--buffer-size', '20000',
            '--hidden-dim', '256',
            '--batch-size', '128',
            '--save-dir', 'models/mountaincar_v0',
            '--wandb-project', 'dqn-mountaincar'
        ]
        if not run_command(cmd, "[3/4] Training MountainCar-v0"):
            return
    
    # 4. Pendulum-v1
    if os.path.exists("models/pendulum_v1/dqn_final.pth"):
        print("⚠️  Pendulum model already exists. Skipping...")
    else:
        cmd = [
            sys.executable, 'train.py',
            '--env', 'Pendulum-v1',
            '--double-dqn',
            '--episodes', '1000',
            '--lr', '0.001',
            '--epsilon-decay', '0.998',
            '--buffer-size', '20000',
            '--hidden-dim', '256',
            '--batch-size', '128',
            '--n-bins', '11',
            '--save-dir', 'models/pendulum_v1',
            '--wandb-project', 'dqn-pendulum'
        ]
        if not run_command(cmd, "[4/4] Training Pendulum-v1"):
            return
    
    # ========================================================================
    # PHASE 2: TESTING (100 episodes each)
    # ========================================================================
    print("\n" + "="*70)
    print("  PHASE 2: TESTING ALL AGENTS (100 episodes each)")
    print("="*70 + "\n")
    
    # Test all 4 environments
    environments = [
        ('CartPole-v1', 'models/cartpole_v1/dqn_final.pth', None),
        ('Acrobot-v1', 'models/acrobot_v1/dqn_final.pth', None),
        ('MountainCar-v0', 'models/mountaincar_v0/dqn_final.pth', None),
        ('Pendulum-v1', 'models/pendulum_v1/dqn_final.pth', '11'),  # Needs n-bins
    ]
    
    for i, env_data in enumerate(environments, 1):
        if len(env_data) == 3:
            env_name, model_path, n_bins = env_data
        else:
            env_name, model_path = env_data
            n_bins = None
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}. Skipping {env_name}...")
            continue
        
        cmd = [
            sys.executable, 'test_agents.py',
            '--env', env_name,
            '--model-path', model_path,
            '--num-episodes', '100',
            '--double-dqn',
            '--wandb',
            '--wandb-project', 'dqn-testing'
        ]
        if n_bins:
            cmd.extend(['--n-bins', n_bins])
        
        if not run_command(cmd, f"[{i}/4] Testing {env_name}"):
            print(f"⚠️  Warning: Testing {env_name} failed, continuing...")
    
    # ========================================================================
    # PHASE 3: VIDEO RECORDING (5 episodes each)
    # ========================================================================
    print("\n" + "="*70)
    print("  PHASE 3: RECORDING VIDEOS (5 episodes each)")
    print("="*70 + "\n")
    
    # Record videos for all 4 environments
    video_configs = [
        ('CartPole-v1', 'models/cartpole_v1/dqn_final.pth', 'videos/cartpole_v1', None),
        ('Acrobot-v1', 'models/acrobot_v1/dqn_final.pth', 'videos/acrobot_v1', None),
        ('MountainCar-v0', 'models/mountaincar_v0/dqn_final.pth', 'videos/mountaincar_v0', None),
        ('Pendulum-v1', 'models/pendulum_v1/dqn_final.pth', 'videos/pendulum_v1', '11'),  # Needs n-bins
    ]
    
    for i, video_data in enumerate(video_configs, 1):
        if len(video_data) == 4:
            env_name, model_path, video_folder, n_bins = video_data
        else:
            env_name, model_path, video_folder = video_data
            n_bins = None
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}. Skipping {env_name}...")
            continue
        
        os.makedirs(video_folder, exist_ok=True)
        
        cmd = [
            sys.executable, 'record_video.py',
            '--model-path', model_path,
            '--env', env_name,
            '--num-episodes', '5',
            '--video-folder', video_folder,
            '--double-dqn',
            '--wandb-log',
            '--wandb-project', 'dqn-videos'
        ]
        if n_bins:
            cmd.extend(['--n-bins', n_bins])
        
        if not run_command(cmd, f"[{i}/4] Recording {env_name} videos"):
            print(f"⚠️  Warning: Video recording for {env_name} failed, continuing...")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("  🎉 COMPLETE WORKFLOW FINISHED!")
    print("="*70)
    print("\n📊 Results Summary:\n")
    print("  ✅ Training Complete:")
    print("    - models/cartpole_v1/dqn_final.pth")
    print("    - models/acrobot_v1/dqn_final.pth")
    print("    - models/mountaincar_v0/dqn_final.pth")
    print("    - models/pendulum_v1/dqn_final.pth")
    print("\n  ✅ Testing Complete (100 episodes each):")
    print("    - Results logged to wandb (dqn-testing project)")
    print("\n  ✅ Videos Recorded (5 episodes each):")
    print("    - videos/cartpole_v1/")
    print("    - videos/acrobot_v1/")
    print("    - videos/mountaincar_v0/")
    print("    - videos/pendulum_v1/")
    print("    - Videos logged to wandb (dqn-videos project)")
    print("\n📈 Check wandb.ai for all results!")
    print("="*70 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user. Partial results may be available.")
        sys.exit(1)

