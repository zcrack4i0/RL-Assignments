"""
Quick test to verify CNN-based SAC implementation works correctly
"""
import torch
import numpy as np
import gymnasium as gym
from SAC import SACAgent

def test_lunar_lander():
    """Test MLP-based SAC on LunarLander-v3"""
    print("\n" + "="*60)
    print("Testing SAC on LunarLander-v3 (Vector Observations)")
    print("="*60)
    
    env = gym.make("LunarLander-v3", continuous=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'batch_size': 64,
        'buffer_size': 10000
    }
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Initialize agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        config=config,
        use_cnn=False
    )
    
    print(f"Agent device: {agent.device}")
    print(f"Policy type: {type(agent.policy).__name__}")
    print(f"Critic type: {type(agent.critic).__name__}")
    
    # Test forward pass
    state, _ = env.reset()
    action = agent.select_action(state)
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    
    # Test one episode
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 100:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        
        if agent.memory.size > config['batch_size']:
            loss = agent.update(config['batch_size'])
        
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"Episode completed: {steps} steps, reward: {total_reward:.2f}")
    print(f"Buffer size: {agent.memory.size}")
    print("✓ LunarLander test passed!\n")
    
    env.close()

def test_car_racing():
    """Test CNN-based SAC on CarRacing-v3"""
    print("\n" + "="*60)
    print("Testing SAC on CarRacing-v3 (Image Observations)")
    print("="*60)
    
    env = gym.make("CarRacing-v3", continuous=True)
    image_shape = env.observation_space.shape  # (96, 96, 3)
    action_dim = env.action_space.shape[0]
    
    config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'batch_size': 32,  # Small batch for testing
        'buffer_size': 1000  # Small buffer for testing
    }
    
    print(f"Image shape: {image_shape}, Action dim: {action_dim}")
    
    # Initialize agent with CNN
    agent = SACAgent(
        state_dim=np.prod(image_shape),  # Not used but required
        action_dim=action_dim,
        action_space=env.action_space,
        config=config,
        use_cnn=True,
        input_channels=3,
        image_shape=image_shape
    )
    
    print(f"Agent device: {agent.device}")
    print(f"Policy type: {type(agent.policy).__name__}")
    print(f"Critic type: {type(agent.critic).__name__}")
    
    # Test forward pass
    state, _ = env.reset()
    action = agent.select_action(state)
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    
    # Test memory efficiency
    print(f"\nMemory test:")
    print(f"  State dtype: {state.dtype}")
    print(f"  State memory: {state.nbytes} bytes")
    print(f"  Buffer state dtype: {agent.memory.state.dtype}")
    
    # Test one episode (short)
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 50:  # Short episode for testing
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        
        if agent.memory.size > config['batch_size']:
            loss = agent.update(config['batch_size'])
        
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"\nEpisode completed: {steps} steps, reward: {total_reward:.2f}")
    print(f"Buffer size: {agent.memory.size}")
    
    # Check GPU memory if available
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    print("✓ CarRacing test passed!\n")
    
    env.close()

def test_architecture_switch():
    """Test that the same SAC class handles both architectures"""
    print("\n" + "="*60)
    print("Testing Architecture Auto-Selection")
    print("="*60)
    
    # Create both environments
    lunar_env = gym.make("LunarLander-v3", continuous=True)
    car_env = gym.make("CarRacing-v3", continuous=True)
    
    config = {'learning_rate': 3e-4, 'gamma': 0.99, 'tau': 0.005, 
              'alpha': 0.2, 'batch_size': 32, 'buffer_size': 1000}
    
    # Test LunarLander (should use MLP)
    lunar_agent = SACAgent(
        state_dim=lunar_env.observation_space.shape[0],
        action_dim=lunar_env.action_space.shape[0],
        action_space=lunar_env.action_space,
        config=config,
        use_cnn=False
    )
    
    assert not lunar_agent.use_cnn, "LunarLander should use MLP"
    assert "Gaussian" in type(lunar_agent.policy).__name__, "Should use GaussianPolicy"
    print("✓ LunarLander uses MLP networks")
    
    # Test CarRacing (should use CNN)
    car_agent = SACAgent(
        state_dim=np.prod(car_env.observation_space.shape),
        action_dim=car_env.action_space.shape[0],
        action_space=car_env.action_space,
        config=config,
        use_cnn=True,
        input_channels=3,
        image_shape=car_env.observation_space.shape
    )
    
    assert car_agent.use_cnn, "CarRacing should use CNN"
    assert "CNN" in type(car_agent.policy).__name__, "Should use CNNGaussianPolicy"
    print("✓ CarRacing uses CNN networks")
    
    lunar_env.close()
    car_env.close()
    
    print("✓ Architecture auto-selection test passed!\n")

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# SAC CNN Implementation Verification Tests")
    print("#"*60)
    
    try:
        test_architecture_switch()
        test_lunar_lander()
        test_car_racing()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nYou can now run:")
        print("  - python train_carracing.py       (CarRacing with CNN)")
        print("  - python train_sac_box2d.py       (Both environments)")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
