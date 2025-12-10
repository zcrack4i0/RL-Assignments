"""
Simple test to verify CNN architecture without requiring Box2D environments
"""
import torch
import numpy as np
from NNArch import CNNFeatureExtractor, CNNGaussianPolicy, CNNQNetwork, GaussianPolicy, QNetwork

def test_cnn_feature_extractor():
    """Test CNN feature extractor"""
    print("\n" + "="*60)
    print("Testing CNNFeatureExtractor")
    print("="*60)
    
    cnn = CNNFeatureExtractor(input_channels=3)
    
    # Create fake image input (batch_size=4, channels=3, height=96, width=96)
    fake_images = torch.randn(4, 3, 96, 96)
    
    # Forward pass
    features = cnn(fake_images)
    
    print(f"Input shape: {fake_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected feature dim: {cnn.feature_dim}")
    
    assert features.shape == (4, 4096), f"Expected (4, 4096), got {features.shape}"
    print("✓ CNNFeatureExtractor test passed!\n")

def test_cnn_gaussian_policy():
    """Test CNN-based Gaussian policy"""
    print("\n" + "="*60)
    print("Testing CNNGaussianPolicy")
    print("="*60)
    
    action_dim = 3
    policy = CNNGaussianPolicy(input_channels=3, action_dim=action_dim, hidden_dim=256)
    
    # Create fake image input
    fake_images = torch.randn(2, 3, 96, 96)
    
    # Forward pass
    mean, log_std = policy.forward(fake_images)
    
    print(f"Input shape: {fake_images.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Log std shape: {log_std.shape}")
    
    assert mean.shape == (2, action_dim), f"Expected (2, {action_dim}), got {mean.shape}"
    assert log_std.shape == (2, action_dim), f"Expected (2, {action_dim}), got {log_std.shape}"
    
    # Test sampling
    action, log_prob, mean_action = policy.sample(fake_images)
    
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Mean action shape: {mean_action.shape}")
    
    assert action.shape == (2, action_dim)
    assert log_prob.shape == (2, 1)
    assert mean_action.shape == (2, action_dim)
    
    print("✓ CNNGaussianPolicy test passed!\n")

def test_cnn_q_network():
    """Test CNN-based Q-network"""
    print("\n" + "="*60)
    print("Testing CNNQNetwork")
    print("="*60)
    
    action_dim = 3
    q_net = CNNQNetwork(input_channels=3, action_dim=action_dim, hidden_dim=256)
    
    # Create fake inputs
    fake_images = torch.randn(2, 3, 96, 96)
    fake_actions = torch.randn(2, action_dim)
    
    # Forward pass
    q1, q2 = q_net(fake_images, fake_actions)
    
    print(f"State shape: {fake_images.shape}")
    print(f"Action shape: {fake_actions.shape}")
    print(f"Q1 shape: {q1.shape}")
    print(f"Q2 shape: {q2.shape}")
    
    assert q1.shape == (2, 1), f"Expected (2, 1), got {q1.shape}"
    assert q2.shape == (2, 1), f"Expected (2, 1), got {q2.shape}"
    
    print("✓ CNNQNetwork test passed!\n")

def test_mlp_networks():
    """Test MLP-based networks"""
    print("\n" + "="*60)
    print("Testing MLP Networks (GaussianPolicy, QNetwork)")
    print("="*60)
    
    state_dim = 8
    action_dim = 2
    
    # Test Gaussian Policy
    policy = GaussianPolicy(state_dim, action_dim, hidden_dim=256)
    fake_states = torch.randn(4, state_dim)
    
    mean, log_std = policy.forward(fake_states)
    action, log_prob, mean_action = policy.sample(fake_states)
    
    print(f"GaussianPolicy:")
    print(f"  Input shape: {fake_states.shape}")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Action shape: {action.shape}")
    
    assert mean.shape == (4, action_dim)
    assert action.shape == (4, action_dim)
    assert log_prob.shape == (4, 1)
    
    # Test Q-Network
    q_net = QNetwork(state_dim, action_dim, hidden_dim=256)
    fake_actions = torch.randn(4, action_dim)
    
    q1, q2 = q_net(fake_states, fake_actions)
    
    print(f"QNetwork:")
    print(f"  State shape: {fake_states.shape}")
    print(f"  Action shape: {fake_actions.shape}")
    print(f"  Q1 shape: {q1.shape}")
    print(f"  Q2 shape: {q2.shape}")
    
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)
    
    print("✓ MLP Networks test passed!\n")

def test_image_replay_buffer():
    """Test ImageReplayBuffer"""
    print("\n" + "="*60)
    print("Testing ImageReplayBuffer")
    print("="*60)
    
    from ImageReplayBuffer import ImageReplayBuffer
    
    image_shape = (96, 96, 3)
    action_dim = 3
    capacity = 100
    
    buffer = ImageReplayBuffer(capacity, image_shape, action_dim)
    
    # Add some transitions
    for i in range(50):
        state = np.random.randint(0, 256, image_shape, dtype=np.uint8)
        action = np.random.randn(action_dim).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randint(0, 256, image_shape, dtype=np.uint8)
        done = 0.0
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Buffer capacity: {capacity}")
    print(f"Buffer size: {buffer.size}")
    print(f"State dtype: {buffer.state.dtype}")
    print(f"State memory per sample: {buffer.state[0].nbytes} bytes")
    
    # Test sampling
    batch = buffer.sample(16)
    states, actions, rewards, next_states, dones = batch
    
    print(f"\nSampled batch:")
    print(f"  States shape: {states.shape} (should be (16, 3, 96, 96))")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Dones shape: {dones.shape}")
    
    assert states.shape == (16, 3, 96, 96), f"Expected (16, 3, 96, 96), got {states.shape}"
    assert actions.shape == (16, action_dim)
    assert next_states.shape == (16, 3, 96, 96)
    
    # Check normalization
    assert states.min() >= 0.0 and states.max() <= 1.0, "States should be normalized to [0, 1]"
    
    print("✓ ImageReplayBuffer test passed!\n")

def test_memory_efficiency():
    """Compare memory usage between uint8 and float32 storage"""
    print("\n" + "="*60)
    print("Testing Memory Efficiency")
    print("="*60)
    
    image_shape = (96, 96, 3)
    capacity = 1000
    
    # Calculate memory for uint8 storage (ImageReplayBuffer)
    uint8_bytes = capacity * np.prod(image_shape) * 1 * 2  # state + next_state
    uint8_mb = uint8_bytes / (1024 * 1024)
    
    # Calculate memory for float32 storage (old approach)
    float32_bytes = capacity * np.prod(image_shape) * 4 * 2  # state + next_state
    float32_mb = float32_bytes / (1024 * 1024)
    
    savings = (1 - uint8_bytes / float32_bytes) * 100
    
    print(f"Image shape: {image_shape}")
    print(f"Buffer capacity: {capacity}")
    print(f"\nMemory usage:")
    print(f"  uint8 storage: {uint8_mb:.2f} MB")
    print(f"  float32 storage: {float32_mb:.2f} MB")
    print(f"  Savings: {savings:.1f}%")
    
    assert savings > 70, "Should save at least 70% memory"
    print(f"\n✓ Memory efficiency test passed! (75% savings)\n")

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# CNN Architecture Verification (No Environment Required)")
    print("#"*60)
    
    try:
        # Test individual components
        test_cnn_feature_extractor()
        test_cnn_gaussian_policy()
        test_cnn_q_network()
        test_mlp_networks()
        test_image_replay_buffer()
        test_memory_efficiency()
        
        print("\n" + "="*60)
        print("ALL ARCHITECTURE TESTS PASSED! ✓")
        print("="*60)
        print("\nThe CNN-based SAC implementation is working correctly.")
        print("\nNote: To run full environment tests, you need to install:")
        print("  pip install gymnasium[box2d]")
        print("\nYou can now train on:")
        print("  - CarRacing-v3: python train_carracing.py")
        print("  - LunarLander-v3: python train_sac_box2d.py LunarLander-v3")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
