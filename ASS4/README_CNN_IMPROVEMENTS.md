# SAC Implementation for CarRacing-v3 and LunarLander-v3

## Overview

This implementation now supports both **CarRacing-v3** (image observations) and **LunarLander-v3** (vector observations) using Soft Actor-Critic (SAC) with automatic architecture selection.

## Key Improvements

### 1. **CNN-Based Networks for Image Observations**

- Added `CNNFeatureExtractor` for processing 96x96x3 images from CarRacing-v3
- Implemented `CNNGaussianPolicy` and `CNNQNetwork` for image-based policy and Q-learning
- Reduces dimensionality from 27,648 (flattened) to 4,096 CNN features

### 2. **Efficient Memory Management**

- `ImageReplayBuffer`: Stores images as uint8 (0-255) to save ~75% memory
- Converts to float32 and normalizes to [0, 1] only during sampling
- Handles image transposition from (H, W, C) to (C, H, W) automatically

### 3. **Automatic Architecture Selection**

- SAC agent automatically detects observation type (vector vs image)
- Uses MLP networks for LunarLander-v3 (8-dimensional vectors)
- Uses CNN networks for CarRacing-v3 (96x96x3 images)

### 4. **Optimized Hyperparameters for CarRacing**

The following configurations have been optimized for CarRacing-v3:

```python
# Configuration 1: Balanced (Recommended)
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
}

# Configuration 2: Large Buffer
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
}

# Configuration 3: Fast Learning
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
```

## File Structure

### Core Files

- **`SAC.py`**: Updated SAC agent with CNN/MLP support
- **`NNArch.py`**: Neural network architectures (CNN and MLP)
- **`ImageReplayBuffer.py`**: Memory-efficient replay buffer for images
- **`ReplayBuffer.py`**: Standard replay buffer for vectors

### Training Scripts

- **`train_carracing.py`**: Specialized training for CarRacing-v3 with CNN
- **`train_sac_box2d.py`**: Universal training for both environments (auto-detects)
- **`train_sac_full.py`**: Legacy full training script
- **`train_sac_small.py`**: Quick test script

## Usage

### Training CarRacing-v3

#### Option 1: Specialized Script (Recommended)

```powershell
python train_carracing.py
```

#### Option 2: Quick Single Config Test

```powershell
python train_carracing.py single
```

#### Option 3: Universal Script

```powershell
python train_sac_box2d.py CarRacing-v3
```

### Training LunarLander-v3

```powershell
python train_sac_box2d.py LunarLander-v3
```

### Training Both Environments

```powershell
python train_sac_box2d.py
```

## Network Architectures

### CNN-Based Networks (CarRacing-v3)

**CNNFeatureExtractor:**

```
Input: (batch, 3, 96, 96)
Conv2d(3→32, kernel=8, stride=4) + ReLU → (batch, 32, 23, 23)
Conv2d(32→64, kernel=4, stride=2) + ReLU → (batch, 64, 10, 10)
Conv2d(64→64, kernel=3, stride=1) + ReLU → (batch, 64, 8, 8)
Flatten → (batch, 4096)
```

**CNNGaussianPolicy:**

```
CNNFeatureExtractor → 4096 features
Linear(4096, 256) + ReLU
Linear(256, 256) + ReLU
Mean: Linear(256, action_dim)
LogStd: Linear(256, action_dim)
```

**CNNQNetwork (Double Q-Learning):**

```
Two parallel Q-networks:
CNNFeatureExtractor → 4096 features
Concat[features, action] → 4096 + action_dim
Linear(4096+action_dim, 256) + ReLU
Linear(256, 256) + ReLU
Q-value: Linear(256, 1)
```

### MLP-Based Networks (LunarLander-v3)

**GaussianPolicy:**

```
Input: (batch, 8)
Linear(8, 256) + ReLU
Linear(256, 256) + ReLU
Mean: Linear(256, action_dim)
LogStd: Linear(256, action_dim)
```

**QNetwork (Double Q-Learning):**

```
Two parallel Q-networks:
Concat[state, action] → state_dim + action_dim
Linear(state_dim+action_dim, 256) + ReLU
Linear(256, 256) + ReLU
Q-value: Linear(256, 1)
```

## CarRacing-v3 Optimizations

### 1. Frame Skipping

- Default: 4 frames per action
- Reduces computational cost by 75%
- Actions are repeated for `frame_skip` frames

### 2. Early Termination

- Episode ends if car goes off-track for >10 consecutive steps
- Prevents wasting time on failed episodes

### 3. Warm-up Period

- First 10-20 episodes use random actions
- Fills replay buffer with diverse experiences before learning

### 4. Memory Efficiency

- Images stored as uint8: 1 byte per pixel
- Flattened approach would need: 96×96×3 = 27,648 floats = 110 KB per image
- CNN approach: 96×96×3 = 27,648 bytes = 27 KB per image
- **75% memory savings!**

## Expected Performance

### LunarLander-v3 (Continuous)

- **Solved threshold:** Average reward > 200 over 100 episodes
- **Expected training time:** 500-1000 episodes
- **Best config:** lr=3e-4, batch_size=256

### CarRacing-v3 (Continuous)

- **Good performance:** Average reward > 700
- **Excellent performance:** Average reward > 850
- **Expected training time:** 1500-2000 episodes
- **Best config:** lr=3e-4, batch_size=128, frame_skip=4

## Monitoring Training

### WandB Integration

All training scripts log to Weights & Biases:

- Episode rewards
- Episode lengths
- Training losses
- Moving averages (10 and 100 episodes)
- Alpha (entropy temperature)
- Buffer size

### Video Recording

- LunarLander: Every 20th episode
- CarRacing: Every 50th episode
- Test episodes: All episodes recorded

### Saved Models

- `saved_models/sac_CarRacing-v3_cnn_best.pth`: Best during training
- `saved_models/sac_CarRacing-v3_cnn_final.pth`: Final model
- `saved_models/sac_LunarLander-v3_best.pth`: Best during training
- `saved_models/sac_LunarLander-v3_final.pth`: Final model

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `buffer_size` (e.g., 25000)
2. Reduce `batch_size` (e.g., 64)
3. Increase `frame_skip` (e.g., 6)

### Slow Training

1. Increase `frame_skip` (trades off control precision for speed)
2. Enable GPU acceleration (automatically detected)
3. Reduce logging frequency

### Poor Performance on CarRacing

1. Increase training episodes (try 2000+)
2. Adjust `warmup_episodes` (10-20 recommended)
3. Try different learning rates (1e-4 to 5e-4)
4. Ensure GPU is being used

## Requirements

```
torch>=2.0.0
gymnasium[box2d]>=0.28.0
numpy>=1.24.0
wandb>=0.15.0
moviepy>=1.0.0
```

Install with:

```powershell
pip install -r requirements.txt
```

## Citation

Based on Soft Actor-Critic (SAC):

- Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (2018)
- Haarnoja et al. "Soft Actor-Critic Algorithms and Applications" (2019)

## Notes

- GPU is highly recommended for CarRacing-v3 (CNNs are computationally expensive)
- LunarLander-v3 can train effectively on CPU
- Training from scratch takes significant time - be patient!
- Monitor WandB for learning curves and debugging
