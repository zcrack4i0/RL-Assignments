# Deep Q-Network (DQN) Implementation

This project implements a Deep Q-Network (DQN) agent to solve reinforcement learning environments using PyTorch and Gymnasium (formerly OpenAI Gym). The implementation includes experience replay, target networks, and integration with Weights & Biases for experiment tracking.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Video Recording](#video-recording)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [References](#references)

## 🎯 Overview

Deep Q-Network (DQN) is a reinforcement learning algorithm that combines Q-learning with deep neural networks. This implementation demonstrates key concepts including:

- **Experience Replay**: Stores past experiences and samples them randomly to break correlations
- **Target Network**: Uses a separate target network that updates slowly for stability
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation during training
- **Experiment Tracking**: Integrates with Weights & Biases (wandb) for monitoring training progress

The default environment is CartPole-v1, where the agent learns to balance a pole on a cart by moving left or right.

## ✨ Features

- 🧠 **DQN & Double DQN**: Choose between standard DQN or Double DQN (DDQN)
- 💾 **Experience Replay Buffer**: Efficient storage and sampling of transitions
- 📊 **Weights & Biases Integration**: Automatic logging of metrics and hyperparameters
- 🎥 **Video Recording**: Record and save agent gameplay videos
- 💾 **Model Checkpointing**: Save and load trained models
- ⚙️ **Configurable Hyperparameters**: Easy customization via command-line arguments
- 🎮 **Gymnasium Support**: Compatible with any Gymnasium discrete action space environment

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Set up Weights & Biases** (optional but recommended):

```bash
wandb login
```

Follow the prompts to enter your API key from https://wandb.ai

## 🚀 Usage

### Training

Train a DQN agent with default parameters:

```bash
python train.py
```

Train with custom hyperparameters:

```bash
python train.py --env CartPole-v1 --episodes 500 --batch-size 64 --lr 0.001 --gamma 0.99
```

Available training arguments:

- `--env`: Gymnasium environment name (default: 'CartPole-v1')
- `--episodes`: Number of training episodes (default: 500)
- `--batch-size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon-start`: Starting epsilon for exploration (default: 1.0)
- `--epsilon-min`: Minimum epsilon (default: 0.01)
- `--epsilon-decay`: Epsilon decay rate (default: 0.995)
- `--buffer-size`: Replay buffer size (default: 10000)
- `--hidden-dim`: Hidden layer dimension (default: 128)
- `--target-update`: Target network update frequency (default: 10)
- `--save-dir`: Directory to save models (default: 'models')
- `--wandb-project`: Wandb project name (default: 'dqn-rl-assignment')
- `--no-wandb`: Disable wandb logging
- `--double-dqn`: Use Double DQN instead of standard DQN (reduces overestimation bias)

### Evaluation

Evaluate a trained agent:

```bash
python train.py --evaluate --load-model models/dqn_final.pth
```

This will run 10 evaluation episodes with rendering enabled, showing the agent's performance.

### Video Recording

Record videos of your trained agent:

```bash
python record_video.py --model-path models/dqn_final.pth --num-episodes 5
```

Video recording arguments:

- `--model-path`: Path to trained model (required)
- `--env`: Gymnasium environment name (default: 'CartPole-v1')
- `--num-episodes`: Number of episodes to record (default: 5)
- `--video-folder`: Folder to save videos (default: 'videos')
- `--wandb-log`: Log videos to wandb
- `--wandb-project`: Wandb project name (default: 'dqn-rl-assignment')
- `--double-dqn`: Use if model was trained with Double DQN

Videos will be saved in the `videos/` folder as MP4 files.

## 📁 Project Structure

```
.
├── dqn_network.py         # DQN neural network architecture
├── replay_buffer.py       # Experience replay buffer implementation
├── dqn_agent.py          # DQN agent with training logic
├── train.py              # Training script with wandb integration
├── record_video.py       # Video recording script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── models/              # Saved model checkpoints (created during training)
└── videos/              # Recorded videos (created during recording)
```

## ⚙️ Configuration

### Default Hyperparameters

The following default hyperparameters work well for CartPole-v1:

| Parameter          | Default Value | Description                          |
| ------------------ | ------------- | ------------------------------------ |
| Learning Rate      | 0.001         | Step size for gradient descent       |
| Gamma (γ)          | 0.99          | Discount factor for future rewards   |
| Epsilon Start      | 1.0           | Initial exploration rate             |
| Epsilon Min        | 0.01          | Minimum exploration rate             |
| Epsilon Decay      | 0.995         | Rate of exploration decay            |
| Batch Size         | 64            | Number of samples per training step  |
| Buffer Size        | 10000         | Maximum replay buffer capacity       |
| Hidden Dimension   | 128           | Neurons in hidden layers             |
| Target Update Freq | 10            | Steps between target network updates |

### Environment Configuration

The code is compatible with any Gymnasium environment with discrete action spaces. To try different environments:

```bash
# MountainCar
python train.py --env MountainCar-v0 --episodes 1000

# Acrobot
python train.py --env Acrobot-v1 --episodes 500

# LunarLander (requires gymnasium[box2d])
python train.py --env LunarLander-v2 --episodes 1000
```

Note: You may need to adjust hyperparameters for different environments.

## 🔄 Standard DQN vs Double DQN

This implementation supports both algorithms:

### Standard DQN

```bash
python train.py  # Default
```

- Uses the **target network** for both action selection and evaluation
- Simple and effective for most problems
- Can suffer from **overestimation bias** (Q-values are too optimistic)

### Double DQN (DDQN)

```bash
python train.py --double-dqn  # Recommended for better performance
```

- Uses **policy network** to select actions, **target network** to evaluate them
- **Reduces overestimation bias** leading to more stable learning
- Often achieves better final performance and faster convergence

**Key Difference:**

```python
# Standard DQN
Q_target = reward + γ * max(Q_target_net(next_state))

# Double DQN
action = argmax(Q_policy_net(next_state))
Q_target = reward + γ * Q_target_net(next_state, action)
```

**When to Use:**

- **Standard DQN**: Simple environments, faster initial testing
- **Double DQN**: Complex environments, want best performance, reduce overestimation

## 📊 Results

### Training Progress

During training, the following metrics are logged to wandb:

- **Episode Reward**: Total reward obtained in each episode
- **Episode Length**: Number of steps in each episode
- **Epsilon**: Current exploration rate
- **Average Loss**: Mean loss over the episode
- **Average Reward (100)**: Rolling average over last 100 episodes

### Expected Performance (CartPole-v1)

- **Solved Threshold**: Average reward ≥ 195 over 100 consecutive episodes
- **Training Time**: ~200-300 episodes to solve (varies with random seed)
- **Final Performance**: ~500 timesteps per episode (maximum)

### Monitoring with Weights & Biases

View your training progress at: https://wandb.ai/your-username/dqn-rl-assignment

The wandb dashboard provides:

- Real-time training metrics
- Hyperparameter tracking
- Model comparison across runs
- System metrics (CPU, GPU, memory)

## 📚 References

### Papers

- **DQN Paper**: Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.
- **DQN Improvements**: Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double Q-learning." AAAI.

### Documentation

- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

### Key Concepts

1. **Q-Learning**: Learn an action-value function Q(s, a) that estimates expected return
2. **Experience Replay**: Store transitions in buffer and sample randomly to decorrelate data
3. **Target Network**: Use separate network for computing targets to improve stability
4. **Epsilon-Greedy**: Explore with probability ε, otherwise take greedy action

## 🎓 Assignment Notes

This implementation covers the following concepts typically required in RL Assignment 2:

✅ Deep Q-Network architecture with PyTorch  
✅ Experience replay buffer  
✅ Target network for stable training  
✅ Epsilon-greedy exploration strategy  
✅ Gymnasium environment integration  
✅ Weights & Biases experiment tracking  
✅ Video recording of agent performance  
✅ Model saving and loading  
✅ Evaluation metrics and logging

## 🤝 Contributing

Feel free to modify and extend this implementation for your own experiments!

## 📝 License

This project is provided for educational purposes.

---

**Good luck with your reinforcement learning journey! 🚀**
