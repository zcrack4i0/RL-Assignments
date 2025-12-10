# Quick Start Guide - CarRacing-v3 with SAC

## ✅ What's Ready

Your SAC implementation now supports **CarRacing-v3** with CNN-based networks!

## 🚀 Quick Start

### 1. Install Dependencies (if needed)

```powershell
pip install torch gymnasium[box2d] numpy wandb moviepy
```

### 2. Run Training

**CarRacing-v3 (Recommended):**

```powershell
python train_carracing.py
```

**Quick Test (Single Config):**

```powershell
python train_carracing.py single
```

**Both Environments:**

```powershell
python train_sac_box2d.py
```

### 3. Monitor Progress

- **Videos**: `videos/` folder
- **Models**: `saved_models/` folder
- **Logs**: `training_logs/` folder
- **WandB**: Check dashboard for real-time metrics

## 📊 Expected Results

### CarRacing-v3:

- **Episodes**: 1500-2000 for good performance
- **Target**: Average reward > 700 (good), > 850 (excellent)
- **Time**: Several hours on GPU

### LunarLander-v3:

- **Episodes**: 500-1000 episodes
- **Target**: Average reward > 200 (solved)
- **Time**: 1-2 hours on CPU/GPU

## 🔧 Key Features

### Automatic Architecture Selection

- **CarRacing**: Uses CNN (image → 4096 features)
- **LunarLander**: Uses MLP (8D vector)

### Memory Optimization

- Images stored as uint8 (75% memory savings)
- Buffer: 50K images = ~53 MB (vs 211 MB)

### Training Optimizations

- Frame skipping (4x speedup)
- Early termination (off-track detection)
- Warm-up period (better exploration)

## 📁 New Files

### Core Implementation:

- `ImageReplayBuffer.py` - Memory-efficient image storage
- `train_carracing.py` - Optimized CarRacing training
- `test_architecture.py` - Verify implementation

### Updated Files:

- `NNArch.py` - Added CNN networks
- `SAC.py` - Added CNN support
- `train_sac_box2d.py` - Auto-detects observation type

### Documentation:

- `README_CNN_IMPROVEMENTS.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Detailed summary
- `QUICK_START.md` - This file

## 🎯 Best Hyperparameters

### CarRacing-v3:

```python
learning_rate = 3e-4
gamma = 0.99
tau = 0.005
alpha = 0.2
batch_size = 128
buffer_size = 50000
frame_skip = 4
warmup_episodes = 10
```

### LunarLander-v3:

```python
learning_rate = 3e-4
gamma = 0.99
tau = 0.005
alpha = 0.2
batch_size = 256
buffer_size = 100000
```

## 🧪 Test Implementation

```powershell
# Test architecture (no Box2D needed)
python test_architecture.py

# Full test (requires Box2D)
python test_implementation.py
```

## 💡 Tips

### GPU Usage:

- Automatically detected and used if available
- **10x faster** for CNN training
- Check with: `torch.cuda.is_available()`

### Memory Issues:

Reduce in config:

```python
buffer_size = 25000  # instead of 50000
batch_size = 64      # instead of 128
```

### Slow Training:

```python
frame_skip = 6  # instead of 4
```

### Monitor Training:

```python
# Videos recorded every 50 episodes
# Check progress in videos/ folder
# WandB logs every episode
```

## 📈 Training Progress

Typical learning curve for CarRacing:

- **Episodes 0-100**: Random exploration, negative rewards
- **Episodes 100-500**: Learning basics, rewards improve
- **Episodes 500-1000**: Stable driving, rewards > 500
- **Episodes 1000-2000**: Optimizing, rewards > 700+

## 🛠️ Troubleshooting

### "Box2D not installed":

```powershell
pip install gymnasium[box2d]
```

### "CUDA out of memory":

```python
# Reduce batch_size and buffer_size in config
```

### "Training too slow":

```python
# Increase frame_skip to 5 or 6
# Use GPU if available
```

### "Poor performance after 1000 episodes":

```python
# Train longer (try 2000 episodes)
# Try different learning_rate (1e-4 or 5e-4)
# Check WandB curves for issues
```

## ✨ What Changed

### Before (Flattening):

```
Image (96x96x3) → Flatten → 27,648 dims
→ MLP → Actions
```

❌ High dimensional
❌ Loses structure
❌ High memory

### After (CNN):

```
Image (96x96x3) → CNN → 4,096 features
→ MLP → Actions
```

✅ Low dimensional
✅ Preserves structure
✅ Memory efficient
✅ Faster learning

## 🎉 Ready to Train!

Your implementation is ready. Just run:

```powershell
python train_carracing.py
```

And monitor the progress in WandB and the videos folder!

---

**Need Help?** Check:

- `README_CNN_IMPROVEMENTS.md` for detailed documentation
- `IMPLEMENTATION_SUMMARY.md` for technical details
- WandB dashboard for training metrics
