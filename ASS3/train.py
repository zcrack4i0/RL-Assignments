import wandb
import itertools
import json
import os
import torch
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from SAC import SACAgent
from DiscreteToContinuousWrapper import DiscreteToContinuousWrapper
from PPO import PPOAgent
from Actor_critic import A2CAgent
import numpy as np

# Ensure directories exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("best_configs", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# [cite: 5-10] Hyperparameter Search Space
hyperparams = {
    "learning_rate": [1e-3, 3e-4],       
    "gamma": [0.99, 0.95],               
    "batch_size": [64, 128],             
    "buffer_size": [10000, 50000],       
}

def make_env(env_name, model_type, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    # [cite: 11] SAC Wrapper Logic for Discrete Envs
    if model_type == "SAC" and env_name != "Pendulum-v1":
        env = DiscreteToContinuousWrapper(env)
    return env

def get_env_dims(env):
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        is_continuous = True
    else:
        action_dim = env.action_space.n
        is_continuous = False
    return state_dim, action_dim, is_continuous

# --- PHASE 1: TRAINING & OPTIMIZATION ---
def train_and_validate(model_type, env_name, config):
    run_name = f"TRAIN_{model_type}_{env_name}_lr{config['learning_rate']}_bs{config['batch_size']}"
    wandb.init(project="CMPS458-Assignment3", config=config, reinit=True, name=run_name, group="optimization",mode="offline")
    
    env = make_env(env_name, model_type)
    state_dim, action_dim, is_continuous = get_env_dims(env)
    
    # Instantiate Agent
    if model_type == "SAC":
        agent = SACAgent(state_dim, action_dim, env.action_space, config)
    elif model_type == "PPO":
        agent = PPOAgent(state_dim, action_dim, config, is_continuous)
    elif model_type == "A2C":
        agent = A2CAgent(state_dim, action_dim, config, is_continuous)

    # Train for 500 Episodes [cite: 12]
    print(f"   > Training {model_type} on {env_name}...")
    for episode in range(500):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if model_type == "SAC":
                action = agent.select_action(state)
            else:
                action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if model_type == "SAC":
                agent.store_transition(state, action, reward, next_state, done)
                if agent.memory.size > config['batch_size']:
                    loss = agent.update(config['batch_size'])
                    wandb.log({"train_loss": loss})
            elif model_type in ["PPO", "A2C"]:
                agent.store_reward(reward, done)
            
            state = next_state
            total_reward += reward

        if model_type in ["PPO", "A2C"]:
            loss = agent.update()
            wandb.log({"train_loss": loss})
        
        wandb.log({"train_reward": total_reward})

    # Short Validation to determine "Best" (5 Episodes)
    val_rewards = []
    for _ in range(5):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            # Use evaluate=True for SAC to be deterministic
            if model_type == "SAC":
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        val_rewards.append(ep_reward)
    
    avg_val_reward = np.mean(val_rewards)
    wandb.finish()
    
    return avg_val_reward, agent

# --- PHASE 2: FINAL TESTING ---
def test_best_model(model_type, env_name, config, model_path):
    run_name = f"TEST_{model_type}_{env_name}_BEST"
    wandb.init(project="CMPS458-Assignment3", config=config, reinit=True, name=run_name, group="final_testing",mode="offline")
    
    print(f"--- Starting Final Test: {model_type} on {env_name} ---")
    
    # 1. Setup Env & Agent
    # [cite: 18] RecordVideo Wrapper
    env = make_env(env_name, model_type, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=f"./videos/{model_type}_{env_name}", episode_trigger=lambda x: x % 19 == 0)
    
    state_dim, action_dim, is_continuous = get_env_dims(env)
    
    # 2. Re-instantiate Agent
    if model_type == "SAC":
        agent = SACAgent(state_dim, action_dim, env.action_space, config)
        checkpoint = torch.load(model_path)
        agent.policy.load_state_dict(checkpoint['policy'])
        # SAC critics aren't strictly needed for testing, but good practice to load if resumed
    elif model_type == "PPO":
        agent = PPOAgent(state_dim, action_dim, config, is_continuous)
        agent.policy.load_state_dict(torch.load(model_path))
    elif model_type == "A2C":
        agent = A2CAgent(state_dim, action_dim, config, is_continuous)
        agent.network.load_state_dict(torch.load(model_path))

    # 3. Run 100 Test Episodes [cite: 17]
    durations = []
    rewards = []
    
    for i in range(100):
        state, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        
        while not done:
            if model_type == "SAC":
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            ep_reward += reward
            
        durations.append(steps)
        rewards.append(ep_reward)
        wandb.log({"test_duration": steps, "test_reward": ep_reward})
    
    avg_duration = np.mean(durations)
    avg_reward = np.mean(rewards)
    
    print(f"Test Complete. Avg Duration: {avg_duration}, Avg Reward: {avg_reward}")
    wandb.log({"final_avg_duration": avg_duration, "final_avg_reward": avg_reward})
    env.close()
    wandb.finish()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    environments = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]
    models = ["PPO", "A2C","SAC"]
    
    # Generate configs
    keys, values = zip(*hyperparams.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_registry = {} # To store info about the winners

    # ==========================
    # PHASE 1: OPTIMIZATION LOOP
    # ==========================
    print("\n=== PHASE 1: HYPERPARAMETER OPTIMIZATION ===")
    for env_name in environments:
        for model in models:
            best_score = -float('inf')
            
            print(f"\n> Optimizing {model} for {env_name}...")
            
            for config in configurations:
                try:
                    score, agent = train_and_validate(model, env_name, config)
                    print(f"   Config: {config} | Validation Score: {score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        
                        # Save Best Model
                        save_path = f"saved_models/{model}_{env_name}_best.pth"
                        agent.save(save_path)
                        
                        # Update Registry
                        best_registry[f"{model}_{env_name}"] = {
                            "config": config,
                            "path": save_path,
                            "score": best_score
                        }
                        print(f"   -> New Best Found! Saved to {save_path}")
                        
                except Exception as e:
                    print(f"   !!! Error with config {config}: {e}")

    # Save Registry to JSON for safety
    with open("best_configs/final_best_registry.json", "w") as f:
        json.dump(best_registry, f, indent=4)
        
    print("\n=== OPTIMIZATION COMPLETE. REGISTRY SAVED. ===")

    # ==========================
    # PHASE 2: FINAL TESTING
    # ==========================
    print("\n=== PHASE 2: FINAL TESTING & RECORDING ===")
    
    # Iterate through the registry of winners
    for key, data in best_registry.items():
        # Parse key "SAC_CartPole-v1" -> "SAC", "CartPole-v1"
        # Be careful if env name has underscores. Standard Gym envs usually don't.
        # Safer way: we know the structure of our keys.
        parts = key.split('_')
        model_type = parts[0]
        # Join the rest back in case env name had underscores (though Gym uses dashes usually)
        env_name = "_".join(parts[1:]) 
        
        config = data['config']
        path = data['path']
        
        try:
            test_best_model(model_type, env_name, config, path)
        except Exception as e:
            print(f"!!! Error testing {key}: {e}")

    print("\n=== ALL ASSIGNMENT TASKS COMPLETE ===")