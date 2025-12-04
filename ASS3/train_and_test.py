# train.py (updated)
import gymnasium as gym
import torch
import numpy as np
import itertools
import json
import os
from datetime import datetime
import wandb

from a2c_agent import A2CAgent

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("videos", exist_ok=True)


def get_env_details(env_name):
    """Helper to get environment dimensions and discreteness."""
    env = gym.make(env_name)
    # handle potential scalar observation spaces robustly
    obs = env.observation_space
    if hasattr(obs, "shape") and obs.shape is not None:
        state_dim = obs.shape[0]
    else:
        state_dim = 1

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
    else:
        action_dim = env.action_space.shape[0]
        discrete = False
    env.close()
    return state_dim, action_dim, discrete


def train_agent(env_name, agent_class, params, max_episodes=500, use_wandb=False, job_type="train"):
    """Train an A2C agent with given parameters (now using configurable n-step)."""

    env = gym.make(env_name)
    state_dim, action_dim, discrete = get_env_details(env_name)

    # Build config dict expected by the a2c agent implementation
    config = {
        "learning_rate": params.get("lr", 7e-4),
        "gamma": params.get("gamma", 0.99),
        "entropy_coef": params.get("entropy_coef", 0.01),
        "value_coef": params.get("value_coef", 0.5),
        "n_steps": params.get("n_steps", 5),
        # optional:
        "max_grad_norm": params.get("max_grad_norm", 0.5),
        "hidden_dim": params.get("hidden_dim", 256),
    }

    # Initialize agent (only A2C implemented here)
    if agent_class == "A2C":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            is_continuous=not discrete,
            action_space=env.action_space,
            device=device,
        )
        # provide a convenience boolean used elsewhere in your code
        agent.discrete = discrete
    else:
        env.close()
        return None, []

    # ===================== WANDB INIT =====================
    if use_wandb:
        run_name = f"{agent_class}_{env_name}_{job_type}_lr{config['learning_rate']}_g{config['gamma']}_n{config['n_steps']}"
        wandb.init(
            project="RL_Assignment_Experiments",
            group=env_name,
            job_type=job_type,
            name=run_name,
            config={"env_name": env_name, "agent": agent_class, "max_episodes": max_episodes, **config},
            reinit=True,
        )
    # ======================================================

    episode_rewards = []
    episode_losses = []

    print(f"Training {agent_class} on {env_name} with n_steps={config['n_steps']}")

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        losses_this_episode = []

        while not done:
            # training -> sample stochastically; evaluation/deterministic will be used later
            action = agent.select_action(state, deterministic=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # store reward and mask (agent.store_reward should handle mask semantics)
            agent.store_reward(reward, done)

            # perform n-step update when buffer has at least n_steps
            if len(agent.rewards) >= getattr(agent, "n_steps", 5):
                loss = agent.update()
                losses_this_episode.append(loss)

            state = next_state
            total_reward += reward

        # final update for leftover steps (episode end)
        if len(agent.rewards) > 0:
            loss = agent.update()
            losses_this_episode.append(loss)

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(losses_this_episode) if losses_this_episode else 0.0)

        # Calculate rolling average
        avg_reward_50 = float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 1 else 0.0

        # WANDB logging
        if use_wandb:
            wandb.log({
                "episode": episode,
                "train_reward": float(total_reward),
                "avg_reward_50": avg_reward_50,
                "episode_steps": steps,
                "train_loss": float(np.mean(losses_this_episode)) if losses_this_episode else 0.0,
                "n_steps": config["n_steps"],
            })

        if (episode + 1) % 50 == 0 or episode == 0:
            print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.2f} | Avg50: {avg_reward_50:.2f} | Steps: {steps} | Loss: {np.mean(losses_this_episode) if losses_this_episode else 0.0:.4f}")

    env.close()
    if use_wandb:
        wandb.finish()

    # return trained agent and reward history
    return agent, episode_rewards


def hyperparameter_search(env_name, agent_class, n_trials=6):
    """Grid search for A2C hyperparameters (LR, Gamma, Entropy Coef, n_steps)"""

    # Example search space (expand as you like)
    gammas = [0.99]
    lrs = [7e-4]
    entropy_coefs = [0.01]
    n_steps_list = [1,5, 10]  # include n-step choices

    if agent_class == "A2C":
        if env_name == "CartPole-v1":
            gammas = [0.98, 0.99]
            lrs = [1e-3, 7e-4]
            entropy_coefs = [0.005, 0.01]
            n_steps_list = [5, 10]
        elif env_name == "Acrobot-v1":
            gammas = [0.98, 0.99]
            lrs = [7e-4, 3e-4]
            entropy_coefs = [0.01, 0.005]
            n_steps_list = [5, 10, 20]
        elif env_name == "MountainCar-v0":
            gammas = [0.999]
            lrs = [1e-4, 5e-5]
            entropy_coefs = [0.01]
            n_steps_list = [20, 50]
        elif env_name == "Pendulum-v1":
            gammas = [0.99]
            lrs = [7e-4, 1e-4]
            entropy_coefs = [0.0, 0.001]
            n_steps_list = [10, 20]

        param_combinations = list(itertools.product(gammas, lrs, entropy_coefs, n_steps_list))
    else:
        return None, float("-inf"), []

    best_params = None
    best_reward = float("-inf")
    results = []

    for gamma, lr, entropy_coef, n_steps in param_combinations[:n_trials]:
        params = {"gamma": gamma, "lr": lr, "entropy_coef": entropy_coef, "n_steps": n_steps}
        print(f"\nTesting params: {params}")

        agent, rewards = train_agent(
            env_name,
            agent_class,
            params,
            max_episodes=1000,  # shorter during search
            use_wandb=True,
            job_type="hp_search",
        )

        if agent is None:
            continue

        avg_reward = float(np.mean(rewards[-100:])) if len(rewards) > 0 else float("-inf")
        results.append({"params": params, "avg_reward": avg_reward})

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = params

    return best_params, best_reward, results


# ----------------------
# Testing functions (unchanged except deterministic eval)
# ----------------------
def test_agent(env_name, agent, n_tests=100, record_video=False, log_step_rewards=True):
    """Test trained agent with optional per-step reward statistics."""

    if record_video:
        video_folder = f"./videos/{env_name}_{agent.__class__.__name__}_{datetime.now().strftime('%Y%m%d%H%M')}"
        os.makedirs(video_folder, exist_ok=True)
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: x < 5, disable_logger=True)
    else:
        env = gym.make(env_name)

    test_rewards = []
    episode_lengths = []
    step_rewards_all = []   # store rewards per step

    # ensure agent network in eval mode
    try:
        agent.network.eval()
    except Exception:
        pass

    for test_ep in range(n_tests):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        step_rewards = []

        with torch.no_grad():
            while not done:
                action = agent.select_action(state, deterministic=True)
                if getattr(agent, "discrete", False) and isinstance(action, np.ndarray) and action.ndim > 0:
                    action = action.item()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                step_rewards.append(float(reward))

        test_rewards.append(total_reward)
        episode_lengths.append(steps)
        step_rewards_all.append(step_rewards)

        if test_ep % 20 == 0:
            print(f"Test {test_ep+1}/{n_tests}, Reward: {total_reward:.2f}, Steps: {steps}")
            if log_step_rewards:
                print(f"  First 10 step rewards: {step_rewards[:10]}")

    env.close()

    # save per-step reward curves
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    step_rewards_file = f"{results_dir}/{env_name}_{agent.__class__.__name__}_step_rewards.json"
    safe_json = [[float(r) for r in ep] for ep in step_rewards_all]
    with open(step_rewards_file, "w") as f:
        json.dump(safe_json, f, indent=2)
    print(f"\nSaved per-step reward data to {step_rewards_file}")

    return {
        "mean_reward": float(np.mean(test_rewards)),
        "std_reward": float(np.std(test_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "step_rewards_all": safe_json,
    }


# ===================== MAIN EXPERIMENT =====================
def run_full_experiment():
    environments = ["Pendulum-v1"] #"CartPole-v1", "Acrobot-v1", "MountainCar-v0", 
    agent_classes = ["A2C"]

    results = {}

    for env_name in environments:
        print(f"\n{'='*60}\nEnvironment: {env_name}\n{'='*60}\n")
        results[env_name] = {}

        for agent_class in agent_classes:
            if agent_class != "A2C":
                continue

            print(f"\nAgent: {agent_class}\n{'-'*60}")

            best_params, best_reward, search_results = hyperparameter_search(env_name, agent_class, n_trials=6)
            if best_params is None:
                print(f"Skipping {agent_class} for {env_name}")
                continue

            print(f"\nBest params: {best_params}")
            print(f"Best validation reward: {best_reward:.2f}")

            # Train with best params
            print(f"\nTraining {agent_class} on {env_name} with best params (500 episodes)...")
            agent, training_rewards = train_agent(env_name, agent_class, best_params, max_episodes=500, use_wandb=True, job_type="final_train")

            # Save model
            model_dir = "./models"
            torch.save(agent.network.state_dict(), f"{model_dir}/{env_name}_{agent_class}.pth")

            # Test agent
            print(f"\nTesting {agent_class} on {env_name} (100 episodes)...")
            test_results = test_agent(env_name, agent, n_tests=100, record_video=True)

            results[env_name][agent_class] = {"best_params": best_params, "test_results": test_results}

            print(f"\nTest Results for {agent_class} on {env_name}:")
            print(f"  Mean Reward: {test_results['mean_reward']:.2f} ± {test_results['std_reward']:.2f}")
            print(f"  Mean Episode Length: {test_results['mean_length']:.2f} ± {test_results['std_length']:.2f}")

    # Save all results
    results_dir = "./results"
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results_serializable = convert_to_serializable(results)
    with open(f"{results_dir}/experiment_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    print("\nExperiment Complete!")
    print(f"Results saved to {results_dir}/experiment_results.json")
    print("Models saved to ./models/")
    print("Videos saved to ./videos/")
    return results


if __name__ == "__main__":
    results = run_full_experiment()
