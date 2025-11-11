
import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from grid_maze_env import GridMazeEnv
from policy_iteration import GridMazeMDP, policy_iteration

def pos_to_state(x, y, grid_size=5):
    return x * grid_size + y

def state_to_pos(s, grid_size=5):

    return divmod(s, grid_size)

def record_episode(episode_num, grid_size=5, video_folder="./videos"):

    # Generate random goal and bad cells
    all_states = list(range(grid_size * grid_size))
    goal_state = np.random.choice(all_states)
    bad_states = list(np.random.choice(
        [s for s in all_states if s != goal_state],
        size=2,
        replace=False
    ))

    # Train policy iteration for this configuration
    print(f"\n{'='*60}")
    print(f"Episode {episode_num}: Training Policy Iteration")
    print(f"{'='*60}")
    mdp = GridMazeMDP(grid_size=grid_size, goal_state=goal_state, bad_states=bad_states)
    optimal_policy, V = policy_iteration(mdp)

    # Convert state indices to grid coordinates
    goal_x, goal_y = state_to_pos(goal_state, grid_size)
    bad1_x, bad1_y = state_to_pos(bad_states[0], grid_size)
    bad2_x, bad2_y = state_to_pos(bad_states[1], grid_size)

    print(f"Goal: ({goal_x}, {goal_y})")
    print(f"Bad cells: ({bad1_x}, {bad1_y}), ({bad2_x}, {bad2_y})")
    
    # Ensure video output folder exists
    os.makedirs(video_folder, exist_ok=True)

    # Create environment with RecordVideo wrapper
    env = GridMazeEnv(grid_size=grid_size, render_mode="rgb_array", policy_overlay=optimal_policy)
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        episode_trigger=lambda x: True,  # Record every episode
        name_prefix=f"grid_maze_episode_{episode_num}"
    )
    # Important: start a new episode for RecordVideo to open a file
    _obs, _info = env.reset()
    
    # Set the maze layout to match the trained policy
    env.unwrapped.goal_pos = np.array([goal_x, goal_y])
    env.unwrapped.bad1_pos = np.array([bad1_x, bad1_y])
    env.unwrapped.bad2_pos = np.array([bad2_x, bad2_y])
    
    # Random start position (not goal or bad cells)
    available_positions = []
    for x in range(grid_size):
        for y in range(grid_size):
            pos = np.array([x, y])
            if not (np.array_equal(pos, env.unwrapped.goal_pos) or 
                    np.array_equal(pos, env.unwrapped.bad1_pos) or 
                    np.array_equal(pos, env.unwrapped.bad2_pos)):
                available_positions.append(pos)
    
    start_pos = available_positions[np.random.choice(len(available_positions))]
    env.unwrapped.agent_pos = start_pos
    env.unwrapped.done = False
    # Render once after positioning to ensure a frame is captured
    try:
        env.render()
    except Exception:
        pass
    
    # Run the episode
    print(f"Start: ({start_pos[0]}, {start_pos[1]})")
    print(f"\nRecording episode {episode_num}...")
    done = False
    step_count = 0
    total_reward = 0
    
    while not done and step_count < 100:
        ax, ay = env.unwrapped.agent_pos
        action = optimal_policy[ax, ay]
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        print(f"  Step {step_count}: ({ax},{ay}) -> Action={action}, Reward={reward:.1f}")
    
    env.close()
    
    print(f"✅ Episode {episode_num} completed!")
    print(f"   Total steps: {step_count}")
    print(f"   Total reward: {total_reward:.1f}")
    print(f"   Result: {'GOAL REACHED!' if total_reward > 0 else 'Hit bad cell'}")
    
    return step_count, total_reward

def main():
    """Record multiple episodes with different random configurations."""
    num_episodes = 10
    grid_size = 5
    video_folder = "./videos"
    
    print("="*60)
    print("Grid Maze - Recording Agent Performance")
    print("="*60)
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Number of Episodes: {num_episodes}")
    print(f"Video Folder: {video_folder}")
    print("="*60)
    
    results = []
    for i in range(1, num_episodes + 1):
        steps, reward = record_episode(i, grid_size, video_folder)
        results.append((i, steps, reward))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Episode':<10} {'Steps':<10} {'Reward':<10} {'Result'}")
    print("-"*60)
    for ep, steps, reward in results:
        result = "SUCCESS" if reward > 0 else "FAILURE"
        print(f"{ep:<10} {steps:<10} {reward:<10.1f} {result}")
    
    avg_steps = np.mean([s for _, s, r in results if r > 0])
    success_rate = sum(1 for _, _, r in results if r > 0) / len(results) * 100
    
    print("-"*60)
    print(f"Success Rate: {success_rate:.1f}%")
    if success_rate > 0:
        print(f"Average Steps (successful): {avg_steps:.1f}")
    print("="*60)
    print(f"\n📹 Videos saved to: {video_folder}")

if __name__ == "__main__":
    main()
