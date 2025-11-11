import time
import numpy as np
from grid_maze_env import GridMazeEnv
from policy_iteration import GridMazeMDP, policy_iteration

# Convert (x, y) -> state index and back
def pos_to_state(x, y, grid_size=5):
    return x * grid_size + y

def state_to_pos(s, grid_size=5):
    return divmod(s, grid_size)


def run_episode_with_random_goal():
    grid_size = 5

    # Random goal and bad cells
    all_states = list(range(grid_size * grid_size))
    goal_state = np.random.choice(all_states)
    bad_states = list(np.random.choice(
        [s for s in all_states if s != goal_state],
        size=2,
        replace=False
    ))

    # 1️⃣ Train policy iteration for this random layout
    mdp = GridMazeMDP(grid_size=grid_size, goal_state=goal_state, bad_states=bad_states)
    optimal_policy, V = policy_iteration(mdp)

    # 2️⃣ Map state index → grid coordinates
    goal_x, goal_y = state_to_pos(goal_state, grid_size)
    bad1_x, bad1_y = state_to_pos(bad_states[0], grid_size)
    bad2_x, bad2_y = state_to_pos(bad_states[1], grid_size)

    # 3️⃣ Create environment with policy overlay
    env = GridMazeEnv(grid_size=grid_size, render_mode="human", policy_overlay=optimal_policy)
    
    # Set the layout to match the MDP
    env.goal_pos = np.array([goal_x, goal_y])
    env.bad1_pos = np.array([bad1_x, bad1_y])
    env.bad2_pos = np.array([bad2_x, bad2_y])
    
    # Random start position (not goal or bad cells)
    available_positions = []
    for x in range(grid_size):
        for y in range(grid_size):
            pos = np.array([x, y])
            if not (np.array_equal(pos, env.goal_pos) or 
                    np.array_equal(pos, env.bad1_pos) or 
                    np.array_equal(pos, env.bad2_pos)):
                available_positions.append(pos)
    
    start_pos = available_positions[np.random.choice(len(available_positions))]
    env.agent_pos = start_pos
    env.done = False
    
    # Get initial observation without resetting (which would randomize positions)
    obs = env._get_obs()

    print(f"Goal = {env.goal_pos}")
    print(f"Bad cells = {env.bad1_pos}, {env.bad2_pos}")
    print(f"Start = {env.agent_pos}")
    print("\nRunning optimized policy...")
    print("Optimal Policy (0=→,1=↑,2=←,3=↓):")
    symbols = ['→', '↑', '←', '↓']
    for i, row in enumerate(optimal_policy):
        print(f"Row {i}: {' '.join(symbols[a] for a in row)}")
    print("------------------------------")
    
    # Initial render
    if env.render_mode == "human":
        env._render_frame()
    time.sleep(1)

    done = False
    step_count = 0

    while not done and step_count < 100:
        ax, ay = env.agent_pos
        action = optimal_policy[ax, ay]
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step_count+1}: ({ax},{ay}) -> Action={action}, Reward={reward}")
        time.sleep(0.4)
        step_count += 1

    env.close()
    print("Finished episode.\n")


# Run multiple episodes with different random goals
if __name__ == "__main__":
    num_runs = 3
    print(f"\n{'='*60}")
    print(f"Running {num_runs} episodes with random maze configurations")
    print(f"{'='*60}\n")
    
    for i in range(1, num_runs + 1):
        print(f"========== Run {i}/{num_runs} ==========")
        run_episode_with_random_goal()
       
    
    print(f"\n{'='*60}")
    print("All episodes completed!")
    print(f"{'='*60}")
