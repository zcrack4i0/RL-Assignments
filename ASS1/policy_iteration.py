"""
Policy Iteration for Grid Maze MDP

Implements the Policy Iteration dynamic programming algorithm to solve
the Grid Maze problem. The algorithm alternates between policy evaluation
(computing state values) and policy improvement (updating policy greedily)
until convergence.

Author: CMPS458 Assignment 1
Date: Fall 2025
"""

import numpy as np

class GridMazeMDP:
    """
    Markov Decision Process model for Grid Maze environment.
    
    Builds complete transition model P(s'|s,a) and reward function R(s).
    Assumes goal and bad cell positions are fixed for an episode.
    
    Args:
        grid_size: Size of square grid (default 5x5)
        goal_state: Index of goal state (or random if None)
        bad_states: List of bad state indices (or random if None)
        gamma: Discount factor for future rewards (default 0.95)
    """
    def __init__(self, grid_size=5, goal_state=None, bad_states=None, gamma=0.95):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  
        self.gamma = gamma

        # If goal or bad cells are not provided, pick random ones
        all_states = list(range(self.n_states))
        if goal_state is None:
            goal_state = np.random.choice(all_states)
        if bad_states is None:
            bad_states = list(np.random.choice(
                [s for s in all_states if s != goal_state],
                size=2,
                replace=False
            ))

        self.goal_state = goal_state
        self.bad_states = bad_states

        # Reward setup - step penalty for all states
        self.rewards = np.full(self.n_states, -1.0)
        
        # Terminal states get their terminal rewards
        self.rewards[self.goal_state] = 10.0
        for b in self.bad_states:
            self.rewards[b] = -10.0


        # Build transition probabilities
        self.P = self._build_transition_model()

    def _build_transition_model(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        moves = {
            0: np.array([0, 1]),   # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0])    # down
        }

        for s in range(self.n_states):
            # Terminal states (goal and bad cells) are absorbing
            if s == self.goal_state or s in self.bad_states:
                for a in range(self.n_actions):
                    P[s, a, s] = 1.0
                continue
            
            x, y = divmod(s, self.grid_size)

            for a in range(self.n_actions):
                # Determine perpendicular actions
                if a == 0:  # right → up, down
                    perpendiculars = [1, 3]
                elif a == 1:  # up → right, left
                    perpendiculars = [0, 2]
                elif a == 2:  # left → up, down
                    perpendiculars = [1, 3]
                elif a == 3:  # down → right, left
                    perpendiculars = [0, 2]

                all_actions = [a] + perpendiculars
                probs = [0.7, 0.15, 0.15]

                valid_actions = []
                valid_probs = []

                for a2, p in zip(all_actions, probs):
                    nx, ny = x + moves[a2][0], y + moves[a2][1]
                    # If the move would leave the grid, stay in the same cell
                    if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                        nx, ny = x, y  # self-transition

                    valid_actions.append((a2, p, nx, ny))

                # Normalize probabilities (optional but safe)
                total_prob = sum(p for _, p, _, _ in valid_actions)
                for a2, p, nx, ny in valid_actions:
                    ns = nx * self.grid_size + ny
                    P[s, a, ns] += p / total_prob

        return P

    def _move(self, x, y, action):
        if action == 0:  # right
            y += 1
        elif action == 1:  # up
            x -= 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # down
            x += 1
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)
        return x, y



def policy_iteration(env, theta=1e-6, max_iterations=1000):
    nS, nA = env.n_states, env.n_actions
    P, R, gamma = env.P, env.rewards, env.gamma

    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)  # Initialize with all zeros (right)

    for it in range(max_iterations):
        # --- Policy Evaluation ---
        eval_iter = 0
        while True:
            delta = 0
            for s in range(nS):
                v = V[s]
                a = policy[s]
                # Bellman equation: V(s) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
                # Using immediate reward model: reward obtained when leaving state s
                V[s] = sum(P[s, a, s2] * (R[s] + gamma * V[s2]) for s2 in range(nS))
                delta = max(delta, abs(v - V[s]))
            eval_iter += 1
            if delta < theta:
                break

        # --- Policy Improvement ---
        policy_stable = True
        for s in range(nS):
            # Skip terminal states
            if s == env.goal_state or s in env.bad_states:
                continue
                
            old_action = policy[s]
            action_values = np.zeros(nA)

            for a in range(nA):
                # Calculate expected return for each action
                action_values[a] = sum(P[s, a, s2] * (R[s] + gamma * V[s2]) for s2 in range(nS))

            # Choose action with highest expected return
            policy[s] = np.argmax(action_values)

            if old_action != policy[s]:
                policy_stable = False

        if policy_stable:
            print(f"✅ Converged after {it + 1} iterations")
            break

    return policy.reshape((env.grid_size, env.grid_size)), V.reshape((env.grid_size, env.grid_size))


if __name__ == "__main__":
    env = GridMazeMDP(grid_size=5)
    policy, V = policy_iteration(env)

    print("Optimal Policy (0=→, 1=↑, 2=←, 3=↓):")
    print(policy)
    print("\nState Values:")
    print(np.round(V, 2))
