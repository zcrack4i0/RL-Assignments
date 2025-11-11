import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class GridMazeEnv(gym.Env):    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=5, render_mode=None, policy_overlay=None):
        super(GridMazeEnv, self).__init__()

        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0=right, 1=up, 2=left, 3=down
        low = np.zeros(8, dtype=np.int32)
        high = np.array([grid_size - 1] * 8, dtype=np.int32)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

        self.render_mode = render_mode
        self.window_size = 400
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        self.policy_overlay = policy_overlay

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start, goal, and 2 bad cells
        cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        np.random.shuffle(cells)
        self.agent_pos = np.array(cells[0])
        self.goal_pos = np.array(cells[1])
        self.bad1_pos = np.array(cells[2])
        self.bad2_pos = np.array(cells[3])

        obs = np.concatenate([self.agent_pos, self.goal_pos, self.bad1_pos, self.bad2_pos])
        self.done = False

        if self.render_mode == "human":
            self._render_frame()

        return obs, {}


    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Define possible perpendicular directions for each action
        if action == 0:  # Right → Up, Down
            perpendiculars = [1, 3]
        elif action == 1:  # Up → Right, Left
            perpendiculars = [0, 2]
        elif action == 2:  # Left → Up, Down
            perpendiculars = [1, 3]
        elif action == 3:  # Down → Right, Left
            perpendiculars = [0, 2]

        all_actions = [action] + perpendiculars
        probs = [0.7, 0.15, 0.15]

        # Define movement vectors (x=row, y=col)
        moves = {
            0: np.array([0, 1]),   # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0])    # down
        }

        # Filter out moves that go out of the grid
        valid_actions = []
        valid_probs = []

        for a, p in zip(all_actions, probs):
            new_pos = self.agent_pos + moves[a]
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                valid_actions.append(a)
                valid_probs.append(p)

        # Re-normalize probabilities
        valid_probs = np.array(valid_probs)
        valid_probs = valid_probs / valid_probs.sum()

        # Choose an action respecting bounds
        chosen_action = np.random.choice(valid_actions, p=valid_probs)

        # Move the agent
        self.agent_pos += moves[chosen_action]

        # Apply rewards - step penalty for each move
        reward = -1.0
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10.0
            self.done = True
        elif np.array_equal(self.agent_pos, self.bad1_pos) or np.array_equal(self.agent_pos, self.bad2_pos):
            reward = -10.0
            self.done = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, self.done, False, {}
    def _get_obs(self):
        return np.concatenate([self.agent_pos, self.goal_pos, self.bad1_pos, self.bad2_pos])

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _draw_policy_arrows(self, surface):
        if self.policy_overlay is None:
            return

        arrow_color = (100, 100, 100)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                action = self.policy_overlay[x, y]
                cx = y * self.cell_size + self.cell_size // 2
                cy = x * self.cell_size + self.cell_size // 2
                size = self.cell_size // 4

                if action == 0:  # right →
                    pygame.draw.line(surface, arrow_color, (cx - size, cy), (cx + size, cy), 2)
                    pygame.draw.polygon(surface, arrow_color, [(cx + size, cy), (cx + size - 5, cy - 5), (cx + size - 5, cy + 5)])
                elif action == 1:  # up ↑
                    pygame.draw.line(surface, arrow_color, (cx, cy + size), (cx, cy - size), 2)
                    pygame.draw.polygon(surface, arrow_color, [(cx, cy - size), (cx - 5, cy - size + 5), (cx + 5, cy - size + 5)])
                elif action == 2:  # left ←
                    pygame.draw.line(surface, arrow_color, (cx + size, cy), (cx - size, cy), 2)
                    pygame.draw.polygon(surface, arrow_color, [(cx - size, cy), (cx - size + 5, cy - 5), (cx - size + 5, cy + 5)])
                elif action == 3:  # down ↓
                    pygame.draw.line(surface, arrow_color, (cx, cy - size), (cx, cy + size), 2)
                    pygame.draw.polygon(surface, arrow_color, [(cx, cy + size), (cx - 5, cy + size - 5), (cx + 5, cy + size - 5)])

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

        # Draw goal (green)
        gx, gy = self.goal_pos
        pygame.draw.rect(self.window, (0, 255, 0),
                         pygame.Rect(gy * self.cell_size, gx * self.cell_size, self.cell_size, self.cell_size))

        # Draw bad cells (red)
        for bx, by in [self.bad1_pos, self.bad2_pos]:
            pygame.draw.rect(self.window, (255, 0, 0),
                             pygame.Rect(by * self.cell_size, bx * self.cell_size, self.cell_size, self.cell_size))
        # Draw policy arrows if provided
        self._draw_policy_arrows(self.window)
        # Draw agent (blue)
        ax, ay = self.agent_pos
        pygame.draw.rect(self.window, (0, 0, 255),
                         pygame.Rect(ay * self.cell_size, ax * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_rgb_array(self):
        # Ensure pygame is initialized for drawing operations
        if not pygame.get_init():
            pygame.init()

        # Draw to an offscreen surface
        surface = pygame.Surface((self.window_size, self.window_size))
        surface.fill((255, 255, 255))

        # Grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, (200, 200, 200), rect, 1)

        # Goal (green)
        gx, gy = self.goal_pos
        pygame.draw.rect(surface, (0, 255, 0),
                         pygame.Rect(gy * self.cell_size, gx * self.cell_size, self.cell_size, self.cell_size))

        # Bad cells (red)
        for bx, by in [self.bad1_pos, self.bad2_pos]:
            pygame.draw.rect(surface, (255, 0, 0),
                             pygame.Rect(by * self.cell_size, bx * self.cell_size, self.cell_size, self.cell_size))

        # Policy arrows overlay
        # Temporarily swap window to reuse drawing function if needed
        prev_window = self.window
        try:
            self.window = surface
            self._draw_policy_arrows(surface)
        finally:
            self.window = prev_window

        # Agent (blue)
        ax, ay = self.agent_pos
        pygame.draw.rect(surface, (0, 0, 255),
                         pygame.Rect(ay * self.cell_size, ax * self.cell_size, self.cell_size, self.cell_size))

        # Convert to RGB array (H, W, 3)
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
