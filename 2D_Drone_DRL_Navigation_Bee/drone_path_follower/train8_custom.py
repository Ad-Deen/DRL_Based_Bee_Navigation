"""
path_follower_sac.py

Path Following with SAC + LiDAR for dynamic obstacle avoidance.
Merged with Procedural World Generation (Maze/Sparse) and A* + LOS + Chaikin path pipeline.
"""

import os
import sys
import heapq
import time
from collections import deque
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import csv 
from SAC_policy import CustomSACPolicy 

# =====================================================================
# CONFIGURATION
# =====================================================================
# MODE = "train"                  # Set to "train" to train, "test" to test
MODE = "test"    
TEST_EPISODES = 10
MAP_MODE = "sparse"                 # "maze", "sparse", or "mixed"
VELOCITY_EXPERT_PATH = "/home/deen/ppo_test/drone_flight_expert/velocity_expert_final"
SAVE_DIR = "/home/deen/ppo_test/drone_path_follower/"
MODEL_NAME = "path_follower"
PLOT_INTERVAL = 5
RENDER_EPISODE_FREQ = 50
SAVE_EPISODE_INTERVAL = 10
LOAD_EXISTING_MODEL = True
# LOAD_EXISTING_MODEL = False

DYNAMIC_OBJ_MIN_SIZE = 1.0
DYNAMIC_OBJ_MAX_SIZE = 3.2

WORLD_SIZE = 50.0
MAX_DYNAMIC_OBSTACLES = 0      # Set to e.g. 5 for moving obstacles
MAX_STEPS = 3000
DRONE_RADIUS = 0.3
GRID_RES = 1.0                 # 1m per cell → 50×50 grid for A*

# ---- CLEARANCE CONSTRAINT ----
MIN_CLEARANCE_CELLS = 2        # Path stays >= 2.0m from walls

# =====================================================================
# 1. WORLD GENERATION & PATH PLANNING (From vin_world_gen.py)
# =====================================================================
def generate_maze_recursive(H, W, cell=6):
    n = min(H, W) // cell
    occ = np.ones((H, W), dtype=np.uint8)
    visited = np.zeros((n, n), dtype=bool)
    start = (np.random.randint(n), np.random.randint(n))
    stack = [start]
    visited[start] = True
    sr, sc = start
    occ[sr*cell + 1 : sr*cell + cell, sc*cell + 1 : sc*cell + cell] = 0
    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not visited[nr, nc]:
                neighbors.append((nr, nc, dr, dc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc, dr, dc = neighbors[np.random.randint(len(neighbors))]
        occ[nr*cell + 1 : nr*cell + cell, nc*cell + 1 : nc*cell + cell] = 0
        if dr != 0:
            wall_row = r*cell + cell if dr > 0 else r*cell
            occ[wall_row, c*cell + 1 : c*cell + cell] = 0
        else:
            wall_col = c*cell + cell if dc > 0 else c*cell
            occ[r*cell + 1 : r*cell + cell, wall_col] = 0
        visited[nr, nc] = True
        stack.append((nr, nc))
    return occ



def generate_sparse(H, W):
    occ = np.zeros((H, W), dtype=np.uint8)
    n_obs = np.random.randint(10, 25)
    for _ in range(n_obs):
        for _attempt in range(30):
            cx = np.random.randint(3, W - 3)
            cy = np.random.randint(3, H - 3)
            r = np.random.randint(2, 5)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = (xx*xx + yy*yy) <= r*r
            region = occ[cy-r:cy+r+1, cx-r:cx+r+1]
            if region.shape != mask.shape or region[mask].sum() > 0:
                continue
            region[mask] = 1
            break
    return occ

def compute_clearance_mask(occ, min_clearance_cells):
    free = (occ == 0).astype(np.uint8)
    dist_cells = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    safe_mask = (dist_cells - 0.5 >= min_clearance_cells).astype(np.uint8)
    return safe_mask, dist_cells

def is_connected(mask, start, goal):
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue = deque([start])
    visited[start] = True
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] == 1 and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return False

def sample_start_goal(safe_mask, min_dist=25, max_attempts=200):
    free = np.argwhere(safe_mask == 1)
    if len(free) < 10: return None, None
    for _ in range(max_attempts):
        s = tuple(free[np.random.randint(len(free))])
        g = tuple(free[np.random.randint(len(free))])
        if np.hypot(s[0]-g[0], s[1]-g[1]) < min_dist: continue
        if is_connected(safe_mask, s, g): return s, g
    return None, None

def generate_world(mode=MAP_MODE, H=50, W=50, min_clearance_cells=2):
    for _attempt in range(30):
        actual = ('maze' if np.random.random() < 0.5 else 'sparse') if mode == 'mixed' else mode
        occ = generate_maze_recursive(H, W) if actual == 'maze' else generate_sparse(H, W)
        safe_mask, dist_cells = compute_clearance_mask(occ, min_clearance_cells)

        # Pad the boundaries so paths don't hug the edge of the world
        occ_padded = np.pad(occ, 1, mode='constant', constant_values=1)
        safe_mask_padded, dist_cells_padded = compute_clearance_mask(occ_padded, min_clearance_cells)
        safe_mask = safe_mask_padded[1:-1, 1:-1]
        dist_cells = dist_cells_padded[1:-1, 1:-1]

        if safe_mask.sum() < 200: continue
        s, g = sample_start_goal(safe_mask)
        if s is not None:
            return occ, safe_mask, dist_cells, s, g, actual
    # Fallback
    occ = np.zeros((H, W), dtype=np.uint8)
    safe_mask, dist_cells = compute_clearance_mask(occ, min_clearance_cells)
    return occ, safe_mask, dist_cells, (5, 5), (H-5, W-5), 'empty'

def astar(occ, start_rc, goal_rc, safe_mask):
    H, W = occ.shape
    if safe_mask[start_rc] != 1 or safe_mask[goal_rc] != 1: return None
    pq = [(0.0, start_rc[0], start_rc[1])]
    came_from = {}
    cost = {start_rc: 0.0}
    while pq:
        _, r, c = heapq.heappop(pq)
        if (r, c) == goal_rc: break
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and safe_mask[nr, nc] == 1:
                step = 1.4142 if dr != 0 and dc != 0 else 1.0
                new_cost = cost[(r, c)] + step
                if (nr, nc) not in cost or new_cost < cost[(nr, nc)]:
                    cost[(nr, nc)] = new_cost
                    h = np.hypot(nr - goal_rc[0], nc - goal_rc[1])
                    heapq.heappush(pq, (new_cost + h, nr, nc))
                    came_from[(nr, nc)] = (r, c)
    if goal_rc not in came_from and goal_rc != start_rc: return None
    path = [goal_rc]
    while path[-1] != start_rc:
        path.append(came_from[path[-1]])
    return path[::-1]

def _segment_is_clear(p1, p2, dist_cells, grid_res, min_clearance_m, oversample=4):
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    L = float(np.linalg.norm(p2 - p1))
    if L < 1e-6: return True
    n = max(2, int(np.ceil(L / (grid_res / oversample))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    xs = p1[0] + ts * (p2[0] - p1[0])
    ys = p1[1] + ts * (p2[1] - p1[1])
    cs = np.clip(np.round(xs / grid_res - 0.5).astype(int), 0, dist_cells.shape[1] - 1)
    rs = np.clip(np.round(ys / grid_res - 0.5).astype(int), 0, dist_cells.shape[0] - 1)
    return bool(np.all((dist_cells[rs, cs] - 0.5) * grid_res >= min_clearance_m))

def simplify_path_los(path_xy, dist_cells, grid_res, min_clearance_m, stride=1):
    path_xy = np.asarray(path_xy, dtype=np.float32)
    if len(path_xy) <= 2: return path_xy
    idx = list(range(0, len(path_xy), stride))
    if idx[-1] != len(path_xy) - 1: idx.append(len(path_xy) - 1)
    pts = path_xy[idx]
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_is_clear(pts[i], pts[j], dist_cells, grid_res, min_clearance_m):
            j -= 1
        out.append(pts[j])
        i = j
    return np.asarray(out, dtype=np.float32)

def smooth_path_chaikin(path_xy, n_iter=4):
    path = np.asarray(path_xy, dtype=np.float32)
    if len(path) < 2: return path
    for _ in range(n_iter):
        new = [path[0]]
        for i in range(len(path) - 1):
            p, q = path[i], path[i + 1]
            new.append(0.75 * p + 0.25 * q)
            new.append(0.25 * p + 0.75 * q)
        new.append(path[-1])
        path = np.array(new, dtype=np.float32)
    return path

def discrete_to_continuous_map(occ, grid_res=1.0):
    H, W = occ.shape
    out = []
    for r in range(H):
        for c in range(W):
            if occ[r, c] == 1:
                x0, y0 = c * grid_res, r * grid_res
                out.append((x0, y0, x0 + grid_res, y0 + grid_res))
    return out

def discrete_path_to_continuous(path_rc, grid_res=1.0):
    pts = np.array(path_rc, dtype=np.float32)
    return np.stack([(pts[:, 1] + 0.5) * grid_res, (pts[:, 0] + 0.5) * grid_res], axis=1)

def point_aabb_distance(x, y, aabb):
    x0, y0, x1, y1 = aabb
    cx = float(np.clip(x, x0, x1))
    cy = float(np.clip(y, y0, y1))
    return np.hypot(x - cx, y - cy)

def clip_to_safe(path_xy, obstacles_xy, min_clearance_m):
    out = path_xy.copy()
    for i, (x, y) in enumerate(path_xy):
        min_d = float('inf')
        worst_aabb = None
        for aabb in obstacles_xy:
            d = point_aabb_distance(x, y, aabb)
            if d < min_d:
                min_d = d
                worst_aabb = aabb
        if min_d < min_clearance_m and worst_aabb is not None:
            x0, y0, x1, y1 = worst_aabb
            cx = np.clip(x, x0, x1)
            cy = np.clip(y, y0, y1)
            dx, dy = x - cx, y - cy
            dist = np.hypot(dx, dy)
            if dist < 1e-6:
                cx, cy = (x0+x1)/2, (y0+y1)/2
                dx, dy = x - cx, y - cy
                dist = np.hypot(dx, dy) + 1e-6
            out[i] = (cx + dx/dist * min_clearance_m, cy + dy/dist * min_clearance_m)
    return out

# =====================================================================
# 2. ENVIRONMENT
# =====================================================================
class ActionRepeat(gym.Wrapper):
    def __init__(self, env, k=5):
        super().__init__(env)
        self.k = k

    def step(self, a):
        total_r = 0.0
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(a)
            total_r += r
            if term or trunc:
                break
        return obs, total_r, term, trunc, info

    # --- ADD THIS BLOCK ---
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    # ----------------------

class PathFollowingEnv(gym.Env):
    def __init__(self, velocity_expert, render_mode=None):
        super().__init__()
        self.velocity_expert = velocity_expert
        self.render_mode = render_mode

        self.world_size = WORLD_SIZE
        self.max_steps = MAX_STEPS
        self.grid_res = GRID_RES
        self.grid_h = int(self.world_size / self.grid_res)
        self.grid_w = int(self.world_size / self.grid_res)

        # self.gamma = 0.998
        self.prev_phi = 0.0

        self.smooth_deadzone = 0.5    # |Δa| below this → no penalty
        self.smooth_max_delta = 2.5   # |Δa| at which penalty saturates
        self.smooth_max_pen = 0.5     # penalty magnitude at saturation

        # Physics
        self.mass = 3.0
        self.max_thrust_each = 5.0 * 9.8
        self.g = 9.8
        self.dt = 0.02
        self.d = 0.4
        self.I = 0.3
        self.drone_radius = DRONE_RADIUS

        # LiDAR
        self.n_rays = 180
        self.max_range = 20.0
        self.ray_angles = np.linspace(0, 2*np.pi, self.n_rays, endpoint=False)

        # Drone state
        self.x = 0.0; self.y = 0.0
        self.vx = 0.0; self.vy = 0.0
        self.theta = 0.0; self.vtheta = 0.0
        self.target_vx = 0.0; self.target_vy = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_thrust = (0.0, 0.0)

        # Goal + path
        self.goal_x = 0.0; self.goal_y = 0.0
        self.goal_radius = 1.5
        self.path_xy = None
        
        # Obstacles
        self.static_obstacles = []   # Now AABBs (x0, y0, x1, y1)
        self.dynamic_obstacles = []  # Circles (x, y, r, vx, vy)

        # Book-keeping
        self.step_counter = 0
        self._episode_reward = 0.0
        self._episode_length = 0

        self.current_lidar = np.ones(self.n_rays, dtype=np.float32)


        # Spaces
        self.n_state = 15
        obs_dim = self.n_rays + self.n_state    # 180 + 15 = 195
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

        # History
        self.last_accel = 0.0
        self.prev_vx = 0.0; self.prev_vy = 0.0

        self.dist_to_path_grid = None
        self.path_cumulative_arc = None
        self._path_cursor = -1
        

        self.last_min_lidar = 0.0
        
    

    def _compute_path_distance_grid(self, path_xy, cell=0.5):
        H = int(self.world_size / cell) + 1
        W = int(self.world_size / cell) + 1
        dense = []
        for i in range(len(path_xy) - 1):
            n = max(2, int(np.linalg.norm(path_xy[i+1] - path_xy[i]) / cell))
            for t in np.linspace(0, 1, n, endpoint=False):
                dense.append(path_xy[i] * (1-t) + path_xy[i+1] * t)
        dense = np.array(dense, dtype=np.float32)
        ys, xs = np.meshgrid(np.arange(H) * cell, np.arange(W) * cell, indexing='ij')
        grid_pts = np.stack([xs.flatten(), ys.flatten()], axis=1)
        dists = np.zeros(len(grid_pts), dtype=np.float32)
        chunk = 2000
        for s in range(0, len(grid_pts), chunk):
            e = min(s + chunk, len(grid_pts))
            d = np.linalg.norm(grid_pts[s:e, None, :] - dense[None, :, :], axis=2)
            dists[s:e] = d.min(axis=1)
        return dists.reshape(H, W)

    def _find_closest_path_idx(self, agent_xy, window=25, reset_threshold=10.0):
        if self._path_cursor is None or self._path_cursor < 0:
            dists_sq = np.sum((self.path_xy - agent_xy)**2, axis=1)
            idx = int(np.argmin(dists_sq))
        else:
            lo = max(0, self._path_cursor - window)
            hi = min(len(self.path_xy), self._path_cursor + window + 1)
            dists_sq = np.sum((self.path_xy[lo:hi] - agent_xy)**2, axis=1)
            local_idx = int(np.argmin(dists_sq))
            idx = lo + local_idx
            if dists_sq[local_idx] > reset_threshold ** 2:
                dists_sq = np.sum((self.path_xy - agent_xy)**2, axis=1)
                idx = int(np.argmin(dists_sq))
        self._path_cursor = idx
        return idx          

    def _extract_lookahead_at(self, agent_xy, closest_idx, lookahead_dist=3.0):
        if len(self.path_xy) < 2: return self.path_xy[0], np.array([1.0, 0.0], dtype=np.float32)
        accum = 0.0
        for i in range(closest_idx, len(self.path_xy) - 1):
            seg = self.path_xy[i+1] - self.path_xy[i]
            seg_len = float(np.linalg.norm(seg))
            if accum + seg_len >= lookahead_dist:
                frac = (lookahead_dist - accum) / (seg_len + 1e-8)
                waypoint = self.path_xy[i] + frac * seg
                tangent = seg / (seg_len + 1e-8)
                return waypoint.astype(np.float32), tangent.astype(np.float32)
            accum += seg_len
        tangent = self.path_xy[-1] - self.path_xy[-2]
        tangent /= (np.linalg.norm(tangent) + 1e-8)
        return self.path_xy[-1], tangent.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        
        
        self.prev_vx = 0.0; self.prev_vy = 0.0
        
        self._path_cursor = -1

        # 1. GENERATE WORLD
        occ, safe_mask, dist_cells, s_rc, g_rc, mode_name = generate_world(
            mode=MAP_MODE, H=self.grid_h, W=self.grid_w, min_clearance_cells=MIN_CLEARANCE_CELLS
        )
        
        # 2. A* PATH PLANNING
        path_rc = astar(occ, s_rc, g_rc, safe_mask)
        if path_rc is None:
            path_rc = [s_rc, g_rc]

        # 3. CONTINUOUS CONVERSION & SMOOTHING
        self.static_obstacles = discrete_to_continuous_map(occ, self.grid_res)
        self.static_arr = np.array(self.static_obstacles, dtype=np.float32)
        path_xy = discrete_path_to_continuous(path_rc, self.grid_res)
        
        path_simple = simplify_path_los(path_xy, dist_cells, self.grid_res, MIN_CLEARANCE_CELLS * self.grid_res)
        path_smooth = smooth_path_chaikin(path_simple, n_iter=4)
        self.path_xy = clip_to_safe(path_smooth, self.static_obstacles, MIN_CLEARANCE_CELLS * self.grid_res)

        # 4. SET START & GOAL
        self.x, self.y = float(self.path_xy[0][0]), float(self.path_xy[0][1])
        self.goal_x, self.goal_y = float(self.path_xy[-1][0]), float(self.path_xy[-1][1])

        # 5. COMPUTE ARC LENGTHS
        diff = np.diff(self.path_xy, axis=0)
        seg_lengths = np.linalg.norm(diff, axis=1)
        self.path_cumulative_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.total_arc_length = float(self.path_cumulative_arc[-1]) + 1e-8
        self.dist_to_path_grid = self._compute_path_distance_grid(self.path_xy)

        # 6. PLACE DYNAMIC OBSTACLES
        self.dynamic_obstacles = []
        for _ in range(MAX_DYNAMIC_OBSTACLES):
            for _attempt in range(50):
                ox, oy = np.random.uniform(5.0, self.world_size - 5.0, 2)
                r = np.random.uniform(DYNAMIC_OBJ_MIN_SIZE, DYNAMIC_OBJ_MAX_SIZE)
                if np.hypot(ox - self.x, oy - self.y) < r + 3.0: continue
                if np.hypot(ox - self.goal_x, oy - self.goal_y) < r + 2.5: continue
                # Ensure dynamic obstacles don't spawn inside static AABBs
                if any(point_aabb_distance(ox, oy, aabb) < r + 0.5 for aabb in self.static_obstacles): continue
                speed = np.random.uniform(1.0, 3.0)
                angle = np.random.uniform(0, 2*np.pi)
                self.dynamic_obstacles.append((ox, oy, r, speed*np.cos(angle), speed*np.sin(angle)))
                break

        # 7. RESET PHYSICS
        self.vx = 0.0; self.vy = 0.0
        self.theta = 0.0; self.vtheta = 0.0
        self.target_vx = 0.0; self.target_vy = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_counter = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self.last_accel = 0.0 
        self.current_lidar = self._compute_lidar()

        # Initialize position-based potential at start
        self.prev_phi = self._compute_potential(self.x, self.y)

        return self._get_obs(), {}
    
# ---------------------- New Potential func -------------------------
    def _compute_potential(self, x, y):
        d_path = self._interp_dist_to_path(x, y)
        d_obs  = float(np.min(self.current_lidar)) * self.max_range

        if d_path < 1.0:
            # slow fall: 20 → 15, slope −5/m
            path_term = 20.0 - 5.0 * d_path
        elif d_path < 2.0:
            # hard fall: 15 → 0, slope −15/m
            path_term = 15.0 - 15.0 * (d_path - 1.0)
        elif d_path < 10.0:
            # negative fall: 0 → −30, slope −3.75/m
            path_term = -30.0 * (d_path - 2.0) / 8.0
        else:
            # clamp
            path_term = -30.0

        # --- Obstacle repulsion ---
        if d_obs < 10.0:
            obs_term = -30.0 * (1.0 - d_obs / 10.0)
        else:
            obs_term = 0.0

        # --- Linear arc progression ---
        # arc = cumulative path length to closest point on the path
        # +k·arc  →  +k per meter of forward progress, −k per meter of regression,
        # 0 for lateral. Total episode arc reward = k · total_arc_length.
        closest_idx = self._find_closest_path_idx(np.array([x, y], dtype=np.float32))
        arc = float(self.path_cumulative_arc[closest_idx])
        progress_term = 1.0 * arc                    # ← k = 1.0

        d_goal = np.hypot(x - self.goal_x, y - self.goal_y)
        goal_term = -0.5 * min(d_goal, 5.0)

        return path_term + goal_term + obs_term + progress_term

    def _interp_dist_to_path(self, x, y):
        """Bilinear interp of the precomputed distance grid — smooth gradient, no 0.5 m cell jumps."""
        if self.dist_to_path_grid is None:
            return 999.0
        cell = 0.5
        H, W = self.dist_to_path_grid.shape
        fx = np.clip(x / cell - 0.5, 0.0, W - 1.001)
        fy = np.clip(y / cell - 0.5, 0.0, H - 1.001)
        x0 = int(fx); y0 = int(fy)
        tx = fx - x0; ty = fy - y0
        x1 = min(x0 + 1, W - 1); y1 = min(y0 + 1, H - 1)
        return float(
            self.dist_to_path_grid[y0, x0] * (1-tx)*(1-ty) +
            self.dist_to_path_grid[y0, x1] * tx*(1-ty) +
            self.dist_to_path_grid[y1, x0] * (1-tx)*ty +
            self.dist_to_path_grid[y1, x1] * tx*ty
        )
    
# ------------------------------------- end --------------------------------

    def _compute_lidar(self):
        rays = np.ones(self.n_rays, dtype=np.float32)
        global_angles = self.theta + self.ray_angles
        ray_dirs = np.stack([np.cos(global_angles), np.sin(global_angles)], axis=1)

        # Vectorized AABB Raycast
        if len(self.static_arr) > 0:
            b = self.static_arr
            R = self.max_range
            # Cull AABBs outside sensor range
            b = b[(b[:,2] > self.x-R) & (b[:,0] < self.x+R) & (b[:,3] > self.y-R) & (b[:,1] < self.y+R)]
            
            if len(b) > 0:
                dx = np.where(np.abs(ray_dirs[:,0]) < 1e-8, 1e-8, ray_dirs[:,0])[None]
                dy = np.where(np.abs(ray_dirs[:,1]) < 1e-8, 1e-8, ray_dirs[:,1])[None]
                
                t1 = (b[:,0:1] - self.x) / dx
                t2 = (b[:,2:3] - self.x) / dx
                t3 = (b[:,1:2] - self.y) / dy
                t4 = (b[:,3:4] - self.y) / dy
                
                tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
                tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
                
                valid = tmax >= np.maximum(0, tmin)
                t = np.where(tmin > 0, tmin, tmax)
                t = np.where(valid & (t > 0) & (t < R), t, R)
                
                rays = np.minimum(rays, t.min(axis=0) / R)

        # Dynamic obstacles (Circles)
        for (ox, oy, r, _, _) in self.dynamic_obstacles:
            rel = np.array([ox - self.x, oy - self.y])
            b = -2.0 * (ray_dirs @ rel)
            c = float(rel @ rel) - r*r
            disc = b*b - 4*c
            valid = disc >= 0
            sq = np.sqrt(np.where(valid, disc, 0.0))
            t1 = (-b - sq) / 2.0
            t2 = (-b + sq) / 2.0
            t_min = np.where((t1 > 0) & (t1 < t2), t1, t2)
            t_min = np.where(valid & (t_min > 0) & (t_min < self.max_range), t_min, self.max_range)
            rays = np.minimum(rays, t_min / self.max_range)

        # Outer Walls
        dx_arr = ray_dirs[:, 0]; dy_arr = ray_dirs[:, 1]
        t_x = np.where(dx_arr > 1e-6, (self.world_size - self.x) / (dx_arr + 1e-8),
              np.where(dx_arr < -1e-6, -self.x / (dx_arr - 1e-8), self.max_range))
        t_y = np.where(dy_arr > 1e-6, (self.world_size - self.y) / (dy_arr + 1e-8),
              np.where(dy_arr < -1e-6, -self.y / (dy_arr - 1e-8), self.max_range))
        t_wall = np.minimum(t_x, t_y)
        t_wall = np.clip(t_wall, 0.0, self.max_range)
        rays = np.minimum(rays, t_wall / self.max_range)

        return rays

    def _get_obs(self):
        lidar = self.current_lidar
        agent_xy = np.array([self.x, self.y], dtype=np.float32)
        closest_idx = self._find_closest_path_idx(agent_xy)
        current_arc = float(self.path_cumulative_arc[closest_idx])
        waypoint, tangent = self._extract_lookahead_at(agent_xy, closest_idx, 3.0)

        # CHANGE /3.0 to /10.0 so it can see up to 15 meters away
        la_dx = float(np.clip((waypoint[0] - self.x) / 15.0, -1.0, 1.0))
        la_dy = float(np.clip((waypoint[1] - self.y) / 15.0, -1.0, 1.0))
        
        norm_vx = float(np.clip(self.vx / 10.0, -1.0, 1.0))
        norm_vy = float(np.clip(self.vy / 10.0, -1.0, 1.0))
        # <--- CHANGED: Use the stored last_accel
        norm_accel = float(np.clip(self.last_accel / 1.0, 0.0, 1.0))

        cell = 0.5
        gi = int(np.clip(self.y / cell, 0, self.dist_to_path_grid.shape[0] - 1))
        gj = int(np.clip(self.x / cell, 0, self.dist_to_path_grid.shape[1] - 1))
        cte = float(self.dist_to_path_grid[gi, gj])
        norm_cte = float(np.clip(cte / 10.0, 0.0, 1.0)) 

        norm_arc = float(np.clip(current_arc / self.total_arc_length, 0.0, 1.0))
        self.current_arc_length = current_arc

        speed = np.hypot(self.vx, self.vy)
        if speed > 0.5:
            vel_angle = np.arctan2(self.vy, self.vx)
            tan_angle = np.arctan2(tangent[1], tangent[0])
            diff = np.arctan2(np.sin(vel_angle - tan_angle), np.cos(vel_angle - tan_angle))
            cos_align = float(np.cos(diff))
        else:
            cos_align = 0.0

        min_lidar = float(np.min(lidar)) * self.max_range
        norm_min_lidar = float(np.clip(min_lidar / self.max_range, 0.0, 1.0))

        norm_theta = float(np.clip(self.theta / np.pi, -1.0, 1.0))
        norm_vtheta = float(np.clip(self.vtheta / 3.0, -1.0, 1.0))

        # <--- ADD THIS
        prev_action_norm = np.clip(self.prev_action / 10.0, -1.0, 1.0)

        # <--- CHANGED: Append the 2 new features
        state = np.array([
            la_dx, la_dy, tangent[0], tangent[1],
            norm_vx, norm_vy, norm_accel, norm_cte, norm_arc, cos_align, norm_min_lidar,
            norm_theta, norm_vtheta,
            prev_action_norm[0], prev_action_norm[1]  # <--- NEW
        ], dtype=np.float32)

        return np.concatenate([lidar, state]).astype(np.float32)

    def _check_collision(self, x, y):
        if x < 0 or x >= self.world_size or y < 0 or y >= self.world_size:
            return True
        for aabb in self.static_obstacles:
            if point_aabb_distance(x, y, aabb) < self.drone_radius:
                return True
        for (ox, oy, r, _, _) in self.dynamic_obstacles:
            if np.hypot(x - ox, y - oy) < r + self.drone_radius:
                return True
        return False

    def _update_dynamic_obstacles(self):
        new_obs = []
        for (ox, oy, r, vx, vy) in self.dynamic_obstacles:
            ox += vx * self.dt; oy += vy * self.dt
            if ox - r < 0: ox = r; vx = -vx
            elif ox + r > self.world_size: ox = self.world_size - r; vx = -vx
            if oy - r < 0: oy = r; vy = -vy
            elif oy + r > self.world_size: oy = self.world_size - r; vy = -vy
            new_obs.append((ox, oy, r, vx, vy))
        self.dynamic_obstacles = new_obs

    def step(self, action):
        self.step_counter += 1
        self._episode_length += 1

        self.target_vx = float(np.clip(action[0], -10.0, 10.0))
        self.target_vy = float(np.clip(action[1], -10.0, 10.0))

        obs_exp = np.array([
            self.vx / 10.0, self.vy / 10.0, self.theta / np.pi, self.vtheta / 3.0,
            self.target_vx / 10.0, self.target_vy / 10.0,
        ], dtype=np.float32)
        thrust, _ = self.velocity_expert.predict(obs_exp, deterministic=True)
        L, R = thrust
        self.last_thrust = (L, R)

        f_l, f_r = L * self.max_thrust_each, R * self.max_thrust_each
        total_t = f_l + f_r
        ax = (total_t / self.mass) * np.sin(self.theta)
        ay = (total_t / self.mass) * np.cos(self.theta) - self.g
        alpha = ((f_r - f_l) * self.d) / self.I

        self.vx += ax * self.dt; self.vy += ay * self.dt
        self.vtheta += alpha * self.dt; self.theta += self.vtheta * self.dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        self.x += self.vx * self.dt; self.y += self.vy * self.dt
        self.vx = np.clip(self.vx, -15.0, 15.0); self.vy = np.clip(self.vy, -15.0, 15.0)
        self.last_accel = np.hypot(self.vx - self.prev_vx, self.vy - self.prev_vy) 

        self._update_dynamic_obstacles()
        self.current_lidar = self._compute_lidar()

        agent_xy = np.array([self.x, self.y], dtype=np.float32)
        closest_idx = self._find_closest_path_idx(agent_xy)
        new_arc = float(self.path_cumulative_arc[closest_idx])

        collided = self._check_collision(self.x, self.y)
        goal_reached = np.hypot(self.x - self.goal_x, self.y - self.goal_y) < self.goal_radius
        truncated = self.step_counter >= self.max_steps
        terminated = collided or goal_reached

                # --- Initialize reward breakdown for logging ---
        breakdown = {"terminal": 0.0, "pbrs": 0.0, "action_pen": 0.0, "tilt_pen": 0.0, 
                     "lidar_pen": 0.0, "accel_pen": 0.0}
        self.last_min_lidar = float(np.min(self.current_lidar)) * self.max_range
        
# =========================== Reward Shaping ======================================
        # --- 1. Position-based PBRS (Φ(x,y) — computed everywhere, incl. terminal) ---
        # Keep these three fields only for logging
        cell = 0.5
        gi = int(np.clip(self.y / cell, 0, self.dist_to_path_grid.shape[0] - 1))
        gj = int(np.clip(self.x / cell, 0, self.dist_to_path_grid.shape[1] - 1))
        self.last_cte = float(self.dist_to_path_grid[gi, gj])

        speed = np.hypot(self.vx, self.vy)
        steer_diff = 0.0
        if speed > 0.05:
            vel_angle = np.arctan2(self.vy, self.vx)
            waypoint, _ = self._extract_lookahead_at(agent_xy, closest_idx, 15.0)
            angle_to_path = np.arctan2(waypoint[1] - self.y, waypoint[0] - self.x)
            steer_diff = np.arctan2(np.sin(vel_angle - angle_to_path), np.cos(vel_angle - angle_to_path))

        self.last_arc = new_arc
        self.last_lookahead_angle_diff = float(steer_diff) if speed > 0.05 else 0.0

        # Φ at actual new position — no special-casing for terminal states
        phi_next = self._compute_potential(self.x, self.y)

        shaping_reward = phi_next - self.prev_phi
        self.prev_phi = phi_next

        reward = shaping_reward
        breakdown["pbrs"] = shaping_reward

        # --- 3. Smoothness penalty (piecewise deadzone + linear ramp + clamp) ---
        if not terminated and not truncated and self.step_counter > 1:
            da = action - self.prev_action
            da_mag = float(np.hypot(da[0], da[1]))
            if da_mag <= self.smooth_deadzone:
                r_smooth = 0.0
            elif da_mag <= self.smooth_max_delta:
                r_smooth = -self.smooth_max_pen * (da_mag - self.smooth_deadzone) / (self.smooth_max_delta - self.smooth_deadzone)
            else:
                r_smooth = -self.smooth_max_pen
            reward += r_smooth
            breakdown["action_pen"] = r_smooth

        # --- 2. Add Terminal Penalties/Bonuses ---
        if collided:
            term_reward = -70.0
            reward += term_reward
            breakdown["terminal"] = term_reward
        elif goal_reached:
            term_reward = 100.0
            reward += term_reward
            breakdown["terminal"] = term_reward
        elif truncated:
            term_reward = 0.0
            reward += term_reward
            breakdown["terminal"] = term_reward

        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self._episode_reward += reward
# ==========================================================================
        info = {'breakdown': breakdown}
        if terminated or truncated:
            info['episode_reward'] = self._episode_reward
            info['episode_length'] = self._episode_length
            info['is_success'] = bool(goal_reached)
            info['collision'] = bool(collided)
        obs = self._get_obs()
        self.prev_vx = self.vx
        self.prev_vy = self.vy
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="cv2", timesteps=0, total_reward=0):
        if self.render_mode is None or mode != "cv2": return
        W, H = 800, 800
        canvas = np.zeros((H, W, 3), dtype=np.uint8) + 30
        scale = W / self.world_size
        def w2p(x, y): return int(x * scale), int(H - y * scale)

        # Static Obstacles (Grids)
        for (x0, y0, x1, y1) in self.static_obstacles:
            cv2.rectangle(canvas, w2p(x0, y1), w2p(x1, y0), (90, 90, 90), -1)

        # Dynamic obstacles (circles)
        for (ox, oy, r, vx, vy) in self.dynamic_obstacles:
            cv2.circle(canvas, w2p(ox, oy), max(1, int(r * scale)), (50, 50, 220), -1)
            p1 = w2p(ox, oy)
            p2 = w2p(ox + vx * 1.5, oy + vy * 1.5)
            cv2.arrowedLine(canvas, p1, p2, (255, 200, 200), 1, tipLength=0.3)

        # Reference path (green)
        # if self.path_xy is not None and len(self.path_xy) >= 2:
        #     for i in range(len(self.path_xy) - 1):
        #         cv2.line(canvas,
        #                  w2p(self.path_xy[i][0], self.path_xy[i][1]),
        #                  w2p(self.path_xy[i+1][0], self.path_xy[i+1][1]),
        #                  (0, 255, 0), 2)

        # =========================================================
        # FIX: Calculate drone pixels BEFORE using them below
        # =========================================================
        drone_px, drone_py = w2p(self.x, self.y)

        # # --- Visualize the Lookahead Waypoint and CTE ---
        # if self.path_xy is not None:
        #     agent_xy = np.array([self.x, self.y], dtype=np.float32)
            
        #     # 1. Find closest point and lookahead point
        #     closest_idx = self._find_closest_path_idx(agent_xy)
        #     waypoint, _ = self._extract_lookahead_at(agent_xy, closest_idx, 3.0)
            
        #     # Convert to pixels
        #     closest_pt = self.path_xy[closest_idx]
        #     closest_px = w2p(closest_pt[0], closest_pt[1])
        #     waypoint_px = w2p(waypoint[0], waypoint[1])
            
        #     # 2. Draw a line from drone to the Lookahead Waypoint (Cyan)
        #     cv2.line(canvas, (drone_px, drone_py), waypoint_px, (255, 255, 0), 1, cv2.LINE_AA)
            
        #     # 3. Draw the Lookahead Waypoint (Magenta)
        #     cv2.circle(canvas, waypoint_px, 6, (255, 0, 255), -1)
        #     cv2.circle(canvas, waypoint_px, 6, (255, 255, 255), 1)
            
        #     # 4. Draw the Closest Point on the path (Yellow)
        #     cv2.circle(canvas, closest_px, 5, (0, 255, 255), -1)
            
        #     # 5. Draw the CTE line (Red)
        #     cv2.line(canvas, (drone_px, drone_py), closest_px, (0, 0, 255), 1)
        # -----------------------------------------------------

        # Goal
        gp = w2p(self.goal_x, self.goal_y)
        cv2.circle(canvas, gp, int(self.goal_radius * scale), (0, 220, 0), -1)

        # LiDAR rays
        # NOTE: I removed the "drone_px, drone_py = w2p(self.x, self.y)" line from here 
        # because we already calculated it at the top.
        lidar = self._compute_lidar()
        for i in range(0, self.n_rays, 3):
            a = self.theta + self.ray_angles[i]
            d = float(lidar[i]) * self.max_range
            end = w2p(self.x + d * np.cos(a), self.y + d * np.sin(a))
            norm = min(d / self.max_range, 1.0)
            color = (int(255 * (1 - norm)), int(255 * norm), 0)
            cv2.line(canvas, (drone_px, drone_py), end, color, 1, cv2.LINE_AA)

        # Drone Body & Thrusters
        half_len = 0.8
        rear_world  = (self.x - half_len * np.cos(self.theta), self.y + half_len * np.sin(self.theta))
        front_world = (self.x + half_len * np.cos(self.theta), self.y - half_len * np.sin(self.theta))
        rear_px  = w2p(*rear_world)
        front_px = w2p(*front_world)
        cv2.line(canvas, rear_px, front_px, (255, 255, 0), 4)
        cv2.circle(canvas, (drone_px, drone_py), 3, (255, 255, 0), -1)

        l_ratio = float(self.last_thrust[0])
        r_ratio = float(self.last_thrust[1])
        exhaust_x = -np.sin(self.theta); exhaust_y = -np.cos(self.theta)

        if l_ratio > 0.05:
            flen = 0.2 + 1.0 * l_ratio
            cv2.line(canvas, rear_px, w2p(rear_world[0] + flen * exhaust_x, rear_world[1] + flen * exhaust_y), (0, 0, 255), 3)
        if r_ratio > 0.05:
            flen = 0.2 + 1.0 * r_ratio
            cv2.line(canvas, front_px, w2p(front_world[0] + flen * exhaust_x, front_world[1] + flen * exhaust_y), (0, 0, 255), 3)

        # Velocity arrows
        if np.hypot(self.target_vx, self.target_vy) > 0.1:
            cv2.arrowedLine(canvas, (drone_px, drone_py), w2p(self.x + self.target_vx * 0.4, self.y + self.target_vy * 0.4), (0, 255, 0), 2, tipLength=0.35)
        if np.hypot(self.vx, self.vy) > 0.1:
            cv2.arrowedLine(canvas, (drone_px, drone_py), w2p(self.x + self.vx * 0.4, self.y + self.vy * 0.4), (255, 100, 0), 2, tipLength=0.35)

        # HUD
        font = cv2.FONT_HERSHEY_SIMPLEX
        dist = np.hypot(self.x - self.goal_x, self.y - self.goal_y)
        cv2.putText(canvas, f"Steps: {timesteps}", (10, 30), font, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"Reward: {total_reward:.1f}", (10, 60), font, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"Dist to goal: {dist:.2f} m", (10, 90), font, 0.6, (0, 255, 0), 1)
        cv2.putText(canvas, f"Current Vx: {self.vx:6.2f} m/s", (10, 120), font, 0.5, (255, 100, 0), 1)
        cv2.putText(canvas, f"Current Vy: {self.vy:6.2f} m/s", (10, 145), font, 0.5, (255, 100, 0), 1)
        cv2.putText(canvas, "PHASE 1: Exploration Only", (10, H - 20), font, 0.6, (100, 200, 255), 1)

        cv2.imshow("Path Follower (Procedural World)", canvas)
        cv2.waitKey(1)

# =====================================================================
# 3. FEATURE EXTRACTOR
# =====================================================================
class PathFollowerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_state = 15
        self.pad = 2
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=0), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(12), nn.Flatten(),
            nn.Linear(384, 64), nn.ReLU(),
        )
        self.state_net = nn.Sequential(nn.Linear(n_state, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.combined = nn.Sequential(nn.Linear(64 + 64, features_dim), nn.ReLU())

    def _circular_pad(self, x):
        p = self.pad
        return torch.cat([x[:, :, -p:], x, x[:, :, :p]], dim=2)

    def forward(self, obs):
        lidar = self._circular_pad(obs[:, :180].unsqueeze(1))
        state = obs[:, 180:]
        return self.combined(torch.cat([self.lidar_conv(lidar), self.state_net(state)], dim=1))

# =====================================================================
# 4. CALLBACK
# =====================================================================
class PlottingCallback(BaseCallback):
    def __init__(self, eval_env, save_dir, model_name, plot_interval=1, render_episode_freq=100, save_interval=1):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.model_name = model_name
        self.plot_interval = plot_interval
        self.render_episode_freq = render_episode_freq
        self.save_interval = save_interval
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_success = []
        self.ep_counter = 0
        os.makedirs(self.save_dir, exist_ok=True)
        self.history_file = os.path.join(self.save_dir, "path_history.npz")
        self._load_history()
        plt.ion()
        self.fig, (self.ax_r, self.ax_l) = plt.subplots(2, 1, figsize=(8, 6))

    def _load_history(self):
        if os.path.exists(self.history_file):
            d = np.load(self.history_file, allow_pickle=True)
            self.ep_rewards = list(d['ep_rewards'])
            self.ep_lengths = list(d['ep_lengths'])
            self.ep_success = list(d['ep_success'])
            self.ep_counter = int(d['ep_counter'])
            print(f"[CALLBACK] Loaded {self.ep_counter} past episodes.")

    def _save_history(self):
        np.savez(self.history_file,
                 ep_rewards=np.array(self.ep_rewards),
                 ep_lengths=np.array(self.ep_lengths),
                 ep_success=np.array(self.ep_success),
                 ep_counter=self.ep_counter)

    def _on_step(self):
        # Check for either dones (older SB3) or terminated/truncated (newer SB3)
        dones = self.locals.get("dones")
        terminated = self.locals.get("terminated")
        truncated = self.locals.get("truncated")
        
        is_done = False
        if dones is not None:
            is_done = dones[0]
        elif terminated is not None:
            is_done = terminated[0] or truncated[0]

        if is_done:
            infos = self.locals.get("infos")
            # SB3's Monitor wrapper puts the reward inside infos[0]['episode']
            if infos and "episode" in infos[0]:
                info = infos[0]
                self.ep_counter += 1
                self.ep_rewards.append(info['episode']['r'])
                self.ep_lengths.append(info['episode']['l'])
                self.ep_success.append(int(info.get('is_success', 0)))
                self._save_history()
                
                if self.ep_counter % self.save_interval == 0:
                    # Overwrite the same files to save disk space
                    checkpoint = os.path.join(self.save_dir, self.model_name)
                    self.model.save(checkpoint)
                    self.model.save_replay_buffer(checkpoint + "_replay_buffer.pkl")
                    
                if self.ep_counter % self.plot_interval == 0:
                    self._update_plot()
                if self.ep_counter % self.render_episode_freq == 0:
                    self._render_eval()
        return True

    def _update_plot(self):
        # Guard against empty lists (prevents the IndexError crash on Ctrl+C)
        if len(self.ep_rewards) == 0:
            return

        eps = np.arange(1, len(self.ep_rewards) + 1)
        self.ax_r.clear()
        self.ax_r.plot(eps, self.ep_rewards, 'b-', alpha=0.4, label='Reward')
        if len(self.ep_rewards) >= 20:
            ma = np.convolve(self.ep_rewards, np.ones(20)/20, mode='valid')
            self.ax_r.plot(np.arange(20, len(self.ep_rewards)+1), ma, 'r-', label='MA20')
        if len(self.ep_success) >= 20:
            sr = np.convolve(self.ep_success, np.ones(20)/20, mode='valid')
            ax2 = self.ax_r.twinx()
            ax2.plot(np.arange(20, len(self.ep_success)+1), sr, 'g--', alpha=0.8)
            ax2.set_ylabel('Success', color='g'); ax2.set_ylim(0, 1)
        self.ax_r.set_ylabel('Reward'); self.ax_r.set_xlabel('Episode')
        self.ax_r.legend(loc='upper left'); self.ax_r.grid(True)

        self.ax_l.clear()
        self.ax_l.plot(eps, self.ep_lengths, 'g-', alpha=0.4)
        if len(self.ep_lengths) >= 20:
            ma = np.convolve(self.ep_lengths, np.ones(20)/20, mode='valid')
            self.ax_l.plot(np.arange(20, len(self.ep_lengths)+1), ma, 'r-')
        self.ax_l.set_ylabel('Steps'); self.ax_l.set_xlabel('Episode'); self.ax_l.grid(True)

        sr_last = np.mean(self.ep_success[-20:]) if self.ep_success else 0.0
        self.fig.suptitle(f"Ep {self.ep_counter} | R(last): {self.ep_rewards[-1]:.1f} | Success(20): {sr_last:.2f}", fontsize=10, y=0.96)
        self.fig.tight_layout(rect=[0, 0, 1, 0.93])
        self.fig.canvas.draw()
        self.fig.savefig(os.path.join(self.save_dir, 'path_plot.png'), dpi=100)
        plt.pause(0.01)


    def _render_eval(self):
        obs, _ = self.eval_env.reset()
        terminated = truncated = False
        steps, total_r = 0, 0.0
        while not (terminated or truncated):
            a, _ = self.model.predict(obs, deterministic=False)
            obs, r, terminated, truncated, _ = self.eval_env.step(a)
            total_r += r
            self.eval_env.render(mode="cv2", timesteps=steps, total_reward=total_r)
            steps += 1
            if cv2.waitKey(1) & 0xFF == 27: break
        cv2.waitKey(500); cv2.destroyAllWindows()
        print(f"[EVAL] steps={steps}, reward={total_r:.1f}")

# =====================================================================
# 5. TRAIN & TEST
# =====================================================================
def train():
    if not os.path.exists(VELOCITY_EXPERT_PATH + ".zip"):
        print(f"ERROR: velocity expert missing."); sys.exit(1)
    velocity_expert = PPO.load(VELOCITY_EXPERT_PATH, device="cpu")
    print("Velocity expert loaded.")

    train_env = ActionRepeat(PathFollowingEnv(velocity_expert, render_mode=None), k=5)
    eval_env = ActionRepeat(PathFollowingEnv(velocity_expert, render_mode="cv2"), k=5)

    policy_kwargs = dict(
        features_extractor_class=PathFollowerExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=nn.ReLU,
    )

    model_path = None
    buffer_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_replay_buffer.pkl")
    if LOAD_EXISTING_MODEL:
        potential_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.zip")
        if os.path.exists(potential_path):
            model_path = potential_path
            print(f"[LOAD] Resuming from {model_path}")

    if model_path is not None:
        model = SAC.load(model_path, env=train_env, device="cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(buffer_path):
            try:
                model.load_replay_buffer(buffer_path)
                print(f"[LOAD] Replay buffer loaded from {buffer_path}")
                model.learning_starts = 0   # ← ADD THIS — force immediate policy use
            except Exception as e:
                print(f"[LOAD] Could not load replay buffer: {e}")
                
    else:
        model = SAC(
            "MlpPolicy", train_env, 
            learning_rate=1e-4, 
            buffer_size=500_000,
            # learning_starts=20_000, 
            learning_starts=0, 
            batch_size=512, 
            tau=0.005, 
            gamma=0.998,
            train_freq=1, 
            gradient_steps=1, 
            ent_coef=0.1, 
            target_entropy="auto",
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    callback = PlottingCallback(
        eval_env=eval_env, save_dir=SAVE_DIR, model_name=MODEL_NAME,
        plot_interval=PLOT_INTERVAL, render_episode_freq=RENDER_EPISODE_FREQ, save_interval=SAVE_EPISODE_INTERVAL,
    )

    try:
        model.learn(total_timesteps=500_000, callback=callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        model.save(os.path.join(SAVE_DIR, f"{MODEL_NAME}_interrupt"))
        model.save_replay_buffer(os.path.join(SAVE_DIR, f"{MODEL_NAME}_replay_buffer.pkl"))
        callback._update_plot()
        sys.exit(0)
    finally:
        model.save(os.path.join(SAVE_DIR, f"{MODEL_NAME}.zip")) # Overwrites the checkpoint with the final state
        model.save_replay_buffer(os.path.join(SAVE_DIR, f"{MODEL_NAME}_replay_buffer.pkl"))
        callback._update_plot()
        print("Final model saved.")

def test():
    if not os.path.exists(VELOCITY_EXPERT_PATH + ".zip"):
        print(f"ERROR: velocity expert missing."); sys.exit(1)
    velocity_expert = PPO.load(VELOCITY_EXPERT_PATH, device="cpu")

    existing = [f for f in os.listdir(SAVE_DIR) if f.startswith(MODEL_NAME) and f.endswith('.zip')]
    if not existing: return
    finals = [f for f in existing if 'final' in f]
    mp = os.path.join(SAVE_DIR, finals[0]) if finals else os.path.join(SAVE_DIR, max(existing))

    env = ActionRepeat(PathFollowingEnv(velocity_expert, render_mode="cv2"), k=5)
    # model = SAC.load(mp, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    # ✅ REPLACE WITH:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomSACPolicy.load(
        mp,
        obs_dim=195, act_dim=2, features_dim=128,
        action_low=-10.0, action_high=10.0,
        device=device,
    )

    successes = 0
    for ep in range(TEST_EPISODES):
        obs, _ = env.reset()
        terminated = truncated = False
        steps, total_r = 0, 0.0
        info = {}
        
        episode_log = []  # <--- Initialize list for logging
        
        while not (terminated or truncated):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(a)
            total_r += r
            
            # --- Log every step ---
            # bd = info.get('breakdown', {})
            # episode_log.append({
            #     "step": steps,
            #     "total_reward": r,
            #     "action_vx": a[0],
            #     "action_vy": a[1],
            #     "actual_vx": env.unwrapped.vx,                     # was env.vx
            #     "actual_vy": env.unwrapped.vy,                     # was env.vy
            #     "speed": np.hypot(env.unwrapped.vx, env.unwrapped.vy),
            #     "theta_deg": np.rad2deg(env.unwrapped.theta),
            #     "arc_progress": env.unwrapped.last_arc / (env.unwrapped.total_arc_length + 1e-8),
            #     "actual_arc": env.unwrapped.last_arc,
            #     "total_arc": env.unwrapped.total_arc_length,
            #     "lookahead_angle_deg": np.rad2deg(env.unwrapped.last_lookahead_angle_diff),
            #     "cte": env.unwrapped.last_cte,
            #     "min_lidar": env.unwrapped.last_min_lidar,
            #     "r_terminal": bd.get('terminal', 0.0),
            #     "r_pbrs": bd.get('pbrs', 0.0),
            #     "r_action_pen": bd.get('action_pen', 0.0),
            #     "r_tilt_pen": bd.get('tilt_pen', 0.0),
            #     "r_lidar_pen": bd.get('lidar_pen', 0.0),
            #     "r_accel_pen": bd.get('accel_pen', 0.0),
            # })
            # ----------------------

            env.render(mode="cv2", timesteps=steps, total_reward=total_r)
            steps += 1
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows(); return
                
        cv2.waitKey(400)
        ok = bool(info.get('is_success', False))
        successes += int(ok)
        
        # --- Save the logged data to CSV ---
        # csv_path = os.path.join(SAVE_DIR, f"test_ep_{ep+1}_log.csv")
        # with open(csv_path, mode='w', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=episode_log[0].keys())
        #     writer.writeheader()
        #     writer.writerows(episode_log)
        # -----------------------------------
        
        print(f"Ep {ep+1}: reward={total_r:.1f}, steps={steps}, success={ok} | Saved: test_ep_{ep+1}_log.csv")
        cv2.destroyAllWindows()
        
    print(f"\nSuccess rate: {successes}/{TEST_EPISODES}")

if __name__ == "__main__":
    # torch.set_num_threads(1)
    if MODE.lower() == "train": train()
    elif MODE.lower() == "test": test()