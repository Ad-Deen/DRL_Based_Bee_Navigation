"""
velocity_agent_final.py

Single-file solution: environment + training/testing loop.
Set MODE = "train" to train a new model, or "test" to evaluate a saved model.

Wind: Natural-like distribution — per-episode personality + Markov regime
transitions + exponential durations. The expert never sees wind directly;
it must infer disturbance from velocity-error feedback.
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# =====================================================================
# CONFIGURATION
# =====================================================================
MODE = "train"                       # "train" or "test"
TEST_EPISODES = 5

SAVE_DIR = "/home/deen/ppo_test/drone_flight_expert/"
MODEL_NAME = "velocity_expert_wind"
PLOT_INTERVAL = 10
RENDER_EPISODE_FREQ = 100
SAVE_EPISODE_INTERVAL = 20
LOAD_EXISTING_MODEL = True
# =====================================================================


# =====================================================================
# 1. ENVIRONMENT (with natural wind + position tracking + rendering)
# =====================================================================

class VelocityTrackingEnv(gym.Env):
    """
    2D drone velocity-tracking with position for visualization only.
    Observation: [vx, vy, theta, vtheta, target_vx, target_vy]  (6-dim, wind BLIND)
    Action:      [L_thrust, R_thrust]                            (2-dim, [0,1])
    World:       40x40 with wrap-around (toroidal).
    Wind:        Natural-like — per-episode personality, Markov regime
                 transitions, exponential durations. Never in obs.
    """
    def __init__(self):
        super().__init__()
        # ---- Physics ----
        self.mass = 3.0
        self.max_thrust_each = 5.0 * 9.8   # 49 N
        self.g = 9.8
        self.dt = 0.02
        self.d = 0.4
        self.I = 0.3

        # ---- Wind disturbance: natural distribution ----
        self.use_wind = True
        self.wind_max_force = 15.0           # N per axis (peak gust)

        # Per-episode personality (re-sampled in reset())
        self.wind_turn_on_prob      = 0.35   # placeholder, set in reset
        self.wind_stay_on_prob      = 0.25   # placeholder, set in reset
        self.wind_on_duration_mean  = 100    # steps, set in reset
        self.wind_off_duration_mean = 200    # steps, set in reset

        # Runtime state
        self.wind_active = False
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.wind_change_counter = 0
        self.wind_change_interval = 100

        # ---- Target update parameters ----
        self.target_change_interval_min = 30
        self.target_change_interval_max = 80
        self.target_change_counter = 0
        self.target_change_interval = np.random.randint(
            self.target_change_interval_min, self.target_change_interval_max
        )

        # ---- Smooth transition parameters ----
        self.transition_steps = 20
        self.transition_counter = 0
        self.target_vx_start = 0.0
        self.target_vy_start = 0.0
        self.target_vx_end = 0.0
        self.target_vy_end = 0.0
        self.use_smooth_transition = True
        self.smooth_transition_prob = 0.5

        # ---- State limits ----
        self.max_v = 10.0
        self.max_theta = np.pi
        self.max_vtheta = 3.0

        # ---- Misc ----
        self.step_counter = 0
        self.max_episode_steps = 600
        self.world_size = 40.0
        self.x = 0.0
        self.y = 0.0

        # ---- Spaces ----
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ---- Internal state ----
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0
        self.vtheta = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.prev_action = np.array([0.5, 0.5])

        # ---- Episode logging ----
        self._episode_reward = 0.0
        self._episode_length = 0

    # ------------------------------------------------------------------
    # Wind: natural distribution
    # ------------------------------------------------------------------
    def _start_wind_period(self, active=None):
        """
        Enter a new wind regime.
        If `active` is None → decide via Markov chain (regime transition).
        If `active` is True/False → force that regime.
        """
        if not self.use_wind:
            self.wind_active = False
            self.wind_x = 0.0
            self.wind_y = 0.0
            self.wind_change_counter = 0
            self.wind_change_interval = 100
            return

        if active is None:
            if self.wind_active:
                # Currently ON → stay ON w.p. stay_on_prob, else turn OFF
                active = np.random.random() < self.wind_stay_on_prob
            else:
                # Currently OFF → turn ON w.p. turn_on_prob, else stay OFF
                active = np.random.random() < self.wind_turn_on_prob

        self.wind_active = active
        self.wind_change_counter = 0

        if active:
            # Sample a fresh gust magnitude
            self.wind_x = np.random.uniform(-self.wind_max_force, self.wind_max_force)
            self.wind_y = np.random.uniform(-self.wind_max_force, self.wind_max_force)
            # Exponential duration: mostly short, occasionally long
            self.wind_change_interval = max(
                30, int(np.random.exponential(self.wind_on_duration_mean))
            )
        else:
            self.wind_x = 0.0
            self.wind_y = 0.0
            self.wind_change_interval = max(
                50, int(np.random.exponential(self.wind_off_duration_mean))
            )

    def _update_wind(self):
        """Count steps; when period ends, transition via Markov chain."""
        if not self.use_wind:
            return
        self.wind_change_counter += 1
        if self.wind_change_counter >= self.wind_change_interval:
            self._start_wind_period(active=None)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        # Random initial state
        self.vx = np.random.uniform(-0.5, 0.5)
        self.vy = np.random.uniform(-0.5, 0.5)
        self.theta = np.random.uniform(-0.1, 0.1)
        self.vtheta = np.random.uniform(-0.1, 0.1)
        self.step_counter = 0

        # Random position for visualisation
        self.x = np.random.uniform(5.0, 35.0)
        self.y = np.random.uniform(5.0, 35.0)

        self._set_new_target()
        self.prev_action = np.array([0.5, 0.5])

        self._episode_reward = 0.0
        self._episode_length = 0

        self.target_change_counter = 0
        self.target_change_interval = np.random.randint(
            self.target_change_interval_min, self.target_change_interval_max
        )

        # ---- Per-episode wind personality ----
        # Each episode gets its own statistics → calm days vs stormy days
        self.wind_turn_on_prob      = np.random.uniform(0.15, 0.60)
        self.wind_stay_on_prob      = np.random.uniform(0.05, 0.50)
        self.wind_on_duration_mean  = np.random.uniform(50, 150)
        self.wind_off_duration_mean = np.random.uniform(100, 400)

        # ---- Initial regime: Markov from OFF state ----
        self.wind_active = False
        self._start_wind_period(active=None)

        return self._get_obs(), {}

    def _set_new_target(self):
        """Randomly choose a new target, with smooth transition if enabled."""
        new_vx = np.random.uniform(-7.0, 7.0)
        new_vy = np.random.uniform(-7.0, 7.0)

        # 10% chance to keep the same target
        if np.random.random() < 0.1:
            new_vx = self.target_vx
            new_vy = self.target_vy

        self.smooth_transition_prob = np.random.uniform(0.2, 0.8)
        self.use_smooth_transition = np.random.random() < self.smooth_transition_prob

        if self.use_smooth_transition:
            self.target_vx_start = self.target_vx
            self.target_vy_start = self.target_vy
            self.target_vx_end = new_vx
            self.target_vy_end = new_vy
            self.transition_counter = 0
        else:
            self.target_vx = new_vx
            self.target_vy = new_vy

    def _update_target(self):
        if self.use_smooth_transition:
            if self.transition_counter < self.transition_steps:
                alpha = self.transition_counter / self.transition_steps
                self.target_vx = self.target_vx_start + alpha * (self.target_vx_end - self.target_vx_start)
                self.target_vy = self.target_vy_start + alpha * (self.target_vy_end - self.target_vy_start)
                self.transition_counter += 1
            else:
                self.target_vx = self.target_vx_end
                self.target_vy = self.target_vy_end
                self.target_change_counter += 1
                if self.target_change_counter >= self.target_change_interval:
                    self._set_new_target()
                    self.target_change_counter = 0
                    self.target_change_interval = np.random.randint(
                        self.target_change_interval_min, self.target_change_interval_max
                    )
        else:
            self.target_change_counter += 1
            if self.target_change_counter >= self.target_change_interval:
                self._set_new_target()
                self.target_change_counter = 0
                self.target_change_interval = np.random.randint(
                    self.target_change_interval_min, self.target_change_interval_max
                )

    def _get_obs(self):
        norm_vx = np.clip(self.vx / self.max_v, -1.0, 1.0)
        norm_vy = np.clip(self.vy / self.max_v, -1.0, 1.0)
        norm_theta = np.clip(self.theta / self.max_theta, -1.0, 1.0)
        norm_vtheta = np.clip(self.vtheta / self.max_vtheta, -1.0, 1.0)
        norm_tvx = np.clip(self.target_vx / self.max_v, -1.0, 1.0)
        norm_tvy = np.clip(self.target_vy / self.max_v, -1.0, 1.0)
        # NOTE: wind is NOT included — expert must infer via velocity error.
        return np.array([norm_vx, norm_vy, norm_theta, norm_vtheta,
                         norm_tvx, norm_tvy], dtype=np.float32)

    def render(self, mode="cv2", timesteps=0, total_reward=0):
        """Render the drone flying in the 40x40 world with wrap-around."""
        if mode != "cv2":
            print(f"pos: ({self.x:.1f}, {self.y:.1f}), "
                  f"v: ({self.vx:.2f}, {self.vy:.2f}), "
                  f"target: ({self.target_vx:.2f}, {self.target_vy:.2f}), "
                  f"wind: ({self.wind_x:.2f}, {self.wind_y:.2f}), "
                  f"active: {self.wind_active}")
            return

        width, height = 800, 800
        canvas = np.zeros((height, width, 3), dtype=np.uint8) + 30
        scale = width / self.world_size

        def world_to_pixel(x, y):
            px = int(x * scale)
            py = int(height - y * scale)
            return px, py

        # Border
        cv2.rectangle(canvas, (0, 0), (width, height), (100, 100, 100), 1)

        # ---- Target velocity arrow (green) ----
        tvx, tvy = self.target_vx, self.target_vy
        if np.hypot(tvx, tvy) > 0.1:
            start_x, start_y = world_to_pixel(self.x, self.y)
            end_x = int(start_x + tvx * 10)
            end_y = int(start_y - tvy * 10)
            cv2.arrowedLine(canvas, (start_x, start_y), (end_x, end_y),
                            (0, 255, 0), 2, tipLength=0.3)

        # ---- Current velocity arrow (orange) ----
        cvx, cvy = self.vx, self.vy
        if np.hypot(cvx, cvy) > 0.1:
            start_x, start_y = world_to_pixel(self.x, self.y)
            end_x = int(start_x + cvx * 10)
            end_y = int(start_y - cvy * 10)
            cv2.arrowedLine(canvas, (start_x, start_y), (end_x, end_y),
                            (255, 100, 0), 2, tipLength=0.3)

        # ---- Drone body ----
        dx, dy = world_to_pixel(self.x, self.y)
        half_len = 1.0
        nose_world = (self.x + half_len * np.cos(self.theta),
                      self.y - half_len * np.sin(self.theta))
        tail_world = (self.x - half_len * np.cos(self.theta),
                      self.y + half_len * np.sin(self.theta))
        nose_px = world_to_pixel(nose_world[0], nose_world[1])
        tail_px = world_to_pixel(tail_world[0], tail_world[1])
        cv2.line(canvas, tail_px, nose_px, (255, 255, 0), 4)

        # ---- Thrust flames ----
        f_left = self.prev_action[0] * self.max_thrust_each
        f_right = self.prev_action[1] * self.max_thrust_each

        left_motor_world = (self.x - half_len * np.cos(self.theta),
                            self.y + half_len * np.sin(self.theta))
        right_motor_world = (self.x + half_len * np.cos(self.theta),
                             self.y - half_len * np.sin(self.theta))

        left_px = world_to_pixel(left_motor_world[0], left_motor_world[1])
        right_px = world_to_pixel(right_motor_world[0], right_motor_world[1])

        flame_dir_x = -np.sin(self.theta)
        flame_dir_y = np.cos(self.theta)

        if f_left > 2.0:
            flame_len_px = 5 + (f_left / self.max_thrust_each) * 40
            flame_end_x = left_px[0] + flame_len_px * flame_dir_x
            flame_end_y = left_px[1] + flame_len_px * flame_dir_y
            cv2.line(canvas, left_px, (int(flame_end_x), int(flame_end_y)),
                     (0, 0, 255), 4)

        if f_right > 2.0:
            flame_len_px = 5 + (f_right / self.max_thrust_each) * 40
            flame_end_x = right_px[0] + flame_len_px * flame_dir_x
            flame_end_y = right_px[1] + flame_len_px * flame_dir_y
            cv2.line(canvas, right_px, (int(flame_end_x), int(flame_end_y)),
                     (0, 0, 255), 4)

        # ---- HUD ----
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"Total Reward: {total_reward:.1f}", (10, 30),
                    font, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"Vx: {self.vx:.2f} m/s  (Target: {self.target_vx:.2f})",
                    (10, 60), font, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"Vy: {self.vy:.2f} m/s  (Target: {self.target_vy:.2f})",
                    (10, 80), font, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"Tilt: {np.degrees(self.theta):.1f} deg",
                    (10, 110), font, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"Steps: {timesteps}", (10, 140), font, 0.5, (200, 200, 200), 1)

        # ---- Wind indicator: CALM vs GUSTY ----
        if not self.wind_active:
            cv2.putText(canvas, "Wind: CALM",
                        (10, 170), font, 0.5, (150, 150, 150), 1)
        else:
            wind_mag = abs(self.wind_x) + abs(self.wind_y)
            wind_color = (255, 0, 0) if wind_mag > 5.0 else (200, 100, 100)
            cv2.putText(canvas,
                        f"Wind: ({self.wind_x:+.1f}, {self.wind_y:+.1f}) N",
                        (10, 170), font, 0.5, wind_color, 1)

        # ---- Episode personality (small grey hint) ----
        cv2.putText(canvas,
                    f"regime: p_on={self.wind_turn_on_prob:.2f} "
                    f"p_stay={self.wind_stay_on_prob:.2f}",
                    (10, 195), font, 0.4, (120, 120, 120), 1)

        cv2.putText(canvas, "Green Arrow = Target Velocity", (10, height - 30),
                    font, 0.4, (0, 255, 0), 1)
        cv2.putText(canvas, "Orange Arrow = Current Velocity", (10, height - 10),
                    font, 0.4, (255, 100, 0), 1)

        cv2.imshow("Velocity Expert - Drone Navigation", canvas)
        cv2.waitKey(33)

    def step(self, action):
        self.step_counter += 1
        self._episode_length += 1
        action = np.clip(action, 0.0, 1.0)

        # ---- Physics ----
        f_left = action[0] * self.max_thrust_each
        f_right = action[1] * self.max_thrust_each
        total_thrust = f_left + f_right

        # Wind enters as external force; zero when calm, nonzero when gusty
        ax = (total_thrust / self.mass) * np.sin(self.theta) + self.wind_x / self.mass
        ay = (total_thrust / self.mass) * np.cos(self.theta) - self.g + self.wind_y / self.mass
        alpha = ((f_right - f_left) * self.d) / self.I

        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.vtheta += alpha * self.dt
        self.theta += self.vtheta * self.dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Update position (wrap-around)
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.x = self.x % self.world_size
        self.y = self.y % self.world_size

        # Clip velocities
        self.vx = np.clip(self.vx, -20.0, 20.0)
        self.vy = np.clip(self.vy, -20.0, 20.0)

        # Errors
        vx_err = np.clip(self.vx - self.target_vx, -10.0, 10.0)
        vy_err = np.clip(self.vy - self.target_vy, -10.0, 10.0)

        # Reward
        vel_error_penalty = -0.04 * (vx_err**2 + vy_err**2)
        tilt_penalty      = -0.0005 * self.theta**2
        vtheta_penalty    = -0.0005 * self.vtheta**2
        ax_penalty        = -0.0005 * ax**2
        ay_penalty        = -0.0005 * ay**2
        action_diff = action - self.prev_action
        action_penalty = -0.001 * (action_diff[0]**2 + action_diff[1]**2)

        reward = (vel_error_penalty + tilt_penalty + vtheta_penalty +
                  ax_penalty + ay_penalty + action_penalty)

        self.prev_action = action.copy()
        self._episode_reward += reward

        self._update_target()
        self._update_wind()

        truncated = False
        info = {}
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            info['episode_reward'] = self._episode_reward
            info['episode_length'] = self._episode_length

        return self._get_obs(), reward, False, truncated, info


# =====================================================================
# 2. TRAINING / TESTING LOGIC
# =====================================================================

class SimpleVisualCallback(BaseCallback):
    def __init__(self, eval_env, save_dir, model_name,
                 plot_interval=1, render_episode_freq=100, save_interval=1):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.model_name = model_name
        self.plot_interval = plot_interval
        self.render_episode_freq = render_episode_freq
        self.save_interval = save_interval

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_counter = 0

        self.history_file = os.path.join(self.save_dir, "reward_history.npz")
        self._load_history()

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[CALLBACK] Save directory: {os.path.abspath(self.save_dir)}")

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        plt.show(block=False)

    def _load_history(self):
        if os.path.exists(self.history_file):
            data = np.load(self.history_file, allow_pickle=True)
            self.ep_rewards = list(data['ep_rewards'])
            self.ep_lengths = list(data['ep_lengths'])
            self.ep_counter = int(data['ep_counter'])
            print(f"[CALLBACK] Loaded history: {self.ep_counter} episodes.")
        else:
            self.ep_rewards = []
            self.ep_lengths = []
            self.ep_counter = 0

    def _save_history(self):
        np.savez(self.history_file,
                ep_rewards=np.array(self.ep_rewards),
                ep_lengths=np.array(self.ep_lengths),
                ep_counter=self.ep_counter)

    def _on_step(self):
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            infos = self.locals.get("infos")
            if infos is not None and len(infos) > 0:
                info = infos[0]
                if 'episode_reward' in info and 'episode_length' in info:
                    reward = info['episode_reward']
                    length = info['episode_length']
                    self.ep_counter += 1
                    self.ep_rewards.append(reward)
                    self.ep_lengths.append(length)
                    self._save_history()

                    if self.ep_counter % self.save_interval == 0:
                        save_path = os.path.join(self.save_dir, f"{self.model_name}")
                        self.model.save(save_path)

                    if self.ep_counter % self.plot_interval == 0:
                        self._update_plot()

                    if self.ep_counter % self.render_episode_freq == 0:
                        self._render_eval_episode()
        return True

    def _update_plot(self):
        self.ax.clear()
        eps = np.arange(1, len(self.ep_rewards) + 1)
        self.ax.plot(eps, self.ep_rewards, 'b-', alpha=0.5, label='Reward')
        if len(self.ep_rewards) >= 10:
            window = 10
            ma = np.convolve(self.ep_rewards, np.ones(window)/window, mode='valid')
            self.ax.plot(np.arange(window, len(self.ep_rewards)+1), ma, 'r-', label='MA (10)')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw()
        plot_path = os.path.join(self.save_dir, 'velocity_expert_reward_plot.png')
        self.fig.savefig(plot_path, dpi=100)
        plt.pause(0.01)

    def _render_eval_episode(self):
        obs, _ = self.eval_env.reset()
        terminated, truncated = False, False
        steps = 0
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            total_reward += reward
            self.eval_env.render(mode="cv2", timesteps=steps, total_reward=total_reward)
            steps += 1

            if cv2.waitKey(1) & 0xFF == 27:
                print("   [SKIP] Evaluation interrupted.")
                break

        cv2.waitKey(500)
        cv2.destroyAllWindows()
        print(f"--- Eval complete. Steps: {steps}, Total Reward: {total_reward:.1f} ---\n")


def train():
    train_env = VelocityTrackingEnv()
    eval_env = VelocityTrackingEnv()

    model_path = None
    if LOAD_EXISTING_MODEL:
        existing_models = [f for f in os.listdir(SAVE_DIR)
                          if f.startswith(MODEL_NAME) and f.endswith('.zip')]
        if existing_models:
            # Pick the most recently modified file
            latest = max(existing_models,
                        key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)))
            model_path = os.path.join(SAVE_DIR, latest)
            print(f"\n[LOAD] Resuming from latest checkpoint: {model_path}")
        else:
            print("\n[INFO] No existing model found. Starting fresh training.")

    if model_path is not None:
        model = PPO.load(model_path, env=train_env,
                         device="cuda" if torch.cuda.is_available() else "cpu")
        print("[LOAD] Model loaded successfully. Resuming training.\n")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            target_kl=0.02,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU
            ),
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    callback = SimpleVisualCallback(
        eval_env=eval_env,
        save_dir=SAVE_DIR,
        model_name=MODEL_NAME,
        plot_interval=PLOT_INTERVAL,
        render_episode_freq=RENDER_EPISODE_FREQ,
        save_interval=SAVE_EPISODE_INTERVAL
    )

    print("\n[INFO] Starting training for Velocity Expert (natural wind)...")
    print(f"   - Save directory: {os.path.abspath(SAVE_DIR)}")
    print(f"   - Model checkpoint every {SAVE_EPISODE_INTERVAL} episode(s)")
    print(f"   - Reward plot updated every {PLOT_INTERVAL} episode(s)")
    print(f"   - OpenCV render every {RENDER_EPISODE_FREQ} episodes")
    print(f"   - Wind: max ±{train_env.wind_max_force:.0f} N per axis")
    print("   - Per-episode personality + Markov transitions + exp. durations")
    print("   - Press Ctrl+C to interrupt and save the current model & plot.\n")

    try:
        model.learn(total_timesteps=2_000_000, callback=callback)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted. Overwriting base checkpoint...")
        model.save(os.path.join(SAVE_DIR, MODEL_NAME))
        print(f"  [SAVE] {MODEL_NAME}.zip updated with current weights.")
        callback._update_plot()
        sys.exit(0)
    finally:
        final_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_final")
        model.save(final_path)
        print(f"\n[SUCCESS] Final model saved to {final_path}.zip")
        callback._update_plot()
        print("Final plot saved.")


def test(model_path=None, episodes=5):
    env = VelocityTrackingEnv()

    if model_path is None:
        final_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_final.zip")
        base_path  = os.path.join(SAVE_DIR, f"{MODEL_NAME}.zip")

        if os.path.exists(final_path):
            model_path = final_path
            print(f"[INFO] Using final checkpoint: {final_path}")
        elif os.path.exists(base_path):
            model_path = base_path
            print(f"[INFO] Using live checkpoint: {base_path}")
        else:
            print(f"[ERROR] No model found at {base_path} or {final_path}")
            return

    print(f"\n[INFO] Testing model: {model_path}")
    model = PPO.load(model_path, env=env,
                     device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[INFO] Testing for {episodes} episodes (natural wind)...\n")

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        steps = 0
        total_reward = 0.0
        windy_steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            windy_steps += int(env.wind_active)
            env.render(mode="cv2", timesteps=steps, total_reward=total_reward)
            steps += 1

            if cv2.waitKey(1) & 0xFF == 27:
                print("   [ABORT] Testing interrupted by user.")
                cv2.destroyAllWindows()
                return

        cv2.waitKey(500)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}, "
              f"Windy = {windy_steps/steps:.0%} "
              f"(p_on={env.wind_turn_on_prob:.2f}, p_stay={env.wind_stay_on_prob:.2f})")
        cv2.destroyAllWindows()

    print("\n[SUCCESS] Testing complete.")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    torch.set_num_threads(1)

    if MODE.lower() == "train":
        train()
    elif MODE.lower() == "test":
        test(episodes=TEST_EPISODES)
    else:
        print(f"Invalid MODE: '{MODE}'. Use 'train' or 'test'.")
