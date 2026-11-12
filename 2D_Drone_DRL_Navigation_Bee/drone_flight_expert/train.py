"""
velocity_agent_fixed.py

Single‑file solution: environment + training/testing loop.
Set MODE = "train" to train a new model, or "test" to evaluate a saved model.
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
TEST_EPISODES = 5                    # Number of episodes to run in test mode

SAVE_DIR = "/home/deen/ppo_test/drone_flight_expert/"    # Folder for model & plot
MODEL_NAME = "velocity_expert"       # Base name for model files
PLOT_INTERVAL = 1                    # Update reward plot every N episodes
RENDER_EPISODE_FREQ = 50            # OpenCV render every N episodes
SAVE_EPISODE_INTERVAL = 10            # Save model every N episodes
LOAD_EXISTING_MODEL = False          # Set True to resume training from latest checkpoint
# =====================================================================


# =====================================================================
# 1. ENVIRONMENT (with embedded reward logging)
# =====================================================================

class VelocityTrackingEnv(gym.Env):
    """
    A 2D drone velocity‑tracking task.
    The drone must follow (vx_target, vy_target) smoothly.
    """
    def __init__(self):
        super().__init__()
        # ---- Physics ----
        self.mass = 3.0
        self.max_thrust_each = 5.0 * 9.8   # 49 N
        self.g = 9.8
        self.dt = 0.05
        self.d = 0.4
        self.I = 0.3

        # ---- State limits ----
        self.max_v = 10.0
        self.max_theta = np.pi
        self.max_vtheta = 3.0

        # ---- Target update ----
        self.target_change_interval = 50   # steps
        self.step_counter = 0

        # ---- Episode length ----
        self.max_episode_steps = 500

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.vx = np.random.uniform(-0.5, 0.5)
        self.vy = np.random.uniform(-0.5, 0.5)
        self.theta = np.random.uniform(-0.1, 0.1)
        self.vtheta = np.random.uniform(-0.1, 0.1)
        self.step_counter = 0

        self._set_new_target()
        self.prev_action = np.array([0.5, 0.5])

        # Reset episode logging
        self._episode_reward = 0.0
        self._episode_length = 0

        return self._get_obs(), {}

    def _set_new_target(self):
        self.target_vx = np.random.uniform(-5.0, 5.0)
        self.target_vy = np.random.uniform(-5.0, 5.0)

    def _get_obs(self):
        norm_vx = np.clip(self.vx / self.max_v, -1.0, 1.0)
        norm_vy = np.clip(self.vy / self.max_v, -1.0, 1.0)
        norm_theta = np.clip(self.theta / self.max_theta, -1.0, 1.0)
        norm_vtheta = np.clip(self.vtheta / self.max_vtheta, -1.0, 1.0)
        norm_tvx = np.clip(self.target_vx / self.max_v, -1.0, 1.0)
        norm_tvy = np.clip(self.target_vy / self.max_v, -1.0, 1.0)
        return np.array([norm_vx, norm_vy, norm_theta, norm_vtheta,
                         norm_tvx, norm_tvy], dtype=np.float32)

    def render(self, mode="cv2", timesteps=0, total_reward=0):
        if mode != "cv2":
            print(f"vx: {self.vx:.2f}, vy: {self.vy:.2f}, "
                  f"target: ({self.target_vx:.2f}, {self.target_vy:.2f})")
            return

        width, height = 500, 500
        canvas = np.zeros((height, width, 3), dtype=np.uint8) + 30
        center_x, center_y = width // 2, height // 2
        scale = 20

        tvx, tvy = self.target_vx, self.target_vy
        if np.hypot(tvx, tvy) > 0.1:
            end_x = int(center_x + tvx * scale)
            end_y = int(center_y - tvy * scale)
            cv2.arrowedLine(canvas, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)

        cvx, cvy = self.vx, self.vy
        if np.hypot(cvx, cvy) > 0.1:
            end_x = int(center_x + cvx * scale)
            end_y = int(center_y - cvy * scale)
            cv2.arrowedLine(canvas, (center_x, center_y), (end_x, end_y), (255, 100, 0), 2, tipLength=0.3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"Total Reward: {total_reward:.1f}", (10, 30), font, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"Vx: {self.vx:.2f} m/s  (Target: {self.target_vx:.2f})", (10, 60), font, 0.5, (200,200,200), 1)
        cv2.putText(canvas, f"Vy: {self.vy:.2f} m/s  (Target: {self.target_vy:.2f})", (10, 80), font, 0.5, (200,200,200), 1)
        cv2.putText(canvas, f"Tilt: {np.degrees(self.theta):.1f} deg", (10, 110), font, 0.5, (200,200,200), 1)
        cv2.putText(canvas, f"Steps: {timesteps}", (10, 140), font, 0.5, (200,200,200), 1)
        cv2.putText(canvas, "Green Arrow = Target Velocity", (10, height - 30), font, 0.4, (0, 255, 0), 1)
        cv2.putText(canvas, "Orange Arrow = Current Velocity", (10, height - 10), font, 0.4, (255, 100, 0), 1)

        cv2.imshow("Velocity Expert", canvas)
        cv2.waitKey(1)

    def step(self, action):
        self.step_counter += 1
        self._episode_length += 1
        action = np.clip(action, 0.0, 1.0)

        f_left = action[0] * self.max_thrust_each
        f_right = action[1] * self.max_thrust_each
        total_thrust = f_left + f_right

        ax = (total_thrust / self.mass) * np.sin(self.theta)
        ay = (total_thrust / self.mass) * np.cos(self.theta) - self.g
        alpha = ((f_right - f_left) * self.d) / self.I

        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.vtheta += alpha * self.dt
        self.theta += self.vtheta * self.dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # ---- Clip velocities ----
        self.vx = np.clip(self.vx, -20.0, 20.0)
        self.vy = np.clip(self.vy, -20.0, 20.0)

        # ---- Compute errors ----
        vx_err = np.clip(self.vx - self.target_vx, -10.0, 10.0)
        vy_err = np.clip(self.vy - self.target_vy, -10.0, 10.0)

        # ---- Reward ----
        vel_error_penalty = -0.01 * (vx_err**2 + vy_err**2)
        tilt_penalty = -0.001 * self.theta**2
        vtheta_penalty = -0.001 * self.vtheta**2
        ax_penalty = -0.001 * ax**2
        ay_penalty = -0.001 * ay**2
        action_diff = action - self.prev_action
        action_penalty = -0.001 * (action_diff[0]**2 + action_diff[1]**2)
        survival_bonus = 0.01

        reward = (vel_error_penalty + tilt_penalty + vtheta_penalty +
                  ax_penalty + ay_penalty + action_penalty + survival_bonus)

        self.prev_action = action.copy()
        self._episode_reward += reward

        if self.step_counter % self.target_change_interval == 0:
            self._set_new_target()

        # ---- Truncation ----
        truncated = False
        info = {}
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            # Add episode statistics to info
            info['episode_reward'] = self._episode_reward
            info['episode_length'] = self._episode_length

        return self._get_obs(), reward, False, truncated, info


# =====================================================================
# 2. TRAINING / TESTING LOGIC
# =====================================================================

# ---- Custom training callback ----
class SimpleVisualCallback(BaseCallback):
    """
    Logs episode rewards, updates a live plot, renders an OpenCV eval episode,
    and saves the model periodically.
    """
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

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[CALLBACK] Save directory: {os.path.abspath(self.save_dir)}")

        # ---- Live plot ----
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        plt.show(block=False)

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

                    # print(f"\n[EPISODE {self.ep_counter}] Reward = {reward:.2f}, Length = {length} steps")

                    # Save model periodically
                    if self.ep_counter % self.save_interval == 0:
                        save_path = os.path.join(self.save_dir, f"{self.model_name}")
                        self.model.save(save_path)
                        # print(f"  [SAVE] Model saved to {save_path}.zip")

                    # Update plot periodically
                    if self.ep_counter % self.plot_interval == 0:
                        self._update_plot()

                    # Render eval episode periodically
                    if self.ep_counter % self.render_episode_freq == 0:
                        self._render_eval_episode()
                else:
                    print(f"\n[WARNING] Episode ended but info missing reward/length. info keys: {info.keys() if info else 'empty'}")
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
        # print(f"  [PLOT] Updated reward plot -> {plot_path}")

    def _render_eval_episode(self):
        print(f"\n--- [VISUAL EVAL] Episode {self.ep_counter} ---")
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
    """Train a new velocity expert (or resume from checkpoint)."""
    # ---- Create environments (no Monitor wrapper – we do logging inside env) ----
    train_env = VelocityTrackingEnv()
    eval_env = VelocityTrackingEnv()

    # ---- Check for existing model ----
    model_path = None
    if LOAD_EXISTING_MODEL:
        existing_models = [f for f in os.listdir(SAVE_DIR) if f.startswith(MODEL_NAME) and f.endswith('.zip')]
        if existing_models:
            def get_ep_num(fname):
                try:
                    return int(fname.split('_ep')[-1].split('.')[0])
                except:
                    return 0
            latest = max(existing_models, key=get_ep_num)
            model_path = os.path.join(SAVE_DIR, latest)
            print(f"\n[LOAD] Found existing model: {model_path}. Loading...")
        else:
            print("\n[INFO] No existing model found. Starting fresh training.")

    # ---- Create model ----
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

    # ---- Callback ----
    callback = SimpleVisualCallback(
        eval_env=eval_env,
        save_dir=SAVE_DIR,
        model_name=MODEL_NAME,
        plot_interval=PLOT_INTERVAL,
        render_episode_freq=RENDER_EPISODE_FREQ,
        save_interval=SAVE_EPISODE_INTERVAL
    )

    print("\n[INFO] Starting training for Velocity Expert...")
    print(f"   - Save directory: {os.path.abspath(SAVE_DIR)}")
    print(f"   - Model checkpoint every {SAVE_EPISODE_INTERVAL} episode(s)")
    print(f"   - Reward plot updated every {PLOT_INTERVAL} episode(s)")
    print(f"   - OpenCV render every {RENDER_EPISODE_FREQ} episodes")
    print("   - Press Ctrl+C to interrupt and save the current model & plot.\n")

    try:
        model.learn(total_timesteps=2_000_000, callback=callback)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user. Saving current model and plot...")
        # Save the model with a special name
        interrupt_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_interrupt")
        model.save(interrupt_path)
        print(f"  [SAVE] Model saved to {interrupt_path}.zip")
        # Force the callback to save the plot now
        callback._update_plot()
        print("  [SAVE] Plot updated.")
        # Save as latest episode checkpoint if any
        if callback.ep_counter > 0:
            ep_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_ep{callback.ep_counter}")
            model.save(ep_path)
            print(f"  [SAVE] Model also saved as {ep_path}.zip")
        sys.exit(0)
    finally:
        # Always save a final version (in case of normal completion or interrupt)
        final_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_final")
        model.save(final_path)
        print(f"\n[SUCCESS] Final model saved to {final_path}.zip")
        # Also save the plot one last time
        callback._update_plot()
        print("Final plot saved.")


def test(model_path=None, episodes=5):
    """Test a saved model with rendering."""
    env = VelocityTrackingEnv()

    if model_path is None:
        existing_models = [f for f in os.listdir(SAVE_DIR) if f.startswith(MODEL_NAME) and f.endswith('.zip')]
        if not existing_models:
            print(f"[ERROR] No model found in {SAVE_DIR}")
            return
        final_models = [f for f in existing_models if 'final' in f]
        if final_models:
            model_path = os.path.join(SAVE_DIR, final_models[0])
        else:
            def get_ep_num(fname):
                try:
                    return int(fname.split('_ep')[-1].split('.')[0])
                except:
                    return 0
            latest = max(existing_models, key=get_ep_num)
            model_path = os.path.join(SAVE_DIR, latest)

    print(f"\n[INFO] Testing model: {model_path}")
    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[INFO] Testing for {episodes} episodes...\n")

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        steps = 0
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            env.render(mode="cv2", timesteps=steps, total_reward=total_reward)
            steps += 1

            if cv2.waitKey(1) & 0xFF == 27:
                print("   [ABORT] Testing interrupted by user.")
                cv2.destroyAllWindows()
                return

        cv2.waitKey(500)
        # print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}")
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