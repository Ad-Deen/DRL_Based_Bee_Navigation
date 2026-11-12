# Hierarchical DRL Navigation for UAS

A three-layer autonomy stack: **A\*** (global planning) → **SAC** (local navigation, LiDAR) → **PPO** (flight control).

```
World → A* → path → SAC → (vx,vy) → PPO → thrusts → drone
```

---

## ⚠️ Before You Run Anything — Fix These Two Paths

Open **`train6.py`** (main script) and edit the **CONFIGURATION** block at the top:

```python
VELOCITY_EXPERT_PATH = "REPLACE_WITH_PATH_TO/velocity_expert_final"
SAVE_DIR            = "REPLACE_WITH_PATH_TO/drone_path_follower/"
```

| Variable | What it points to | What to put |
|---|---|---|
| `VELOCITY_EXPERT_PATH` | The **frozen PPO flight expert** (`.zip`) — required for both training and testing | Full path **without** `.zip`. Example: `/home/user/models/velocity_expert_final` |
| `SAVE_DIR` | Where SAC model + replay buffer + plots + results are saved/loaded | Full directory path **with** trailing `/`. Must exist or be creatable. |

**If `VELOCITY_EXPERT_PATH.zip` doesn't exist, both `train()` and `test()` will print an error and exit.** That's the sanity check.

---

## Files in the Repo

| File | Purpose | Status |
|---|---|---|
| **`train6.py`** | Main system — full training + evaluation pipeline | ✅ Ready |
| **`train7.py`** | Same as train6 but with per-step CSV logging enabled | ✅ Ready |
| **`train8.py`** | Custom SAC (from-scratch PyTorch) — inference verified, training **experimental** | ⚠️ Test only |
| `SAC_policy.py` | Custom SAC implementation (used by train8) | ✅ Verified for inference |
| `requirements.txt` | Python dependencies | ✅ |

---

## Installation

```bash
python -m venv gym_venv
source gym_venv/bin/activate
pip install -r requirements.txt
```

**Requires:** Python 3.10+, PyTorch (CUDA recommended), Stable-Baselines3 2.5+, OpenCV, NumPy, Matplotlib, Gymnasium.

---

## 🟢 Testing (evaluate a pretrained model)

### Step 1 — Set mode to test

In `train6.py`, confirm:

```python
MODE = "test"
TEST_EPISODES = 10          # change as needed
MAP_MODE = "maze"           # "sparse", "maze", or "mixed"
MAX_DYNAMIC_OBSTACLES = 0   # set to 5 for moving obstacles
```

### Step 2 — Make sure a trained model exists

`test()` looks for `path_follower*.zip` in `SAVE_DIR`. If you don't have one yet, train first (see below).

### Step 3 — Run

```bash
python train6.py
```

Renders each episode live. Prints per-episode reward, steps, and success. Final line:

```
Success rate: X/10
```

---

## 🔵 Training (train a new SAC policy)

### Step 1 — Set mode to train

In `train6.py`:

```python
MODE = "train"
LOAD_EXISTING_MODEL = False   # True to resume from existing checkpoint
```

### Step 2 — Run

```bash
python train6.py
```

Training uses **SB3's SAC** for convenience. Saves to `SAVE_DIR`:

- `path_follower.zip` — model checkpoint (every 10 episodes)
- `path_follower_replay_buffer.pkl` — replay buffer
- `path_history.npz` — episode rewards/lengths/successes
- `path_plot.png` — live-updated training curves

**Interrupt with `Ctrl+C`** — saves a final checkpoint and exits cleanly.

**Resume:** set `LOAD_EXISTING_MODEL = True`, run again. Replay buffer will be reloaded too.

**Training hyperparameters** (already tuned, change only if needed):

```python
learning_rate = 1e-4
buffer_size   = 500_000
batch_size    = 512
gamma         = 0.998
tau           = 0.005
ent_coef      = 0.1
```

---

## 📊 `train7.py` — Training with Data Logging

Identical to `train6.py`, **but** the per-step CSV logging is uncommented. For every episode it writes:

```
SAVE_DIR/test_ep_{N}_log.csv
```

Columns include: `step, total_reward, action_vx, action_vy, actual_vx, actual_vy, cte, arc_progress, lookahead_angle_deg, min_lidar, r_terminal, r_pbrs, r_action_pen`.

**Use this when you want to analyze failure modes** — you can plot CTE vs. time, see exactly when/where the policy broke down.

**Run it identically:**

```bash
python train7.py
```

---

## 🔬 `train8.py` — Custom SAC (Experimental)

**What it does:** replaces SB3's SAC with a from-scratch PyTorch implementation (`SAC_policy.py`). Same environment, same PPO flight expert, same rewards — only the SAC policy is custom.

### What works ✅

- **Inference** — verified bit-exact against SB3 (`max |Δa| = 0.00e+00` over 100 random observations)
- **Training step** — the full SAC update (twin-Q, replay buffer, actor loss, critic loss, auto-α, soft targets) is implemented and validated with a 200-step synthetic demo

### What's still experimental ⚠️

- **Full training on this environment** — not yet run end-to-end. The building blocks are verified, but a full training loop on `PathFollowingEnv` hasn't been tested.

### How to test the custom SAC

**1. Verify inference matches SB3:**

```bash
python SAC_policy.py /path/to/path_follower.zip
```

Expected: `✅ PASS — Custom policy is bit-exact with SB3`

**2. Verify the training path computes:**

```bash
python SAC_policy.py /path/to/path_follower.zip --train-demo
```

Expected: `✅ PASS — training path computes and updates parameters`

**3. Run evaluation with custom loader:**

In `train8.py`, the `test()` function calls `CustomSACPolicy.load(...)` instead of `SAC.load(...)`. Run it like `train6.py`:

```python
MODE = "test"
```
```bash
python train8.py
```

**If episodes play identically to `train6.py`, the custom inference is working.** If you see crashes or NaN actions, revert to `train6.py` — the SB3 version is the safe path.

---

## 🎯 Quick Reference — Which File to Run

| Goal | File | Command |
|---|---|---|
| Train a fresh model | `train6.py` | `MODE="train"` → `python train6.py` |
| Evaluate a trained model | `train6.py` | `MODE="test"` → `python train6.py` |
| Train + save per-step CSVs for analysis | `train7.py` | `python train7.py` |
| Verify custom SAC inference (no env needed) | `SAC_policy.py` | `python SAC_policy.py model.zip` |
| Verify custom SAC training step | `SAC_policy.py` | `python SAC_policy.py model.zip --train-demo` |
| Evaluate using custom SAC (experimental) | `train8.py` | `MODE="test"` → `python train8.py` |

---

## ❓ Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: velocity expert missing` | Fix `VELOCITY_EXPERT_PATH` in the config block |
| `FileNotFoundError` when loading model | Check `SAVE_DIR` — test looks for `path_follower*.zip` there |
| `No 'policy.pth' in ...` (train8) | Wrong `.zip` — not an SB3 SAC checkpoint |
| Training won't start | Check `train_env` construction — PPO expert must load first |
| Episodes run but success rate is 0 | Model might not be trained yet, or `MAP_MODE` mismatches training distribution |

---

## What Each `trainN.py` Represents

| File | What it demonstrates |
|---|---|
| `train6.py` | **The baseline.** Standard SB3 SAC — the reference pipeline |
| `train7.py` | **Instrumented.** Adds per-step telemetry for failure attribution |
| `train8.py` | **Custom SAC.** From-scratch PyTorch policy — proves the SAC dependency can be removed |

The evolution `train6 → train7 → train8` mirrors the research progression: first make it work, then make it measurable, then make it independent of the framework.

---

## License

MIT
