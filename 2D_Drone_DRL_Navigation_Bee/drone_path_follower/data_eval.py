"""
plot_training_creative.py
Single-plot visualization using TEXTURE differentiation:
  - Success rate: bars (per-episode) + bold trend line  [GOLD - hero]
  - Reward: rolling min/max band + bold trend line       [BLUE - supporting]
  - Episode length: thin line only                        [SLATE - whisper]

Reads only the second (later) run's history.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

H  = "drone_path_follower/path_history.npz"
OUT = "drone_path_follower/slide_training_creative.png"

# ---------- Load ----------
d = np.load(H, allow_pickle=True)
R = np.asarray(d["ep_rewards"], dtype=float)
L = np.asarray(d["ep_lengths"], dtype=float)
S = np.asarray(d["ep_success"],  dtype=float)
N = len(R)
eps = np.arange(1, N + 1)

# ---------- Smoothing ----------
def ema(x, alpha=0.05):
    o = np.empty_like(x)
    o[0] = x[0]
    for i in range(1, len(x)):
        o[i] = alpha * x[i] + (1 - alpha) * o[i - 1]
    return o

def rolling_minmax(x, window=30):
    out_min = np.empty_like(x)
    out_max = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out_min[i] = x[lo:i+1].min()
        out_max[i] = x[lo:i+1].max()
    return out_min, out_max

sm_R = ema(R, 0.05)
sm_L = ema(L, 0.05)
sm_S = ema(S, 0.05)
R_min, R_max = rolling_minmax(R, window=30)

# ---------- Color palette ----------
C_REW     = "#1e3a8a"   # deep navy blue
C_REW_F   = "#60a5fa"   # blue fill
C_SUC     = "#b45309"   # deep amber
C_SUC_F   = "#fbbf24"   # gold bars
C_LEN     = "#475569"   # slate
BG        = "#fafafa"   # soft off-white background

# ---------- Global style ----------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.12,
    "grid.color": "#666",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

fig, ax1 = plt.subplots(figsize=(16, 7.5))
fig.patch.set_facecolor(BG)
ax1.set_facecolor(BG)

# ============================================================
# LAYER 1 (back): SUCCESS as bars — GOLD, hero texture
# ============================================================
# Raw per-episode bars — this is the "0 or 1" nature of success
ax_success = ax1.twinx()
ax_success.bar(eps, S, width=1.0, color=C_SUC_F,
               alpha=0.30, edgecolor="none", zorder=1)
# Bold trend line on top of the bars
ax_success.plot(eps, sm_S, color=C_SUC, lw=2.8, zorder=5)
ax_success.set_ylabel("Success rate", color=C_SUC, fontsize=12, weight="bold")
ax_success.tick_params(axis='y', labelcolor=C_SUC)
ax_success.set_ylim(-0.03, 1.03)
ax_success.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax_success.spines["top"].set_visible(False)
ax_success.spines["right"].set_color(C_SUC)
ax_success.spines["right"].set_linewidth(1.4)

# ============================================================
# LAYER 2 (middle): REWARD as rolling band + line — BLUE
# ============================================================
# Rolling min-max band — gives reward a "river" texture distinct from bars
ax1.fill_between(eps, R_min, R_max, color=C_REW_F, alpha=0.18,
                 linewidth=0, zorder=2, label="_nolegend_")
# Bipolar fill from 0 (secondary emphasis on direction)
ax1.fill_between(eps, 0, sm_R, color=C_REW_F, alpha=0.10,
                 linewidth=0, zorder=2, label="_nolegend_")
# Bold trend line — the readable signal
ax1.plot(eps, sm_R, color=C_REW, lw=2.3, zorder=4)
ax1.axhline(0, color=C_REW, lw=0.6, ls="--", alpha=0.4, zorder=3)

ax1.set_ylabel("Episode reward", color=C_REW, fontsize=12, weight="bold")
ax1.tick_params(axis='y', labelcolor=C_REW)
ax1.spines["left"].set_color(C_REW)
ax1.spines["left"].set_linewidth(1.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)   # hide to let success axis take over

# ============================================================
# LAYER 3 (whisper): EPISODE LENGTH — thin slate line
# ============================================================
ax_len = ax1.twinx()
ax_len.spines["right"].set_position(("axes", 1.10))
ax_len.plot(eps, sm_L, color=C_LEN, lw=1.1, alpha=0.55, zorder=2)
# Add a faint fill for readability without competition
ax_len.fill_between(eps, 0, sm_L, color=C_LEN, alpha=0.04, linewidth=0, zorder=1)
ax_len.set_ylabel("Episode length (steps)", color=C_LEN,
                  fontsize=10, alpha=0.85, style="italic")
ax_len.tick_params(axis='y', labelcolor=C_LEN, labelsize=9, colors=C_LEN)
ax_len.spines["right"].set_color(C_LEN)
ax_len.spines["right"].set_alpha(0.5)
ax_len.spines["top"].set_visible(False)

# ============================================================
# ANNOTATIONS: endpoint values at right edge
# ============================================================
ax1.annotate(f"{sm_R[-1]:+.0f}", xy=(N, sm_R[-1]), xytext=(8, 0),
             textcoords="offset points", fontsize=11, weight="bold",
             color=C_REW, va="center")
ax_success.annotate(f"{sm_S[-1]*100:.0f}%",
                    xy=(N, sm_S[-1]), xytext=(8, 8),
                    textcoords="offset points", fontsize=11, weight="bold",
                    color=C_SUC, va="center")
ax_len.annotate(f"{sm_L[-1]:.0f}", xy=(N, sm_L[-1]), xytext=(8, 0),
                textcoords="offset points", fontsize=9,
                color=C_LEN, alpha=0.8, va="center")

# ============================================================
# TITLES + SUBTITLE
# ============================================================
fig.suptitle(
    "Learning trajectory: reward and success converge while episodes shorten",
    fontsize=14, weight="bold", y=0.985, color="#0f172a")
fig.text(
    0.5, 0.94,
    "Success = gold bars + trend  ·  Reward = blue band + trend  ·  "
    "Episode length = slate line (far right)",
    ha="center", fontsize=10, style="italic", color="#64748b")

ax1.set_xlabel("Episode", fontsize=11, color="#334155")

# ============================================================
# LEGEND (custom, top-left of plot)
# ============================================================
legend_elements = [
    Patch(facecolor=C_SUC_F, alpha=0.35, edgecolor=C_SUC,
          label="Success per episode"),
    Line2D([0], [0], color=C_SUC, lw=2.8, label="Success trend"),
    Patch(facecolor=C_REW_F, alpha=0.20, edgecolor=C_REW,
          label="Reward range (30-ep)"),
    Line2D([0], [0], color=C_REW, lw=2.3, label="Reward trend"),
    Line2D([0], [0], color=C_LEN, lw=1.1, alpha=0.7,
           label="Episode length trend"),
]
ax1.legend(handles=legend_elements, loc="upper left", ncol=5,
           fontsize=9, bbox_to_anchor=(0.0, 1.02),
           handlelength=2.2, columnspacing=1.8)

# ============================================================
# LAYOUT
# ============================================================
plt.tight_layout(rect=(0, 0, 1, 0.91))
plt.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT}")

# ---------- Summary ----------
print(f"\nFinal values (last 20-ep average):")
print(f"  Reward:   {sm_R[-1]:+.1f}")
print(f"  Success:  {sm_S[-1]*100:.1f}%")
print(f"  Length:   {sm_L[-1]:.0f} steps")