"""Top-down map (cell frame, meters) of the domain-randomization ranges for the first pixel PPO run.

Regenerate from the repo root: python docs/media/ppo_pixels_run1_dr_ranges.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

sns.set()

# --- geometry from the env defaults (pickplace/envs/cell_env_cfg.py, belt.py, usd_builders.py) ---
BOWL_R = 0.07 + 0.006  # inner radius + wall thickness
TABLE = (-0.30, 1.20, -0.55, 0.55)
BELT_Y, BELT_HALF = 0.30, 0.15
BELT_X = (-0.10, 1.00)
ZONE = (0.25, 0.65)
ENTRY_X = 0.20
REACH = 0.80
SPEED = 0.08

# proposed ranges
PALLET_SIZE = (0.22, 0.26)  # along belt, across belt (widened from 0.22)
PALLET_START = (-0.08, 0.08)  # pallet joint position at reset
OFFSET_X = (-0.02, 0.02)  # bowl on pallet, along belt
OFFSET_Y = (-0.04, 0.04)  # bowl on pallet, across belt
SUPPLY_X, SUPPLY_Y = (0.35, 0.55), (-0.20, 0.00)
SUPPLY_NOMINAL = (0.45, -0.10)
FOOD_SPAWN = 0.02
JOINT_OLD, JOINT_NEW = (-0.05, 1.00), (-0.12, 1.00)

WINDOW = (ZONE[1] - ZONE[0]) / SPEED
# time until the bowl center passes the zone end; extremes combine the pallet start shift and the bowl offset on it
TIME_TO_EXIT = {
    "earliest": (ZONE[1] - (ENTRY_X + PALLET_START[0] + OFFSET_X[0])) / SPEED,
    "today": (ZONE[1] - ENTRY_X) / SPEED,
    "latest": (ZONE[1] - (ENTRY_X + PALLET_START[1] + OFFSET_X[1])) / SPEED,
}

C_TABLE, C_BELT, C_ZONE = "0.75", "0.45", "tab:green"
C_SUPPLY, C_RECV, C_PALLET, C_FOOD = "tab:orange", "tab:blue", "tab:purple", "tab:red"
LEGEND_FS, TITLE_FS, LABEL_FS, TICK_FS = 13, 15, 15, 12


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)


fig = plt.figure(figsize=(15, 14), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1.0], hspace=0.04)
ax_map, ax_map_legend = top.subplots(1, 2, width_ratios=[2.0, 1.0])
ax_pallet, ax_joint = bottom.subplots(1, 2, width_ratios=[1.0, 1.0])

# ---------------- panel 1: whole table ----------------
ax = ax_map
ax.add_patch(Rectangle((TABLE[0], TABLE[2]), TABLE[1] - TABLE[0], TABLE[3] - TABLE[2], color=C_TABLE, alpha=0.35))
ax.add_patch(Rectangle((BELT_X[0], BELT_Y - BELT_HALF), BELT_X[1] - BELT_X[0], 2 * BELT_HALF, color=C_BELT, alpha=0.45))
ax.add_patch(Rectangle((ZONE[0], BELT_Y - BELT_HALF), ZONE[1] - ZONE[0], 2 * BELT_HALF, color=C_ZONE, alpha=0.25))
ax.axvline(ENTRY_X, color="black", linewidth=1.2, linestyle=":")
ax.add_patch(Circle((0, 0), 0.10, color="0.2"))
ax.add_patch(Circle((0, 0), REACH, fill=False, color="0.2", linewidth=1.5, linestyle="--"))

ax.add_patch(Rectangle((SUPPLY_X[0], SUPPLY_Y[0]), SUPPLY_X[1] - SUPPLY_X[0], SUPPLY_Y[1] - SUPPLY_Y[0],
                       color=C_SUPPLY, alpha=0.35))
for x in SUPPLY_X:
    for y in SUPPLY_Y:
        ax.add_patch(Circle((x, y), BOWL_R, fill=False, color=C_SUPPLY, linewidth=1.5, linestyle="--"))
ax.add_patch(Circle(SUPPLY_NOMINAL, BOWL_R, fill=False, color=C_SUPPLY, linewidth=2.5))
ax.add_patch(Rectangle((SUPPLY_NOMINAL[0] - FOOD_SPAWN, SUPPLY_NOMINAL[1] - FOOD_SPAWN), 2 * FOOD_SPAWN, 2 * FOOD_SPAWN,
                       color=C_FOOD, alpha=0.6))

for start, style_, width in ((PALLET_START[0], "--", 1.5), (0.0, "-", 2.5), (PALLET_START[1], "--", 1.5)):
    cx = ENTRY_X + start
    ax.add_patch(Rectangle((cx - PALLET_SIZE[0] / 2, BELT_Y - PALLET_SIZE[1] / 2), *PALLET_SIZE, fill=False,
                           color=C_PALLET, linewidth=width, linestyle=style_))
    for oy in (OFFSET_Y if style_ == "--" else (0.0,)):
        ax.add_patch(Circle((cx, BELT_Y + oy), BOWL_R, fill=False, color=C_RECV, linewidth=width, linestyle=style_))

ax.set_xlim(-0.35, 1.25)
ax.set_ylim(-0.60, 0.60)
ax.set_aspect("equal")
ax.set_anchor("N")
style(ax, "Top-down view of the cell at reset", "x, along the belt (m)", "y, across the table (m)")

ax_map_legend.axis("off")
ax_map_legend.set_title("Legend", fontsize=TITLE_FS)
ax_map_legend.legend(
    handles=[
        Patch(color=C_TABLE, alpha=0.5, label="Table 1.5 x 1.1 m"),
        Patch(color=C_BELT, alpha=0.6, label="Belt, 0.30 m wide"),
        Patch(color=C_ZONE, alpha=0.4, label=f"Reach zone 0.25-0.65 m: {WINDOW:.1f} s at {SPEED} m/s"),
        Line2D([], [], color="black", linestyle=":", label="Belt entry x = 0.20 m"),
        Line2D([], [], color="0.2", linestyle="--", label="Arm reach 0.80 m (base at origin)"),
        Patch(color=C_SUPPLY, alpha=0.5, label="Supply bowl center range\nx 0.35 to 0.55 m, y -0.20 to 0.00 m"),
        Line2D([], [], color=C_SUPPLY, linestyle="--", label="Supply bowl at the range corners"),
        Line2D([], [], color=C_SUPPLY, linewidth=2.5, label="Supply bowl today (0.45, -0.10)"),
        Patch(color=C_FOOD, alpha=0.6, label="Food spawn +-2 cm around bowl center"),
        Line2D([], [], color=C_PALLET, linewidth=2.5, label="Pallet at today's start"),
        Line2D([], [], color=C_PALLET, linestyle="--", label="Pallet at earliest / latest start (+-8 cm)"),
        Line2D([], [], color=C_RECV, linestyle="--", label="Receiving bowl at +-4 cm across the belt"),
    ],
    fontsize=LEGEND_FS,
    loc="upper left",
    frameon=True,
    borderaxespad=0.0,
)

# ---------------- panel 2: pallet close-up ----------------
ax = ax_pallet
ax.add_patch(Rectangle((-0.40, -BELT_HALF), 0.80, 2 * BELT_HALF, color=C_BELT, alpha=0.45))
ax.add_patch(Rectangle((-PALLET_SIZE[0] / 2, -PALLET_SIZE[1] / 2), *PALLET_SIZE, color=C_PALLET, alpha=0.25))
ax.add_patch(Rectangle((OFFSET_X[0], OFFSET_Y[0]), OFFSET_X[1] - OFFSET_X[0], OFFSET_Y[1] - OFFSET_Y[0],
                       color=C_RECV, alpha=0.6))
for ox in OFFSET_X:
    for oy in OFFSET_Y:
        ax.add_patch(Circle((ox, oy), BOWL_R, fill=False, color=C_RECV, linewidth=1.5, linestyle="--"))
ax.add_patch(Circle((0, 0), BOWL_R, fill=False, color=C_RECV, linewidth=2.5))
ax.set_xlim(-0.20, 0.20)
ax.set_ylim(-0.17, 0.17)
ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
ax.set_aspect("equal")
ax.set_anchor("N")
style(ax, "Receiving bowl on the pallet", "along the belt (m)", "across the belt (m)")
ax.legend(
    handles=[
        Patch(color=C_BELT, alpha=0.6, label="Belt, half-width 0.15 m"),
        Patch(color=C_PALLET, alpha=0.4, label="Pallet 0.22 x 0.26 m"),
        Patch(color=C_RECV, alpha=0.6, label="Bowl center range +-2 cm x +-4 cm"),
        Line2D([], [], color=C_RECV, linewidth=2.5, label="Bowl rim, centered (r = 7.6 cm)"),
        Line2D([], [], color=C_RECV, linestyle="--", label="Bowl rim at the offset corners"),
    ],
    fontsize=LEGEND_FS,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
)

# ---------------- panel 3: pallet joint travel ----------------
ax = ax_joint
rows = [
    (2.0, JOINT_OLD, "0.4", f"Travel limit today: {JOINT_OLD[0]:+.2f} to {JOINT_OLD[1]:+.2f} m"),
    (1.0, JOINT_NEW, C_PALLET, f"Travel limit proposed: {JOINT_NEW[0]:+.2f} to {JOINT_NEW[1]:+.2f} m"),
    (
        0.0,
        PALLET_START,
        C_RECV,
        f"Start at reset: {PALLET_START[0]:+.2f} to {PALLET_START[1]:+.2f} m\n"
        f"time to zone exit incl. bowl offset: {TIME_TO_EXIT['earliest']:.1f} s (earliest) / "
        f"{TIME_TO_EXIT['today']:.1f} s (today) / {TIME_TO_EXIT['latest']:.1f} s (latest)",
    ),
]
for y, (lo, hi), color, label in rows:
    ax.plot((lo, hi), (y, y), color=color, linewidth=12, solid_capstyle="butt", label=label)
ax.axvline(0.0, color="black", linewidth=1.2, linestyle=":")
ax.set_yticks([y for y, *_ in rows], [""] * len(rows))  # rows are named in the legend below
ax.set_ylim(-0.7, 2.7)
ax.set_xlim(-0.20, 1.05)
style(ax, "Pallet joint position (0 = belt entry)", "joint position along the belt (m)", "")
ax.legend(fontsize=LEGEND_FS, loc="upper center", bbox_to_anchor=(0.5, -0.18))

fig.align_titles()
fig.savefig("docs/media/ppo_pixels_run1_dr_ranges.png", dpi=150, bbox_inches="tight")
print("saved docs/media/ppo_pixels_run1_dr_ranges.png")
