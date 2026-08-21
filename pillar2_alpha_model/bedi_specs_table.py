"""Table: Level, Direction, and Difference of BEDI vs. the long/short
strategy's own return -- one row per specification, same style as the
"Does the strategy's return correlate with BEDI?" exhibit. Reads
output/all_angles_regression_summary.csv, doesn't re-run anything.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
CRIMSON = "#C8102E"
INK = "#1A1A1A"

df = pd.read_csv("output/all_angles_regression_summary.csv")
df = df[df["index"] == "BEDI"]

SPECS = ["Level", "Direction", "Difference"]
# Direction/Difference clear 5% contemporaneously, but per this project's
# established read (README, "Does the DIRECTION of K/BEDI explain
# equities?"), that's very likely mechanical/reverse-causation, not a real
# predictive relationship -- flagged as ARTIFACT, not ROBUST.
VERDICTS = {"Level": "NULL", "Direction": "ARTIFACT", "Difference": "ARTIFACT"}
VERDICT_COLOR = {"NULL": SLATE, "ARTIFACT": CRIMSON}

rows = []
for spec in SPECS:
    sub = df[df["spec"] == spec]
    contemp_p = sub[sub["test"] == "contemporaneous"]["p_value"].iloc[0]
    lag_ps = [sub[(sub["test"] == "lagged_joint_F") & (sub["n_lags"] == n)]["p_value"].iloc[0] for n in [1, 2, 3, 4]]
    rows.append([spec, contemp_p] + lag_ps + [VERDICTS[spec]])

fig, ax = plt.subplots(figsize=(10.5, 3.3))
ax.axis("off")
fig.text(0.065, 0.94, "LEVEL, DIRECTION, AND DIFFERENCE OF BEDI VS. LONG/SHORT STRATEGY'S OWN RETURN",
          fontsize=13.5, fontweight="bold", color=NAVY, ha="left", va="top")

col_labels = ["Specification", "Contemp. p", "n=1", "n=2", "n=3", "n=4", "Verdict"]
cell_text = [[r[0], f"{r[1]:.3g}", f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.3f}", f"{r[5]:.3f}", r[6]] for r in rows]

tab = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center",
               bbox=[0, 0, 1, 0.82], colWidths=[0.22, 0.15, 0.1, 0.1, 0.1, 0.1, 0.18])
tab.auto_set_font_size(False)
tab.set_fontsize(10.8)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#E3E6EA")
    if r == 0:
        cell.set_facecolor(NAVY)
        cell.set_text_props(color="white", fontweight="bold")
        continue
    bg = "#F5F6F8" if r % 2 == 0 else "white"
    cell.set_facecolor(bg)
    row = rows[r - 1]
    if c == 0:
        cell.set_text_props(ha="left", color=INK, fontweight="bold")
    elif c == 6:
        cell.set_text_props(color=VERDICT_COLOR[row[6]], fontweight="bold")
    else:
        v = [row[1], row[2], row[3], row[4], row[5]][c - 1]
        color = GOLD if v < 0.05 else INK
        cell.set_text_props(color=color, fontweight="bold" if v < 0.05 else "normal")

plt.savefig("output/bedi_specs_table.png", dpi=150, facecolor="white", bbox_inches="tight")
print("saved output/bedi_specs_table.png")
