import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns

os.makedirs("figures", exist_ok=True)

blue = "#2a78d6"
orange = "#eb6834"
dark = "#0b0b0b"
grey = "#52514e"
light_grey = "#898781"
grid_color = "#e6e5df"
axis_color = "#c3c2b7"
colors = {"no": blue, "yes": orange}
names = {"no": "Non-smoker", "yes": "Smoker"}
cmap = LinearSegmentedColormap.from_list("div", ["#256abf", "#f0efec", "#eb6834"])
W = 6.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": axis_color, "axes.labelcolor": grey, "axes.titlecolor": dark,
    "xtick.color": grey, "ytick.color": grey, "text.color": dark,
    "axes.grid": True, "grid.color": grid_color, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100,
})

df = pd.read_csv("clean_data.csv")
bmi_order = ["underweight", "normal", "overweight", "obese", "extreme_obese"]
bmi_names = {"underweight": "Under-\nweight", "normal": "Normal", "overweight": "Over-\nweight",
             "obese": "Obese", "extreme_obese": "Extreme\nobese"}
df["bmi_category"] = pd.Categorical(df["bmi_category"], bmi_order, ordered=True)
regions = ["northeast", "northwest", "southeast", "southwest"]
n = len(df)
no = df[df.smoker == "no"]
yes = df[df.smoker == "yes"]
usd = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")


def save(fig, name):
    for a in fig.axes:
        if "USD" in a.get_ylabel() and not isinstance(a.yaxis.get_major_formatter(), mticker.FuncFormatter):
            a.yaxis.set_major_formatter(usd)
        if "USD" in a.get_xlabel() and not isinstance(a.xaxis.get_major_formatter(), mticker.FuncFormatter):
            a.xaxis.set_major_formatter(usd)
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("saved", name)


def boxes(ax, data, cols, labels=None, positions=None, width=0.55, horizontal=False):
    style = dict(positions=positions, patch_artist=True, widths=width,
                 medianprops=dict(color=dark, linewidth=1.4),
                 whiskerprops=dict(color=grey, linewidth=0.9),
                 capprops=dict(color=grey, linewidth=0.9),
                 flierprops=dict(marker="o", markersize=2.8, markerfacecolor="white",
                                 markeredgecolor=light_grey, markeredgewidth=0.6))
    if horizontal:
        try:
            bp = ax.boxplot(data, orientation="horizontal", **style)
        except TypeError:
            bp = ax.boxplot(data, vert=False, **style)
    else:
        bp = ax.boxplot(data, **style)
    for box, c in zip(bp["boxes"], cols):
        box.set_facecolor(c)
        box.set_alpha(0.45)
        box.set_edgecolor(c)
        box.set_linewidth(1.2)
    if labels is not None:
        pos = positions if positions is not None else range(1, len(data) + 1)
        ax.set_xticks(list(pos))
        ax.set_xticklabels(labels)
        ax.grid(axis="x", visible=False)


def add_n(labels, counts):
    return [f"{l}\nn = {c:,}" for l, c in zip(labels, counts)]


def bar_values(ax, bars, fmt, size=8):
    top = ax.get_ylim()[1]
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.015, fmt(v),
                ha="center", va="bottom", fontsize=size, color=grey)


bins = np.linspace(0, 65000, 27)
fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True)
ax[0].hist(df.charges, bins=bins, color=blue, edgecolor="white", linewidth=0.6)
mean, median = df.charges.mean(), df.charges.median()
ax[0].axvline(mean, color=dark, lw=1.1, ls="--", label=f"Mean = {mean:,.0f}")
ax[0].axvline(median, color=grey, lw=1.1, ls=":", label=f"Median = {median:,.0f}")
ax[0].set(title="(a) All beneficiaries", xlabel="Annual medical charges (USD)", ylabel="Number of people")
ax[0].legend(loc="upper right")
ax[1].hist([no.charges, yes.charges], bins=bins, stacked=True, color=[blue, orange],
           edgecolor="white", linewidth=0.6, label=["Non-smoker", "Smoker"])
ax[1].set(title="(b) By smoking status", xlabel="Annual medical charges (USD)")
ax[1].legend(loc="upper right")
for a in ax:
    a.xaxis.set_major_locator(mticker.MultipleLocator(20000))
fig.tight_layout(w_pad=1.5)
save(fig, "fig_7_1_hist_charges")

q1, q3 = df.charges.quantile([0.25, 0.75])
fence = q3 + 1.5 * (q3 - q1)
n_out = int((df.charges > fence).sum())
fig, ax = plt.subplots(figsize=(W, 2.0))
boxes(ax, [df.charges], [blue], width=0.45, horizontal=True)
ax.set_yticks([])
ax.grid(axis="y", visible=False)
ax.axvline(fence, color=orange, lw=1.0, ls="--")
for x, text, d in [(q1, f"Q1\n{q1:,.0f}", -1), (median, f"Median\n{median:,.0f}", 1), (q3, f"Q3\n{q3:,.0f}", -1)]:
    ax.annotate(text, xy=(x, 1 + 0.24 * d), xytext=(0, 4 * d), textcoords="offset points",
                ha="center", va="bottom" if d > 0 else "top", fontsize=7.5, color=grey)
ax.text(fence + 600, 1.38, f"Upper fence = Q3 + 1.5 × IQR = {fence:,.0f}", fontsize=7.5, color=grey, va="center")
ax.text(fence + 600, 0.66, f"{n_out} outliers above the fence ({n_out / n * 100:.1f}%)", fontsize=7.5,
        color=grey, va="center")
ax.set_ylim(0.45, 1.55)
ax.set(title="Charges: box shows the middle 50%, circles are outliers", xlabel="Annual medical charges (USD)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(10000))
fig.tight_layout()
save(fig, "fig_7_2_box_charges")

fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
labels = ["Non-smoker", "Smoker"]
counts = [len(no), len(yes)]
people = np.array(counts) / n * 100
total = [no.charges.sum(), yes.charges.sum()]
money = np.array(total) / sum(total) * 100
for y, vals in zip([0, 1], [money, people]):
    ax[0].barh(y, vals[0], color=blue, height=0.55, edgecolor="white", linewidth=1.5)
    ax[0].barh(y, vals[1], left=vals[0], color=orange, height=0.55, edgecolor="white", linewidth=1.5)
    ax[0].text(vals[0] / 2, y, f"{vals[0]:.1f}%", ha="center", va="center", color="white",
               fontsize=8.5, fontweight="semibold")
    ax[0].text(vals[0] + vals[1] / 2, y, f"{vals[1]:.1f}%", ha="center", va="center", color="white",
               fontsize=8.5, fontweight="semibold")
ax[0].set_yticks([0, 1])
ax[0].set_yticklabels(["Total charges", "Beneficiaries"])
ax[0].set_xlim(0, 100)
ax[0].set_ylim(-0.6, 1.95)
ax[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax[0].grid(axis="y", visible=False)
ax[0].legend(handles=[Patch(color=blue, label=f"Non-smoker (n = {counts[0]:,})"),
                      Patch(color=orange, label=f"Smoker (n = {counts[1]:,})")],
             loc="upper center", ncol=2, fontsize=8, handlelength=1.2, columnspacing=1.0)
ax[0].set(title="(a) Share of people vs share of all charges", xlabel="Percentage")
means = [no.charges.mean(), yes.charges.mean()]
b = ax[1].bar(labels, means, color=[blue, orange], width=0.55)
ax[1].set_ylim(0, max(means) * 1.25)
bar_values(ax[1], b, lambda v: f"{v:,.0f}")
ax[1].annotate(f"{means[1] / means[0]:.1f}× higher", xy=(0.7, means[1] * 1.15), xytext=(0, means[1] * 1.15),
               fontsize=8, color=grey, ha="center", va="center",
               arrowprops=dict(arrowstyle="->", color=grey, lw=0.8))
ax[1].set(title="(b) Average (mean) charges", ylabel="Annual medical charges (USD)")
ax[1].grid(axis="x", visible=False)
fig.tight_layout(w_pad=2)
save(fig, "fig_7_3_bar_smoker_share_mean")

fig, ax = plt.subplots(figsize=(W * 0.62, 3.2))
data = [no.charges, yes.charges]
boxes(ax, data, [blue, orange], add_n(labels, counts))
for i, g in enumerate(data, start=1):
    ax.text(i + 0.33, g.median(), f"median\n{g.median():,.0f}", fontsize=7.5, color=grey, va="center")
ax.set(title="Charges by smoking status", ylabel="Annual medical charges (USD)")
ax.set_xlim(0.5, 2.85)
fig.tight_layout()
save(fig, "fig_7_4_box_charges_by_smoker")

fig, ax = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True, gridspec_kw={"width_ratios": [2, 1]})
data = [df.loc[df.region == r, "charges"] for r in regions]
boxes(ax[0], data, [blue] * 4, add_n([r.capitalize() for r in regions], [len(x) for x in data]))
ax[0].set(title="(a) By region", ylabel="Annual medical charges (USD)")
data = [df.loc[df.sex == s, "charges"] for s in ["female", "male"]]
boxes(ax[1], data, [blue] * 2, add_n(["Female", "Male"], [len(x) for x in data]))
ax[1].set(title="(b) By sex")
fig.tight_layout(w_pad=1.5)
save(fig, "fig_7_5_box_charges_region_sex")

groups = [("region", r, r.capitalize().replace("east", "-\neast").replace("west", "-\nwest")) for r in regions]
groups += [("sex", "female", "Female"), ("sex", "male", "Male")]
xpos = np.array([0, 1, 2, 3, 4.8, 5.8])
share = [(df.loc[df[c] == v, "smoker"] == "yes").mean() * 100 for c, v, _ in groups]
mean_no = [df.loc[(df[c] == v) & (df.smoker == "no"), "charges"].mean() for c, v, _ in groups]
mean_yes = [df.loc[(df[c] == v) & (df.smoker == "yes"), "charges"].mean() for c, v, _ in groups]
fig, ax = plt.subplots(2, 1, figsize=(W, 5.2), gridspec_kw={"height_ratios": [1, 1.25]})
b = ax[0].bar(xpos, share, color=orange, width=0.6)
overall = (df.smoker == "yes").mean() * 100
ax[0].axhline(overall, color=grey, lw=0.9, ls="--", zorder=0.5)
ax[0].text(xpos[-1] + 0.4, overall, f"All\n{overall:.1f}%", fontsize=7.5, color=grey, va="center", ha="left")
ax[0].set_ylim(0, 30)
for bar in b:
    ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1.0, f"{bar.get_height():.1f}%",
               ha="center", va="top", fontsize=8, color="white", fontweight="semibold")
ax[0].set(title="(a) Share of smokers in each group", ylabel="Smokers (%)")
w = 0.36
b1 = ax[1].bar(xpos - w / 2, mean_no, w, color=blue, label="Non-smoker")
b2 = ax[1].bar(xpos + w / 2, mean_yes, w, color=orange, label="Smoker")
ax[1].set_ylim(0, max(mean_yes) * 1.32)
bar_values(ax[1], list(b1) + list(b2), lambda v: f"{v / 1000:.1f}k", size=7.3)
ax[1].set(title="(b) Average (mean) charges, non-smokers vs smokers", ylabel="Annual medical charges (USD)")
ax[1].legend(loc="upper right", ncol=2)
for a in ax:
    a.set_xticks(xpos)
    a.set_xticklabels([g[2] for g in groups])
    a.grid(axis="x", visible=False)
    a.axvline(3.9, color=axis_color, lw=0.8)
    a.set_xlim(-0.5, 6.75)
fig.tight_layout(h_pad=1.8)
save(fig, "fig_7_6_bar_region_sex_by_smoker")

fig, ax = plt.subplots(figsize=(W, 3.6))
three_groups = [
    ("Non-smoker", df.smoker == "no", dict(color=blue, edgecolor="white", linewidth=0.3, alpha=0.7)),
    ("Smoker, BMI < 30", (df.smoker == "yes") & (df.bmi < 30), dict(facecolor="white", edgecolor=orange, linewidth=0.9)),
    ("Smoker, BMI ≥ 30", (df.smoker == "yes") & (df.bmi >= 30),
     dict(color=orange, edgecolor="white", linewidth=0.3, alpha=0.85)),
]
for name, mask, style in three_groups:
    ax.scatter(df.loc[mask, "age"], df.loc[mask, "charges"], s=12, label=f"{name} (n = {mask.sum():,})", **style)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, markerscale=1.5, handletextpad=0.2,
          columnspacing=1.0, fontsize=8, borderaxespad=0.2)
ax.set_title("Charges versus age", pad=24)
ax.set(xlabel="Age (years)", ylabel="Annual medical charges (USD)")
fig.tight_layout()
save(fig, "fig_7_7_scatter_charges_age")

age_labels = ["18-29", "30-39", "40-49", "50-64"]
df["age_group"] = pd.cut(df.age, [17, 29, 39, 49, 64], labels=age_labels)
avg = df.pivot_table("charges", "age_group", "smoker", aggfunc="mean", observed=True)
cnt = df.pivot_table("charges", "age_group", "smoker", aggfunc="count", observed=True)
fig, ax = plt.subplots(figsize=(W, 3.2))
x = np.arange(len(age_labels))
for s in ["no", "yes"]:
    y = avg[s].values
    ax.plot(x, y, color=colors[s], lw=2, marker="o", markersize=6, label=names[s])
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:,.0f}", (xi, yi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7.5, color=grey)
    ax.text(x[-1] + 0.25, y[-1], f"{names[s]}\n+{y[-1] - y[0]:,.0f}\n(18-29 to 50-64)",
            fontsize=7.5, color=colors[s], va="center")
ax.set_xticks(x)
ax.set_xticklabels([f"{a}\nn = {cnt.loc[a, 'no']:,} / {cnt.loc[a, 'yes']:,}" for a in age_labels])
ax.set_xlim(-0.3, 4.35)
ax.set_ylim(0, 45000)
ax.set(title="Average (mean) charges by age group", xlabel="Age group (years); n = non-smokers / smokers",
       ylabel="Annual medical charges (USD)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig_7_8_line_charges_agegroup")

fig, ax = plt.subplots(figsize=(W, 3.6))
for s, d in [("no", no), ("yes", yes)]:
    ax.scatter(d.bmi, d.charges, s=11, color=colors[s], alpha=0.7, edgecolor="white", linewidth=0.3, label=names[s])
ax.legend(loc="upper left", markerscale=1.6, handletextpad=0.3)
ax.axvline(30, color=grey, lw=1.0, ls="--", zorder=0)
ax.annotate("BMI = 30 (obesity threshold)", xy=(30, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 2), textcoords="offset points", fontsize=8, color=grey, va="bottom", ha="center",
            annotation_clip=False)
ax.set(title="Charges versus BMI", xlabel="Body mass index (kg/m²)", ylabel="Annual medical charges (USD)")
fig.tight_layout()
save(fig, "fig_7_9_scatter_charges_bmi")

fig, ax = plt.subplots(figsize=(W, 3.5))
centers = np.arange(len(bmi_order)) * 1.0
for s, side in [("no", -1), ("yes", 1)]:
    data = [df.loc[(df.bmi_category == c) & (df.smoker == s), "charges"] for c in bmi_order]
    boxes(ax, data, [colors[s]] * len(data), positions=centers + side * 0.2, width=0.34)
ct = pd.crosstab(df.bmi_category, df.smoker)
ax.set_xticks(centers)
ax.set_xticklabels([f"{bmi_names[c]}\nn = {ct.loc[c, 'no']} / {ct.loc[c, 'yes']}" for c in bmi_order], fontsize=7.8)
ax.grid(axis="x", visible=False)
ax.axvline(2.5, color=grey, lw=1.0, ls="--")
ax.text(2.55, 1.0, "BMI ≥ 30 →", transform=ax.get_xaxis_transform(), fontsize=8, color=grey, va="bottom")
ax.legend(handles=[Patch(facecolor=blue, alpha=0.45, edgecolor=blue, label="Non-smoker"),
                   Patch(facecolor=orange, alpha=0.45, edgecolor=orange, label="Smoker")], loc="upper left")
ax.set(title="Charges by BMI category and smoking status", ylabel="Annual medical charges (USD)",
       xlabel="BMI category (WHO, derived by the team); n = non-smokers / smokers")
ax.set_xlim(-0.6, 4.6)
fig.tight_layout()
save(fig, "fig_7_10_box_charges_bmicat_smoker")

cols = ["age", "bmi", "children", "smoker_bin", "charges"]
labels = ["Age", "BMI", "Children", "Smoker\n(0/1)", "Charges"]
fig, ax = plt.subplots(1, 2, figsize=(W, 3.3))
mask = np.triu(np.ones((5, 5), dtype=bool))
for a, method, title in [(ax[0], "pearson", "(a) Pearson r"), (ax[1], "spearman", "(b) Spearman rank correlation")]:
    corr = df[cols].corr(method=method)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0,
                linewidths=2, linecolor="white", square=True, cbar=False,
                annot_kws={"size": 8.5, "color": dark}, ax=a)
    a.set_xticks(np.arange(4) + 0.5)
    a.set_xticklabels(labels[:4], rotation=0, fontsize=7.8)
    a.set_yticks(np.arange(1, 5) + 0.5)
    a.set_yticklabels(labels[1:] if a is ax[0] else [], rotation=0, fontsize=7.8)
    if a is ax[1]:
        a.tick_params(axis="y", left=False)
    a.set_xlim(0, 4)
    a.set_ylim(5, 1)
    a.grid(False)
    a.set_title(title)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.03)
cb.set_label("Correlation")
cb.outline.set_visible(False)
save(fig, "fig_7_11_corr_heatmaps")
