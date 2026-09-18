"""Step 4 - Every figure used in the report."""
import json
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import (RESULTS, FIGURES, SERIES, INK, INK_2, INK_MUTED, GRID, SURFACE,
                    DOMAINS, RANDOM_SEED, NULL_PERCENTILE)
from viz_style import apply_style, strip, subtitle, DIVERGING_CMAP, SEQ_CMAP

apply_style()
FIGURES.mkdir(exist_ok=True)

M = pd.read_csv(RESULTS / "opinion_matrix.csv", index_col=0); M.index = M.index.astype(str)
meta = pd.read_csv(RESULTS / "item_metadata.csv")
S = pd.read_csv(RESULTS / "similarity.csv", index_col=0)
S.index = S.index.astype(str); S.columns = S.columns.astype(str)
Sraw = pd.read_csv(RESULTS / "similarity_raw.csv", index_col=0)
comm = pd.read_csv(RESULTS / "communities.csv", index_col=0)["community"]
comm.index = comm.index.astype(str)
cen = pd.read_csv(RESULTS / "centrality.csv", index_col=0); cen.index = cen.index.astype(str)
item_stats = pd.read_csv(RESULTS / "item_stats.csv", index_col=0)
pcs = pd.read_csv(RESULTS / "pc_scores.csv", index_col=0); pcs.index = pcs.index.astype(str)
net = json.load(open(RESULTS / "network_summary.json"))
ana = json.load(open(RESULTS / "analysis.json"))
prep = json.load(open(RESULTS / "prep_summary.json"))

edges = pd.read_csv(RESULTS / "edges.csv", dtype={"source": str, "target": str})
G = nx.Graph(); G.add_nodes_from(M.index)
G.add_weighted_edges_from(edges.itertuples(index=False, name=None))

# communities with >= 4 members get their own colour; everything smaller is "small / isolated"
sizes = comm.value_counts()
BIG = [c for c in sizes.index if sizes[c] >= 4]
CCOL = {c: SERIES[i] for i, c in enumerate(BIG)}
SMALL_COL = "#b9b7ae"


def cc(v):
    return CCOL.get(comm.get(v, -1), SMALL_COL)


def upper(A):
    iu = np.triu_indices_from(A, k=1)
    v = A[iu]
    return v[~np.isnan(v)]


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("  wrote", name)


# --------------------------------------------------------------------------- #
# 1 - what the raw data looks like
# --------------------------------------------------------------------------- #
def fig_data_profile():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1), width_ratios=[1.35, 1])
    labels = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    cols = [DIVERGING_CMAP(x) for x in [0.0, 0.22, 0.5, 0.78, 1.0]]
    cols = ["#1c5cab", "#86b6ef", "#dedcd6", "#eb9a76", "#d03b3b"]
    cols = ["#d03b3b", "#ec835a", "#dedcd6", "#86b6ef", "#2a78d6"]

    ax = axes[0]
    ypos, dom_order = [], list(DOMAINS)[::-1]
    for k, d in enumerate(dom_order):
        items = meta.loc[meta.domain_code == d, "item"]
        vals = M[items].to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        shares = [np.mean(vals == s) * 100 for s in [-2, -1, 0, 1, 2]]
        left = 0.0
        for s, (sh, c) in enumerate(zip(shares, cols)):
            ax.barh(k, sh, left=left, height=0.62, color=c,
                    edgecolor=SURFACE, linewidth=2)
            if sh >= 7:
                ax.text(left + sh / 2, k, f"{sh:.0f}%", ha="center", va="center",
                        fontsize=7.5, color="#ffffff" if s in (0, 4) else INK,
                        fontweight="bold")
            left += sh
        ypos.append(k)
    ax.set_yticks(ypos, [f"{DOMAINS[d]}  ({d})" for d in dom_order])
    ax.set_xlim(0, 100); ax.set_xlabel("share of answered items (%)")
    ax.set_title("The class agrees with almost everything")
    subtitle(ax, "Response distribution by survey domain, 87 retained respondents")
    strip(ax, grid_axis="x")
    ax.legend(handles=[Line2D([], [], marker="s", linestyle="", markersize=7,
                              color=c, label=l) for c, l in zip(cols, labels)],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5,
              handletextpad=0.4, columnspacing=1.0)

    ax = axes[1]
    answered = M.notna().sum(axis=1)                      # retained nodes
    dropped_counts = {0: 5, 15: 4}                        # the nine excluded respondents
    counts = answered.value_counts().sort_index()
    ax.bar(list(dropped_counts), list(dropped_counts.values()), width=1.6,
           color="#d03b3b", label="excluded (9)")
    ax.bar(counts.index, counts.values, width=1.6, color=SERIES[0], label="retained (87)")
    for x, y in list(dropped_counts.items()) + list(counts.items()):
        if y >= 4:
            ax.text(x, y + 1.2, str(y), ha="center", fontsize=7.2, color=INK_2)
    ax.axvline(30, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(28.5, counts.max() * 0.80, "50% completeness\ncut-off", ha="right",
            fontsize=7.3, color=INK_2)
    ax.set_xlim(-3, 63); ax.set_ylim(0, counts.max() * 1.18)
    ax.set_xlabel("items answered (of 60)"); ax.set_ylabel("respondents")
    ax.set_title("Dropout is block-wise, not scattered")
    subtitle(ax, "Respondents stop at block boundaries — 0, 15, 45 or 50 items — "
                 "so a\ncompleteness cut-off removes whole non-participants, not opinions")
    strip(ax); ax.legend(loc="upper left")
    save(fig, "fig1_data_profile.png")


# --------------------------------------------------------------------------- #
# 2 - why item-standardisation, and where the threshold comes from
# --------------------------------------------------------------------------- #
def fig_similarity_and_threshold():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
    ax = axes[0]
    bins = np.linspace(-0.6, 1.0, 55)
    ax.hist(upper(Sraw.to_numpy()), bins=bins, color=SERIES[3], alpha=0.95,
            label="raw responses")
    ax.hist(upper(S.to_numpy()), bins=bins, color=SERIES[0], alpha=0.75,
            label="after item standardisation")
    ax.axvline(0, color=INK_2, linewidth=1.0)
    ax.set_xlabel("pairwise similarity"); ax.set_ylabel("respondent pairs")
    ax.set_title("Removing the class consensus uncovers the signal")
    subtitle(ax, "91% of raw pairs are positive: that measures the survey, not the people")
    strip(ax); ax.legend(loc="upper right")

    ax = axes[1]
    obs = upper(S.to_numpy())
    nullm, nulls = net["null_model"]["null_mean"], net["null_model"]["null_sd"]
    x = np.linspace(-0.6, 0.8, 400)
    nd = np.exp(-0.5 * ((x - nullm) / nulls) ** 2) / (nulls * np.sqrt(2 * np.pi))
    ax.hist(obs, bins=np.linspace(-0.6, 0.8, 60), density=True, color=SERIES[0],
            alpha=0.85, label="observed pairs")
    ax.plot(x, nd, color="#d03b3b", linewidth=2, label="permutation null")
    tau = net["null_model"]["threshold_tau"]
    ax.axvline(tau, color=INK, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(tau + 0.02, ax.get_ylim()[1] * 0.85, f"$\\tau$ = {tau:.3f}\n(null 99th pct)",
            fontsize=7.5, color=INK)
    ax.set_xlabel("pairwise similarity"); ax.set_ylabel("density")
    ax.set_title("Edges are the pairs the null cannot explain")
    subtitle(ax, f"{net['null_model']['observed_above_tau']} pairs clear $\\tau$; "
                 f"{net['null_model']['expected_above_tau_by_chance']:.0f} would by chance alone")
    strip(ax); ax.legend(loc="upper left")
    save(fig, "fig2_similarity_threshold.png")


# --------------------------------------------------------------------------- #
# 3 - threshold sensitivity
# --------------------------------------------------------------------------- #
def fig_threshold_sweep():
    sw = pd.DataFrame(net["threshold_sweep"])
    tau = net["null_model"]["threshold_tau"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharex=True)
    panels = [("edges", "edges retained", SERIES[0]),
              ("giant_share", "share of nodes in giant component", SERIES[2]),
              ("avg_clustering", "average clustering coefficient", SERIES[1])]
    for ax, (col, lab, c) in zip(axes, panels):
        ax.plot(sw.threshold, sw[col], color=c)
        ax.axvline(tau, color=INK_2, linewidth=1.1, linestyle=(0, (4, 3)))
        i = (sw.threshold - tau).abs().idxmin()
        ax.plot([tau], [sw[col][i]], marker="o", markersize=7, color=c,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(lab, fontsize=8.6)
        ax.set_xlabel("similarity threshold")
        strip(ax)
    axes[0].text(tau + 0.012, sw.edges.max() * 0.82, "chosen $\\tau$", fontsize=7.4, color=INK_2)
    fig.suptitle("The chosen threshold sits where the network stops being one blob",
                 x=0.005, ha="left", fontsize=10, fontweight="bold", color=INK, y=1.26)
    fig.text(0.005, 1.175, "Sensitivity of the network to the similarity cut-off; "
             "every panel shares the same x-axis",
             fontsize=7.8, color=INK_MUTED, ha="left")
    save(fig, "fig3_threshold_sweep.png")


# --------------------------------------------------------------------------- #
# 4 - the network
# --------------------------------------------------------------------------- #
def fig_network():
    """Spring layout on the giant component; the fragments and isolates are parked in a
    labelled strip below it, because a force layout on a disconnected graph just scatters
    them at random and wastes the canvas."""
    # Sets of node ids iterate in an order that varies between Python processes, so every
    # ordering below carries an explicit tiebreak - otherwise the layout is not reproducible.
    comps = sorted((sorted(c, key=int) for c in nx.connected_components(G)),
                   key=lambda c: (-len(c), int(c[0])))
    giant = G.subgraph(comps[0]).copy()
    pos = nx.spring_layout(giant, weight="weight", seed=RANDOM_SEED, k=0.42, iterations=600)
    P = np.array(list(pos.values()))
    lo, hi = P.min(0), P.max(0)
    for v in pos:                                   # normalise the giant to [0,1] x [0.30,1]
        x, y = pos[v]
        pos[v] = np.array([(x - lo[0]) / (hi[0] - lo[0]),
                           0.30 + 0.70 * (y - lo[1]) / (hi[1] - lo[1])])

    strip_nodes = []
    for comp in comps[1:]:
        strip_nodes.append(sorted(comp, key=lambda v: (-G.degree(v), int(v))))
    strip_nodes.sort(key=lambda g: (-len(g), int(g[0])))
    x = 0.01
    for grp in strip_nodes:                         # pairs/triples kept adjacent, then isolates
        for k, v in enumerate(grp):
            pos[v] = np.array([x + k * 0.030, 0.13 if len(grp) > 1 else 0.05])
        x += 0.030 * len(grp) + (0.022 if len(grp) > 1 else 0.004)
        if x > 0.97:
            x = 0.01

    fig, ax = plt.subplots(figsize=(7.6, 5.9))
    ax.set_facecolor(SURFACE)
    for u, v, d in G.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), color="#c9c7bf",
                linewidth=0.4 + 3.2 * (d["weight"] - 0.3), zorder=1, solid_capstyle="round")
    deg = dict(G.degree())
    for v in G.nodes():
        ax.scatter(*pos[v], s=24 + 32 * deg[v], color=cc(v), zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)
    for c in BIG:
        members = [v for v in comps[0] if comm.get(v) == c]   # comps[0] is already sorted
        if len(members) < 2:
            continue
        cx = np.mean([pos[v][0] for v in members]); cy = np.mean([pos[v][1] for v in members])
        ax.text(cx, cy, f"C{c}", fontsize=11, fontweight="bold", color=INK, zorder=6,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=CCOL[c], lw=1.5, alpha=0.94))
    for v in cen.index[:3]:
        ax.annotate(f"#{v}", pos[v], textcoords="offset points", xytext=(10, 7),
                    fontsize=7.3, color=INK_2, zorder=7)
    ax.axhline(0.225, color=GRID, linewidth=1.2)
    ax.text(0.005, 0.195, "FRAGMENTS AND ISOLATES  —  30% of the class has no significant tie",
            fontsize=7.3, color=INK_MUTED, fontweight="bold")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.02, 1.06)
    ax.set_title("The opinion network: a small clustered core, and a third of the class outside it")
    subtitle(ax, f"{G.number_of_nodes()} respondents, {G.number_of_edges()} significant similarity "
                 f"ties; node size = degree, colour = Louvain community, labelled at its centroid")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=CCOL[c],
                      label=f"C{c}  (n={sizes[c]})") for c in BIG]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=7, color=SMALL_COL,
                          label=f"unclustered  (n={int(sum(sizes[c] for c in sizes.index if c not in BIG))})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=4,
              handletextpad=0.4, columnspacing=1.4)
    save(fig, "fig4_network.png")


# --------------------------------------------------------------------------- #
# 5 - degree distribution and the small-world comparison
# --------------------------------------------------------------------------- #
def fig_structure():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), width_ratios=[1.1, 1])
    ax = axes[0]
    deg = np.array([d for _, d in G.degree()])
    vals, counts = np.unique(deg, return_counts=True)
    ax.bar(vals, counts, width=0.68, color=SERIES[0])
    for v, c in zip(vals, counts):
        ax.text(v, c + 0.8, str(c), ha="center", fontsize=7.2, color=INK_2)
    mean_er = G.number_of_edges() * 2 / G.number_of_nodes()
    x = np.arange(0, deg.max() + 1)
    pois = len(deg) * np.exp(-mean_er) * mean_er ** x / np.array([math.factorial(int(i)) for i in x], dtype=float)
    ax.plot(x, pois, color="#d03b3b", linewidth=2, marker="o", markersize=4,
            label="Poisson (random graph, same density)")
    ax.set_xlabel("degree"); ax.set_ylabel("respondents")
    ax.set_title("Degree is over-dispersed relative to a random graph")
    subtitle(ax, f"mean {deg.mean():.2f}, sd {deg.std():.2f}, max {deg.max()}")
    strip(ax); ax.legend(loc="upper right")

    # One axis only: both structural quantities are shown as a ratio to their
    # random-graph expectation, which is what the small-world statistic actually compares.
    ax = axes[1]
    rb = ana["random_baselines"]; mn = ana["main_network"]
    ratios = {
        "Erdős–Rényi\n(same n, m)": (mn["avg_clustering"] / rb["er_avg_clustering"],
                                      mn["giant_avg_path_length"] / rb["er_avg_path_length"]),
        "Configuration\n(same degrees)": (mn["avg_clustering"] / rb["config_avg_clustering"],
                                          mn["giant_avg_path_length"] / rb["config_avg_path_length"]),
    }
    xx = np.arange(len(ratios)); w = 0.34
    cvals = [v[0] for v in ratios.values()]; lvals = [v[1] for v in ratios.values()]
    ax.bar(xx - w / 2 - 0.01, cvals, w, color=SERIES[0], label="clustering  C / C$_{rand}$")
    ax.bar(xx + w / 2 + 0.01, lvals, w, color=SERIES[1], label="path length  L / L$_{rand}$")
    for i, (c_, l_) in enumerate(zip(cvals, lvals)):
        ax.text(i - w / 2 - 0.01, c_ + 0.15, f"{c_:.1f}×", ha="center", fontsize=7.4,
                color=INK_2, fontweight="bold")
        ax.text(i + w / 2 + 0.01, l_ + 0.15, f"{l_:.2f}×", ha="center", fontsize=7.4, color=INK_2)
    ax.axhline(1.0, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(len(ratios) - 0.5, 1.12, "random-graph level", ha="right", fontsize=7.2, color=INK_2)
    ax.set_xticks(xx, list(ratios)); ax.set_ylabel("observed ÷ random baseline")
    ax.set_title("Small-world signature: clustered, yet no further apart")
    subtitle(ax, f"σ = (C/C$_{{rand}}$) ÷ (L/L$_{{rand}}$) = {ana['small_world']['sigma']:.1f}; "
                 f"200 random graphs per baseline")
    ax.set_ylim(0, max(cvals) * 1.28)
    strip(ax); ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig5_structure.png")


# --------------------------------------------------------------------------- #
# 6 - similarity matrix sorted by community
# --------------------------------------------------------------------------- #
def fig_heatmap():
    order = sorted(M.index, key=lambda v: (BIG.index(comm[v]) if comm[v] in BIG else 99,
                                           -cen.loc[v, "degree"]))
    A = S.loc[order, order].to_numpy().copy()
    np.fill_diagonal(A, np.nan)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(A, cmap=DIVERGING_CMAP, vmin=-0.45, vmax=0.45, interpolation="nearest")
    start, ticks, labs = 0, [], []
    for c in BIG:
        n = int(sizes[c])
        ax.add_patch(plt.Rectangle((start - .5, start - .5), n, n, fill=False,
                                   edgecolor=INK, linewidth=1.5))
        ticks.append(start + n / 2 - 0.5); labs.append(f"C{c}")
        start += n
    ticks.append((start + len(order)) / 2 - 0.5); labs.append("unclustered")
    ax.set_xticks([]); ax.set_yticks(ticks, labs, fontsize=7.6)
    ax.tick_params(axis="y", pad=2)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Communities are visible as blocks in the similarity matrix")
    subtitle(ax, "87×87 standardised similarity, rows ordered by Louvain community")
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.set_label("similarity", color=INK_2, fontsize=7.8)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, color=GRID)
    save(fig, "fig6_heatmap.png")


# --------------------------------------------------------------------------- #
# 7 - what each community actually believes
# --------------------------------------------------------------------------- #
def fig_community_profiles():
    profs = {p["community"]: p for p in ana["community_profiles"]}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), width_ratios=[1.25, 1])
    ax = axes[0]
    doms = list(DOMAINS); w = 0.8 / len(BIG)
    xx = np.arange(len(doms))
    for i, c in enumerate(BIG):
        vals = [profs[c][f"z_{d}"] for d in doms]
        ax.bar(xx + i * w - 0.4 + w / 2, vals, w * 0.86, color=CCOL[c],
               label=f"C{c} (n={sizes[c]})")
    ax.axhline(0, color=INK_2, linewidth=1.0)
    ax.set_xticks(xx, [DOMAINS[d] for d in doms])
    ax.set_ylabel("mean standardised opinion (z)")
    ax.set_title("Each community departs from the class mean differently")
    subtitle(ax, "Positive = more supportive than the class average")
    strip(ax); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    ax = axes[1]
    for c in BIG:
        members = [v for v in pcs.index if comm.get(v) == c]
        ax.scatter(pcs.loc[members, "PC1"], pcs.loc[members, "PC2"], s=42, color=CCOL[c],
                   edgecolors=SURFACE, linewidths=1.4, label=f"C{c}", zorder=3)
        ax.annotate(f"C{c}", (pcs.loc[members, "PC1"].mean(), pcs.loc[members, "PC2"].mean()),
                    fontsize=9, fontweight="bold", color=INK, zorder=5, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=CCOL[c], lw=1.2, alpha=0.92))
    rest = [v for v in pcs.index if comm.get(v) not in BIG]
    ax.scatter(pcs.loc[rest, "PC1"], pcs.loc[rest, "PC2"], s=26, color=SMALL_COL,
               edgecolors=SURFACE, linewidths=1.2, zorder=2)
    ax.axhline(0, color=GRID, linewidth=1); ax.axvline(0, color=GRID, linewidth=1)
    v1, v2 = ana["latent_axes"]["explained_variance_top5"][:2]
    ax.set_xlabel(f"PC1 — breadth of endorsement  ({v1:.1%} of variance)")
    ax.set_ylabel(f"PC2 — AI-governance concern  ({v2:.1%})")
    ax.set_title("The same clusters separate on the latent axes")
    subtitle(ax, f"Between-community separation: F(6,58) = {ana['community_separation_F']['PC1'][0]} on PC1")
    strip(ax, grid_axis="both")
    save(fig, "fig7_community_profiles.png")


# --------------------------------------------------------------------------- #
# 8 - consensus and division at the item level
# --------------------------------------------------------------------------- #
def fig_items():
    """Stacked, not side-by-side: the long question labels need the full figure width."""
    dom_col = {d: SERIES[i] for i, d in enumerate(DOMAINS)}
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.6), height_ratios=[1, 1.65],
                             gridspec_kw={"hspace": 0.42})

    ax = axes[0]
    ax.scatter(item_stats["mean"], item_stats["sd"], s=48,
               color=[dom_col[d] for d in item_stats.domain_code],
               edgecolors=SURFACE, linewidths=1.3, zorder=3)
    offsets = {0: (8, 4), 1: (8, -10), 2: (-24, 4), 3: (8, 4), 4: (8, -10)}
    for k, i in enumerate(item_stats.sort_values("sd", ascending=False).index[:5]):
        ax.annotate(i, (item_stats.loc[i, "mean"], item_stats.loc[i, "sd"]),
                    textcoords="offset points", xytext=offsets[k], fontsize=7.4, color=INK)
    for i in item_stats.sort_values("sd").index[:2]:
        ax.annotate(i, (item_stats.loc[i, "mean"], item_stats.loc[i, "sd"]),
                    textcoords="offset points", xytext=(-22, -2), fontsize=7.4, color=INK)
    ax.set_xlabel("mean opinion  (−2 = strongly disagree, +2 = strongly agree)")
    ax.set_ylabel("standard deviation")
    ax.set_title("Consensus and division across the 60 items")
    subtitle(ax, "Bottom-right = settled agreement; top-left = genuinely contested. "
                 "Nothing sits left of zero: not one item was rejected by the class")
    strip(ax, grid_axis="both")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=7,
                              color=dom_col[d], label=DOMAINS[d]) for d in DOMAINS],
              loc="lower left", ncol=2, columnspacing=0.8)

    ax = axes[1]
    top = item_stats.sort_values("sd", ascending=False).head(10).iloc[::-1]
    yy = np.arange(len(top))
    for k, (code, r) in enumerate(top.iterrows()):
        ax.barh(k, r["share_agree"] * 100, color="#2a78d6", height=0.44,
                edgecolor=SURFACE, linewidth=2)
        ax.barh(k, -r["share_disagree"] * 100, color="#d03b3b", height=0.44,
                edgecolor=SURFACE, linewidth=2)
        ax.barh(k, r["share_neutral"] * 100, left=r["share_agree"] * 100,
                color="#dedcd6", height=0.44, edgecolor=SURFACE, linewidth=2)
        ax.text(r["share_agree"] * 100 + r["share_neutral"] * 100 + 2, k,
                f"{r['share_agree'] * 100:.0f}% agree", va="center", fontsize=7,
                color=INK_2)
        ax.text(-r["share_disagree"] * 100 - 2, k, f"{r['share_disagree'] * 100:.0f}%",
                va="center", ha="right", fontsize=7, color=INK_2)
    ax.axvline(0, color=INK, linewidth=1.2)
    for k, (code, r) in enumerate(top.iterrows()):
        q = r["question"]
        ax.text(-84, k + 0.27, f"{code}   {q[:84]}{'…' if len(q) > 84 else ''}",
                fontsize=7.1, color=INK, va="bottom", ha="left")
    ax.set_yticks(yy, ["" for _ in yy])
    ax.set_ylim(-0.7, len(top) - 0.05)
    ax.set_xlim(-84, 116)
    ax.set_xlabel("share of respondents (%)")
    ax.set_title("The ten questions the class splits on")
    subtitle(ax, "Ranked by standard deviation; red = disagree, blue = agree, grey = neutral")
    strip(ax, grid_axis="x")
    save(fig, "fig8_items.png")


# --------------------------------------------------------------------------- #
# 9 - the issue network
# --------------------------------------------------------------------------- #
def fig_issue_network():
    """Two views of the same object: the issue graph, and what each detected cluster is
    made of. The composition panel is what actually settles the question the title asks."""
    ie = pd.read_csv(RESULTS / "issue_edges.csv")
    GI = nx.Graph(); GI.add_nodes_from(M.columns)
    GI.add_weighted_edges_from(ie.itertuples(index=False, name=None))
    GI = GI.subgraph([v for v in GI if GI.degree(v) > 0]).copy()
    icl = pd.read_csv(RESULTS / "issue_communities.csv", index_col=0)["issue_community"]
    dom_col = {d: SERIES[i] for i, d in enumerate(DOMAINS)}
    # One colour scale for the whole figure: colour always means survey domain.
    # Cluster identity is carried by position (left) and by the row label (right).

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), width_ratios=[1.6, 1],
                             gridspec_kw={"wspace": 0.24})

    ax = axes[0]
    pos = nx.kamada_kawai_layout(GI, weight=None)
    pos = nx.spring_layout(GI, pos=pos, weight="weight", seed=RANDOM_SEED,
                           k=0.30, iterations=150)
    for u, v, d in GI.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), color="#cfcdc5",
                linewidth=0.4 + 2.4 * (d["weight"] - 0.35), zorder=1, solid_capstyle="round")
    for v in GI.nodes():
        ax.scatter(*pos[v], s=150, color=dom_col[v[0]], zorder=3,
                   edgecolors=SURFACE, linewidths=1.6)
        ax.text(*pos[v], v, fontsize=5.4, color="#ffffff", fontweight="bold",
                ha="center", va="center", zorder=4)
    ax.margins(0.09)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ic = ana["issue_communities"]
    ax.set_title("The issue network")
    subtitle(ax, f"{GI.number_of_nodes()} connected items, {GI.number_of_edges()} strong ties. "
                 f"The four survey themes do not sit in separate regions.")

    ax = axes[1]
    comp = pd.crosstab(icl, pd.Series({i: i[0] for i in icl.index}))
    comp = comp.reindex(columns=list(DOMAINS), fill_value=0)
    comp = comp.loc[comp.sum(axis=1).sort_values(ascending=False).index]
    yy = np.arange(len(comp))[::-1]
    for y, (cl, row) in zip(yy, comp.iterrows()):
        left = 0
        for d in DOMAINS:
            n = int(row[d])
            if n == 0:
                continue
            ax.barh(y, n, left=left, height=0.68, color=dom_col[d],
                    edgecolor=SURFACE, linewidth=2)
            ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=7.4,
                    color="#ffffff", fontweight="bold")
            left += n
    ax.set_yticks(yy, [f"cluster {c}" for c in comp.index])
    ax.set_xlim(0, 15.4)
    ax.set_xlabel("items in the cluster")
    ax.set_title("Every cluster mixes survey themes")
    subtitle(ax, f"Agreement between detected clusters and the four\nsurvey domains: "
                 f"ARI = {ic['ari_vs_survey_domains']:.2f}  (1.0 = identical, 0 = unrelated)")
    strip(ax, grid_axis="x")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=8,
                               color=dom_col[d], label=f"{DOMAINS[d]} ({d})") for d in DOMAINS],
               loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=4, handletextpad=0.4,
               columnspacing=2.4)
    save(fig, "fig9_issue_network.png")


if __name__ == "__main__":
    fig_data_profile()
    fig_similarity_and_threshold()
    fig_threshold_sweep()
    fig_network()
    fig_structure()
    fig_heatmap()
    fig_community_profiles()
    fig_items()
    fig_issue_network()
