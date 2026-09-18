"""Step 3 - Measure the network: global structure, centrality, communities, latent axes.

Everything printed here is written to results/analysis.json so the report quotes
numbers that came out of the pipeline rather than numbers typed by hand.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

from config import RESULTS, RANDOM_SEED, DOMAINS

rng = np.random.default_rng(RANDOM_SEED)


def load_graph(edge_file, nodes):
    e = pd.read_csv(RESULTS / edge_file, dtype={"source": str, "target": str})
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for _, r in e.iterrows():
        G.add_edge(str(r.source), str(r.target), weight=float(r.weight))
    return G


def global_metrics(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = G.subgraph(comps[0]).copy()
    degs = [d for _, d in G.degree()]
    out = {
        "nodes": n, "edges": m,
        "density": round(nx.density(G), 4),
        "avg_degree": round(2 * m / n, 3),
        "max_degree": int(max(degs)), "min_degree": int(min(degs)),
        "median_degree": float(np.median(degs)),
        "degree_sd": round(float(np.std(degs)), 3),
        "isolates": int(sum(1 for d in degs if d == 0)),
        "components": len(comps),
        "giant_size": giant.number_of_nodes(),
        "giant_share": round(giant.number_of_nodes() / n, 3),
        "avg_clustering": round(nx.average_clustering(G), 4),
        "transitivity": round(nx.transitivity(G), 4),
        "degree_assortativity": round(nx.degree_assortativity_coefficient(G), 4),
        "avg_edge_weight": round(float(np.mean([d["weight"] for *_, d in G.edges(data=True)])), 4),
        "giant_avg_path_length": round(nx.average_shortest_path_length(giant), 4),
        "giant_diameter": int(nx.diameter(giant)),
        "giant_radius": int(nx.radius(giant)),
    }
    return out, giant


def random_baselines(G, n_samples=200):
    """Small-world reference: same n and m (Erdos-Renyi) and same degree sequence."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    er_c, er_l, cm_c, cm_l = [], [], [], []
    deg = [d for _, d in G.degree()]
    for i in range(n_samples):
        R = nx.gnm_random_graph(n, m, seed=int(rng.integers(1e9)))
        er_c.append(nx.average_clustering(R))
        gc = max(nx.connected_components(R), key=len)
        er_l.append(nx.average_shortest_path_length(R.subgraph(gc)))
        C = nx.configuration_model(deg, seed=int(rng.integers(1e9)))
        C = nx.Graph(C); C.remove_edges_from(nx.selfloop_edges(C))
        cm_c.append(nx.average_clustering(C))
        gc = max(nx.connected_components(C), key=len)
        cm_l.append(nx.average_shortest_path_length(C.subgraph(gc)))
    return {
        "samples": n_samples,
        "er_avg_clustering": round(float(np.mean(er_c)), 4),
        "er_avg_path_length": round(float(np.mean(er_l)), 4),
        "config_avg_clustering": round(float(np.mean(cm_c)), 4),
        "config_avg_path_length": round(float(np.mean(cm_l)), 4),
    }


def consensus_communities(G, n_runs=50, resolution=1.0):
    """Louvain is stochastic - run it many times and keep the highest-modularity partition,
    reporting how stable the partition is across runs (mean pairwise ARI)."""
    from sklearn_free_ari import adjusted_rand
    parts = []
    for i in range(n_runs):
        p = louvain_communities(G, weight="weight", seed=i, resolution=resolution)
        parts.append(p)
    qs = [modularity(G, p, weight="weight", resolution=resolution) for p in parts]
    best = parts[int(np.argmax(qs))]
    labels = [partition_labels(G, p) for p in parts]
    aris = [adjusted_rand(labels[i], labels[j])
            for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return best, {
        "runs": n_runs,
        "modularity_best": round(float(max(qs)), 4),
        "modularity_mean": round(float(np.mean(qs)), 4),
        "modularity_sd": round(float(np.std(qs)), 4),
        "n_communities_best": len(best),
        "n_communities_mean": round(float(np.mean([len(p) for p in parts])), 2),
        "partition_stability_ari": round(float(np.mean(aris)), 4),
    }


def partition_labels(G, part):
    lab = {}
    for i, c in enumerate(part):
        for v in c:
            lab[v] = i
    return [lab[v] for v in sorted(G.nodes())]


def main():
    M = pd.read_csv(RESULTS / "opinion_matrix.csv", index_col=0)
    M.index = M.index.astype(str)
    meta = pd.read_csv(RESULTS / "item_metadata.csv")
    nodes = list(M.index)
    S = pd.read_csv(RESULTS / "similarity.csv", index_col=0)
    S.index = S.index.astype(str); S.columns = S.columns.astype(str)

    G = load_graph("edges.csv", nodes)
    Gk = load_graph("edges_knn.csv", nodes)

    res = {}
    res["main_network"], giant = global_metrics(G)
    res["knn_network"], _ = global_metrics(Gk)
    res["random_baselines"] = random_baselines(G)
    c = res["main_network"]["avg_clustering"]; l = res["main_network"]["giant_avg_path_length"]
    rb = res["random_baselines"]
    res["small_world"] = {
        "C_over_Crand": round(c / rb["er_avg_clustering"], 3) if rb["er_avg_clustering"] else None,
        "L_over_Lrand": round(l / rb["er_avg_path_length"], 3),
        "sigma": round((c / rb["er_avg_clustering"]) / (l / rb["er_avg_path_length"]), 3)
        if rb["er_avg_clustering"] else None,
    }

    # ---- centrality ------------------------------------------------------- #
    deg = dict(G.degree())
    strength = {v: round(float(sum(d["weight"] for d in G[v].values())), 4) for v in G}
    btw = nx.betweenness_centrality(G, weight=None, normalized=True)
    clo = nx.closeness_centrality(G)
    # eigenvector centrality is only defined component-wise; compute it on the giant
    # component and score every node outside it as 0.
    eig = {v: 0.0 for v in G}
    eig.update(nx.eigenvector_centrality_numpy(giant, weight="weight"))
    cen = pd.DataFrame({"degree": deg, "strength": strength, "betweenness": btw,
                        "closeness": clo, "eigenvector": eig})
    cen.index.name = "respondent"
    cen = cen.sort_values("degree", ascending=False)
    cen.round(4).to_csv(RESULTS / "centrality.csv")
    res["centrality_top10"] = cen.head(10).round(4).reset_index().to_dict("records")
    res["centrality_correlations"] = cen.corr(method="spearman").round(3).to_dict()

    # ---- communities ------------------------------------------------------ #
    best, cstats = consensus_communities(G)
    res["communities"] = cstats
    def order_partition(part):
        """Size descending, then smallest member id - so labels never depend on set
        iteration order, which varies between Python processes."""
        return sorted(part, key=lambda c: (-len(c), sorted(c)[0]))

    comm = {v: i for i, cset in enumerate(order_partition(best)) for v in cset}
    pd.Series(comm, name="community").rename_axis("respondent").sort_index().to_csv(
        RESULTS / "communities.csv")

    # the same partition restricted to the giant component - modularity on the full
    # graph is inflated by the 13 isolates, each of which is trivially its own community
    bestg, cgstats = consensus_communities(giant)
    res["communities_giant"] = cgstats
    res["communities"]["non_singleton_communities"] = int(sum(1 for c in best if len(c) > 1))
    res["communities"]["communities_with_4plus"] = int(sum(1 for c in best if len(c) >= 4))

    bestk, ckstats = consensus_communities(Gk)
    res["communities_knn"] = ckstats
    commk = {v: i for i, cs in enumerate(order_partition(bestk)) for v in cs}
    from sklearn_free_ari import adjusted_rand
    shared = [v for v in nodes if deg[v] > 0]
    res["communities_knn"]["ari_vs_main"] = round(
        adjusted_rand([comm[v] for v in shared], [commk[v] for v in shared]), 4)

    # ---- community profiles ----------------------------------------------- #
    Z = (M - M.mean()) / M.std(ddof=0)
    sizes = pd.Series(comm).value_counts().sort_index()
    profiles = []
    big = [c for c in sizes.index if sizes[c] >= 4]
    for cidx in sizes.index:
        members = [v for v in nodes if comm.get(v) == cidx]
        zc = Z.loc[members]
        prof = {"community": int(cidx), "size": len(members),
                "members": members,
                "mean_raw_score": round(float(M.loc[members].mean().mean()), 3)}
        for d in DOMAINS:
            cols = meta.loc[meta.domain_code == d, "item"]
            prof[f"z_{d}"] = round(float(zc[cols].mean().mean()), 3)
        if len(members) >= 4:
            diff = zc.mean().sort_values()
            prof["most_negative_items"] = [(i, round(float(diff[i]), 2)) for i in diff.index[:4]]
            prof["most_positive_items"] = [(i, round(float(diff[i]), 2)) for i in diff.index[-4:][::-1]]
        profiles.append(prof)
    res["community_profiles"] = profiles

    # ---- homophily: within vs between community similarity ----------------- #
    within, between = [], []
    idx = {v: k for k, v in enumerate(S.index)}
    Sv = S.to_numpy()
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            va, vb = nodes[a], nodes[b]
            if comm.get(va) is None or comm.get(vb) is None:
                continue
            if sizes[comm[va]] < 4 or sizes[comm[vb]] < 4:
                continue
            s = Sv[idx[va], idx[vb]]
            if np.isnan(s):
                continue
            (within if comm[va] == comm[vb] else between).append(s)
    res["homophily"] = {
        "within_mean": round(float(np.mean(within)), 4),
        "between_mean": round(float(np.mean(between)), 4),
        "gap": round(float(np.mean(within) - np.mean(between)), 4),
        "n_within": len(within), "n_between": len(between),
        "cohens_d": round(float((np.mean(within) - np.mean(between)) /
                                np.sqrt((np.var(within) + np.var(between)) / 2)), 3),
    }

    # ---- item level: consensus and polarisation ---------------------------- #
    item_stats = pd.DataFrame({
        "mean": M.mean(), "sd": M.std(ddof=0),
        "share_agree": (M >= 1).mean(), "share_disagree": (M <= -1).mean(),
        "share_neutral": (M == 0).mean(), "n": M.notna().sum(),
    })
    item_stats = item_stats.join(meta.set_index("item")[["domain_code", "domain", "question"]])
    item_stats["polarisation"] = (item_stats["share_agree"] * item_stats["share_disagree"]) * 4
    item_stats.round(4).sort_values("sd", ascending=False).to_csv(RESULTS / "item_stats.csv")
    res["items_most_divisive"] = item_stats.sort_values("sd", ascending=False).head(8)[
        ["mean", "sd", "share_agree", "share_disagree", "domain", "question"]].round(3
        ).reset_index().to_dict("records")
    res["items_most_consensual"] = item_stats.sort_values("mean", ascending=False).head(6)[
        ["mean", "sd", "share_agree", "domain", "question"]].round(3).reset_index().to_dict("records")
    res["domain_means"] = {d: round(float(M[meta.loc[meta.domain_code == d, "item"]].mean().mean()), 3)
                           for d in DOMAINS}
    res["domain_sds"] = {d: round(float(M[meta.loc[meta.domain_code == d, "item"]].std(ddof=0).mean()), 3)
                         for d in DOMAINS}

    # ---- latent opinion axes (eigen-decomposition of the similarity matrix) - #
    Zf = Z.fillna(0.0)
    U, sv, Vt = np.linalg.svd(Zf.to_numpy(), full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    res["latent_axes"] = {
        "explained_variance_top5": [round(float(v), 4) for v in var[:5]],
        "cumulative_top5": round(float(var[:5].sum()), 4),
    }
    # Sign of a singular vector is arbitrary; orient each axis so that its
    # largest-magnitude loading is positive, which makes "high score" readable.
    for k in range(3):
        if Vt[k][np.argmax(np.abs(Vt[k]))] < 0:
            Vt[k] *= -1
            U[:, k] *= -1
    scores = pd.DataFrame(U[:, :3] * sv[:3], index=Zf.index, columns=["PC1", "PC2", "PC3"])
    scores.round(4).to_csv(RESULTS / "pc_scores.csv")
    load = pd.DataFrame(Vt[:3].T, index=Zf.columns, columns=["PC1", "PC2", "PC3"])
    load.round(4).to_csv(RESULTS / "pc_loadings.csv")
    for pc in ["PC1", "PC2"]:
        s = load[pc].sort_values()
        res[f"{pc}_poles"] = {
            "negative": [(i, round(float(s[i]), 3)) for i in s.index[:5]],
            "positive": [(i, round(float(s[i]), 3)) for i in s.index[-5:][::-1]],
        }

    # ---- how the structural and the latent views line up -------------------- #
    pcs = pd.read_csv(RESULTS / "pc_scores.csv", index_col=0)
    pcs.index = pcs.index.astype(str)
    joined = cen.join(pcs).join(pd.Series(comm, name="community"))
    res["pc_vs_centrality_spearman"] = {
        pc: {m: round(float(joined[[pc, m]].corr(method="spearman").iloc[0, 1]), 3)
             for m in ["degree", "strength", "betweenness", "eigenvector"]}
        for pc in ["PC1", "PC2"]}
    big_c = [c for c in sizes.index if sizes[c] >= 4]
    res["community_pc_means"] = {
        int(c): {pc: round(float(joined.loc[joined.community == c, pc].mean()), 3)
                 for pc in ["PC1", "PC2", "PC3"]} for c in big_c}
    # between-community separation on PC1/PC2 (one-way F statistic, no scipy needed)
    def f_stat(col):
        groups = [joined.loc[joined.community == c, col].to_numpy() for c in big_c]
        allv = np.concatenate(groups); gm = allv.mean()
        ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
        ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
        dfb, dfw = len(groups) - 1, len(allv) - len(groups)
        return round(float((ssb / dfb) / (ssw / dfw)), 3), dfb, dfw
    res["community_separation_F"] = {pc: f_stat(pc) for pc in ["PC1", "PC2", "PC3"]}

    # ---- issue network: do the four survey domains show up empirically? ---- #
    items = list(M.columns)
    GI = load_graph("issue_edges.csv", items)
    res["issue_network"], _ = global_metrics(GI)
    GIg = GI.subgraph(max(nx.connected_components(GI), key=len)).copy()
    ipart, istats = consensus_communities(GIg)
    istats["nodes_clustered"] = GIg.number_of_nodes()
    istats["isolated_items"] = int(sum(1 for _, d in GI.degree() if d == 0))
    res["issue_communities"] = istats
    ilab = {v: i for i, cs in enumerate(order_partition(ipart)) for v in cs}
    items_in = [i for i in items if i in ilab]
    pd.Series(ilab, name="issue_community").rename_axis("item").sort_index().to_csv(
        RESULTS / "issue_communities.csv")
    dom = {r.item: r.domain_code for r in meta.itertuples()}
    res["issue_communities"]["ari_vs_survey_domains"] = round(
        adjusted_rand([ilab[i] for i in items_in], [dom[i] for i in items_in]), 4)
    res["issue_communities"]["sizes"] = sorted([len(c) for c in ipart], reverse=True)
    xt = pd.crosstab(pd.Series({i: dom[i] for i in items_in}, name="survey domain"),
                     pd.Series(ilab, name="detected cluster"))
    xt.to_csv(RESULTS / "issue_domain_crosstab.csv")
    res["issue_domain_crosstab"] = xt.to_dict()

    (RESULTS / "analysis.json").write_text(json.dumps(res, indent=2, default=str))

    print(json.dumps({k: res[k] for k in
                      ["main_network", "knn_network", "random_baselines", "small_world",
                       "communities", "communities_giant", "communities_knn", "homophily",
                       "latent_axes", "community_separation_F", "pc_vs_centrality_spearman",
                       "issue_network", "issue_communities", "domain_means"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
