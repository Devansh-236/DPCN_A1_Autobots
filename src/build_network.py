"""Step 2 - Turn the opinion matrix into a weighted respondent network.

Why item-standardisation comes first
------------------------------------
The class agrees with almost everything (grand mean +1.14 on a -2..+2 scale), so the
correlation between two *raw* response vectors is dominated by the shared item profile:
everybody endorses "climate change requires action" and everybody is lukewarm about
"written exams measure knowledge".  That common profile makes every pair look similar and
carries no information about who resembles whom.  Standardising each item across
respondents removes the class consensus and leaves only the *deviation* from it - which is
exactly what an opinion network should be built on.

Outputs
-------
results/similarity_raw.csv          respondent x respondent, before standardisation
results/similarity.csv              respondent x respondent, after standardisation
results/edges.csv                   thresholded edge list of the main network
results/edges_knn.csv               mutual-kNN edge list (robustness check)
results/issue_edges.csv             issue-issue edge list (60 items)
results/network_summary.json        thresholds, null model, sensitivity sweep
"""
import json
import numpy as np
import pandas as pd

from config import (RESULTS, MIN_OVERLAP, NULL_PERMUTATIONS, NULL_PERCENTILE,
                    KNN_K, RANDOM_SEED)

rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------------------------------- #
# similarity
# --------------------------------------------------------------------------- #
def standardise_items(X: np.ndarray) -> np.ndarray:
    """z-score every column (item) using the respondents who answered it."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd < 1e-9, 1.0, sd)        # a unanimous item carries no information
    return (X - mu) / sd


def pairwise_similarity(X: np.ndarray, min_overlap: int = MIN_OVERLAP) -> np.ndarray:
    """Pearson correlation over commonly-answered items (pairwise-complete)."""
    n = X.shape[0]
    S = np.full((n, n), np.nan)
    obs = ~np.isnan(X)
    for a in range(n):
        for b in range(a, n):
            m = obs[a] & obs[b]
            k = int(m.sum())
            if a == b:
                S[a, a] = 1.0
                continue
            if k < min_overlap:
                continue
            xa, xb = X[a, m], X[b, m]
            xa = xa - xa.mean()
            xb = xb - xb.mean()
            denom = np.sqrt((xa ** 2).sum() * (xb ** 2).sum())
            if denom < 1e-12:
                continue
            S[a, b] = S[b, a] = float((xa * xb).sum() / denom)
    return S


def upper(S):
    iu = np.triu_indices_from(S, k=1)
    v = S[iu]
    return v[~np.isnan(v)]


# --------------------------------------------------------------------------- #
# null model: shuffle each item independently across respondents.
# Item marginals are preserved exactly; any respondent-to-respondent coupling is destroyed.
# --------------------------------------------------------------------------- #
def null_distribution(X_raw: np.ndarray, n_perm: int = NULL_PERMUTATIONS) -> np.ndarray:
    pooled = []
    n, p = X_raw.shape
    for _ in range(n_perm):
        Xp = X_raw.copy()
        for j in range(p):
            col = Xp[:, j]
            obs = ~np.isnan(col)
            vals = col[obs]
            rng.shuffle(vals)
            col[obs] = vals
        Sp = pairwise_similarity(standardise_items(Xp))
        pooled.append(upper(Sp))
    return np.concatenate(pooled)


# --------------------------------------------------------------------------- #
def main():
    M = pd.read_csv(RESULTS / "opinion_matrix.csv", index_col=0)
    M.index = M.index.astype(str)
    ids = list(M.index)
    X_raw = M.to_numpy(dtype=float)

    S_raw = pairwise_similarity(X_raw)                       # no standardisation
    Z = standardise_items(X_raw)
    S = pairwise_similarity(Z)                               # the real similarity

    pd.DataFrame(S_raw, index=ids, columns=ids).to_csv(RESULTS / "similarity_raw.csv")
    pd.DataFrame(S, index=ids, columns=ids).to_csv(RESULTS / "similarity.csv")

    obs_vals = upper(S)
    raw_vals = upper(S_raw)

    # --- threshold from the permutation null ------------------------------- #
    # NULL_PERMUTATIONS full recomputations would be slow; a smaller run is plenty
    # because each permutation already contributes 87*86/2 = 3741 null pairs.
    n_perm = 60
    null_vals = null_distribution(X_raw, n_perm=n_perm)
    tau = float(np.percentile(null_vals, NULL_PERCENTILE))

    # --- sensitivity sweep -------------------------------------------------- #
    import networkx as nx
    sweep = []
    for t in np.round(np.arange(0.05, 0.55, 0.025), 4):
        A = (S > t) & ~np.isnan(S)
        np.fill_diagonal(A, False)
        g = nx.from_numpy_array(np.where(A, S, 0.0))
        g.remove_edges_from([(u, v) for u, v, d in g.edges(data=True) if d["weight"] == 0])
        comps = list(nx.connected_components(g))
        giant = max((len(c) for c in comps), default=0)
        C = nx.average_clustering(g) if g.number_of_edges() else 0.0
        # clustering of random graphs with the same n and m, so the sweep also reports
        # how much of the clustering at each threshold is genuine excess
        m = g.number_of_edges()
        Cr = float(np.mean([nx.average_clustering(
            nx.gnm_random_graph(len(ids), m, seed=int(rng.integers(1e9))))
            for _ in range(40)])) if m else 0.0
        sweep.append({
            "threshold": float(t),
            "edges": m,
            "density": round(nx.density(g), 4),
            "isolates": int(sum(1 for _, d in g.degree() if d == 0)),
            "components": len(comps),
            "giant_share": round(giant / len(ids), 3),
            "avg_clustering": round(C, 4),
            "avg_clustering_random": round(Cr, 4),
            "clustering_excess": round(C / Cr, 2) if Cr > 1e-9 else None,
        })

    # --- main network ------------------------------------------------------- #
    A = (S > tau) & ~np.isnan(S)
    np.fill_diagonal(A, False)
    iu = np.triu_indices_from(A, k=1)
    edges = [(ids[i], ids[j], float(S[i, j]))
             for i, j in zip(*iu) if A[i, j]]
    pd.DataFrame(edges, columns=["source", "target", "weight"]).to_csv(
        RESULTS / "edges.csv", index=False)

    # --- mutual-kNN robustness network -------------------------------------- #
    Sk = np.where(np.isnan(S), -np.inf, S)
    np.fill_diagonal(Sk, -np.inf)
    nn = {i: set(np.argsort(-Sk[i])[:KNN_K]) for i in range(len(ids))}
    knn_edges = [(ids[i], ids[j], float(S[i, j]))
                 for i in range(len(ids)) for j in nn[i]
                 if i < j and i in nn[j]]
    pd.DataFrame(knn_edges, columns=["source", "target", "weight"]).to_csv(
        RESULTS / "edges_knn.csv", index=False)

    # --- issue network: items as nodes -------------------------------------- #
    Zi = standardise_items(X_raw).T                 # items x respondents
    SI = pairwise_similarity(Zi, min_overlap=30)
    items = list(M.columns)
    pd.DataFrame(SI, index=items, columns=items).to_csv(RESULTS / "issue_similarity.csv")
    null_i = np.percentile(np.abs(upper(SI)), 90)   # keep the strongest decile of item pairs
    iu = np.triu_indices_from(SI, k=1)
    issue_edges = [(items[i], items[j], float(SI[i, j]))
                   for i, j in zip(*iu)
                   if not np.isnan(SI[i, j]) and SI[i, j] > null_i]
    pd.DataFrame(issue_edges, columns=["source", "target", "weight"]).to_csv(
        RESULTS / "issue_edges.csv", index=False)

    summary = {
        "n_nodes": len(ids),
        "possible_pairs": int(len(obs_vals)),
        "similarity_raw": {"mean": round(float(raw_vals.mean()), 4),
                           "sd": round(float(raw_vals.std()), 4),
                           "min": round(float(raw_vals.min()), 4),
                           "max": round(float(raw_vals.max()), 4),
                           "share_positive": round(float((raw_vals > 0).mean()), 4)},
        "similarity_standardised": {"mean": round(float(obs_vals.mean()), 4),
                                    "sd": round(float(obs_vals.std()), 4),
                                    "min": round(float(obs_vals.min()), 4),
                                    "max": round(float(obs_vals.max()), 4),
                                    "share_positive": round(float((obs_vals > 0).mean()), 4)},
        "null_model": {"type": "independent within-item permutation",
                       "permutations": n_perm,
                       "null_pairs": int(null_vals.size),
                       "null_mean": round(float(null_vals.mean()), 4),
                       "null_sd": round(float(null_vals.std()), 4),
                       "percentile": NULL_PERCENTILE,
                       "threshold_tau": round(tau, 4),
                       "observed_above_tau": int((obs_vals > tau).sum()),
                       "expected_above_tau_by_chance": round(
                           float((1 - NULL_PERCENTILE / 100) * len(obs_vals)), 1)},
        "main_network": {"edges": len(edges),
                         "density": round(2 * len(edges) / (len(ids) * (len(ids) - 1)), 4)},
        "knn_network": {"k": KNN_K, "edges": len(knn_edges)},
        "issue_network": {"nodes": len(items), "threshold": round(float(null_i), 4),
                          "edges": len(issue_edges)},
        "threshold_sweep": sweep,
    }
    (RESULTS / "network_summary.json").write_text(json.dumps(summary, indent=2))

    print("raw similarity      : mean %.3f  (%.1f%% positive)" %
          (raw_vals.mean(), 100 * (raw_vals > 0).mean()))
    print("standardised        : mean %.3f  sd %.3f" % (obs_vals.mean(), obs_vals.std()))
    print("null mean %.4f sd %.4f  ->  tau(p%.0f) = %.4f" %
          (null_vals.mean(), null_vals.std(), NULL_PERCENTILE, tau))
    print("edges kept          : %d of %d pairs (density %.3f); %.0f expected by chance"
          % (len(edges), len(obs_vals),
             2 * len(edges) / (len(ids) * (len(ids) - 1)),
             (1 - NULL_PERCENTILE / 100) * len(obs_vals)))
    print("mutual-kNN edges    : %d" % len(knn_edges))
    print("issue network       : %d edges over %d items" % (len(issue_edges), len(items)))


if __name__ == "__main__":
    main()
