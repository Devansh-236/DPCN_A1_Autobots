# Opinion Network Formation from a Class Survey

**Assignment 1 — Distributed Protocols for Computer Networks**
Team **Autobots** — Devansh Varshney, Vansh Agarwal, Aasrith Reddy Vedanaparti

We build and analyse a weighted opinion network from a 60-item class survey covering
Technology, Education, Ethics & Society and Environment. Nodes are respondents; an edge means
two people agree with each other *more than the class as a whole agrees about anything*.

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Headline results

| | |
|---|---|
| Respondents retained | **87** of 96 (nine non-participants removed) |
| Edges | **88**, against 37 expected by chance (2.4× enrichment) |
| Average clustering | **0.101** — 6.6× the random-graph value |
| Small-world $\sigma$ | **6.5** (clustered, but no longer to cross) |
| Communities | **7**, modularity 0.677, 94% stable across restarts |
| Respondents with no significant tie | **13** (30% of the class sits outside the giant component) |
| Percolation regime | **supercritical** — $\langle k \rangle = 2.02$, past $\langle k \rangle = 1$, short of $\ln N = 4.47$ |
| Giant component | **0.70** observed vs **0.80** from $S = 1 - e^{-\langle k \rangle S}$ |
| Mean degree of a neighbour | $\langle k^2 \rangle / \langle k \rangle$ = **3.46** vs $\langle k \rangle$ = 2.02 — 62% of respondents are below their neighbours' average |
| Agreement rate across all answers | **80%** — the class is overwhelmingly consensual |

The most contested statements are all pedagogical (compulsory attendance, the validity of written
exams, online learning), while the four survey themes turn out **not** to be how students organise
their opinions (detected issue clusters vs. survey domains: ARI = 0.12).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py
```

That regenerates every number in `results/` and every figure in `figures/` from
`data/Survey_Results_UC.csv`. Runtime is about 15 seconds.

To rebuild the PDF (needs `pandoc` and a LaTeX engine such as `tectonic`):

```bash
cd report && pandoc report.md -o report.pdf --pdf-engine=tectonic
```

## How the network is built

1. **Encode** — Likert to a symmetric scale (`Strongly Disagree` = −2 … `Strongly Agree` = +2).
   `No Comments` is treated as *missing*, not as Neutral: the survey offered both, so declining to
   answer is not a middle opinion.
2. **Clean** — drop respondents who answered under 50% of items. Dropout is block-wise (people
   abandon the form at section boundaries), so this removes nine non-participants, not opinions.
   Residual missingness is 1.19% and is never imputed — similarities use pairwise-complete items.
3. **Remove the consensus** — standardise each item across respondents. Without this step 91% of
   all respondent pairs correlate positively and the network describes the questionnaire rather
   than the people.
4. **Score similarity** — Pearson correlation of standardised vectors over commonly-answered items
   (minimum overlap 20).
5. **Threshold against a null** — permute each item independently across respondents 60 times
   (224,460 null pairs), and keep only similarities above the null's 99th percentile, τ = 0.299.
   No hand-picked cut-off. A sweep from 0.05 to 0.525 confirms the conclusions are not an artefact
   of τ.
6. **Analyse** — global metrics, five centralities, weighted Louvain (best of 50 restarts),
   200 Erdős–Rényi and 200 configuration-model baselines, and an SVD of the opinion matrix for the
   latent axes.
7. **Benchmark against theory** — the giant-component equation $S = 1 - e^{-\langle k \rangle S}$
   solved by bisection, the degree-dispersion ratio $\mathrm{Var}(k)/\langle k \rangle$, the
   neighbour-degree identity $\langle k^2 \rangle / \langle k \rangle$, the small-world estimate
   $\ln N / \ln \langle k \rangle$ against chain, ring and lattice path lengths, and the Laplacian
   spectrum (zero eigenvalues per component, algebraic connectivity $\lambda_2$).

Every claim is re-checked on an independently constructed **mutual 5-nearest-neighbour** graph,
which uses no threshold at all.

## Repository layout

```
data/Survey_Results_UC.csv     raw export, unmodified
src/config.py                  every tunable constant in one place
src/prepare_data.py            step 1 — cleaning and encoding
src/build_network.py           steps 2–5 — similarity, null model, edge lists
src/analyze.py                 step 6 — metrics, communities, latent axes
src/visualize.py               the nine report figures
src/viz_style.py               shared chart styling
src/sklearn_free_ari.py        adjusted Rand index, dependency-free
src/run_all.py                 runs the whole pipeline
results/                       every intermediate artefact (CSV / JSON)
figures/                       the nine figures as published
report/report.md               report source
report/report.pdf              the submitted report
```

### Key outputs in `results/`

| File | Contents |
|---|---|
| `opinion_matrix.csv` | the cleaned 87 × 60 numeric matrix |
| `similarity.csv` / `similarity_raw.csv` | pairwise similarity, with and without item standardisation |
| `edges.csv` / `edges_knn.csv` | the main edge list and the kNN robustness edge list |
| `communities.csv`, `centrality.csv` | per-respondent community label and five centralities |
| `item_stats.csv` | per-item mean, sd, agree/disagree shares |
| `pc_scores.csv`, `pc_loadings.csv` | latent opinion axes |
| `network_summary.json`, `analysis.json` | every number quoted in the report |
| `prep_summary.json` | the cleaning audit trail |

## Reproducibility

All randomness (Louvain restarts, permutation null, random-graph baselines, layout seeds) is
seeded from `RANDOM_SEED` in `src/config.py`, so repeated runs are identical. Dependency versions
are pinned in `requirements.txt`.
