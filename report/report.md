---
title: "Opinion Network Formation from a Class Survey"
subtitle: "Assignment 1 — Distributed Protocols for Computer Networks"
date: "18 September 2026"
geometry: "a4paper,top=1.8cm,bottom=1.8cm,left=1.9cm,right=1.9cm"
fontsize: 10pt
linestretch: 1.0
colorlinks: true
linkcolor: "blue"
urlcolor: "blue"
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
  \usepackage{booktabs}
  \usepackage{caption}
  \usepackage{needspace}
  \captionsetup{font=small,labelfont=bf}
  \setlength{\parskip}{0.30em}
  \usepackage{titlesec}
  \titlespacing*{\section}{0pt}{1.1em}{0.45em}
  \titlespacing*{\subsection}{0pt}{0.8em}{0.35em}
  \setlength{\parindent}{0pt}
---

# Team Name

**Autobots**

| Roll number | Name |
|:--|:--|
| 2023102042 | Devansh Varshney |
| 2023102043 | Vansh Agarwal |
| 2023102031 | Aasrith Reddy Vedanaparti |

# GitHub Link to Our Code

**[https://github.com/Devansh-236/DPCN\_assignment-1-Autobots](https://github.com/Devansh-236/DPCN_assignment-1-Autobots)**

The repository contains the complete, reproducible pipeline. `python src/run_all.py` regenerates
every number and every figure in this report from the raw CSV in one command.

```
data/Survey_Results_UC.csv   raw export, unmodified
src/config.py                every tunable constant in one place
src/prepare_data.py          step 1 - cleaning and encoding
src/build_network.py         steps 2-4 - similarity, null model, edge lists
src/analyze.py               step 5 - metrics, communities, latent axes
src/visualize.py             the nine figures below (+ viz_style.py)
src/run_all.py               runs the whole pipeline in one command
results/                     every intermediate artefact as CSV / JSON
figures/                     the nine figures, as published here
```

Nothing here was typed by hand: each stage writes its numbers to `results/*.json` and the figures
read those files, so text, tables and charts cannot drift apart. All randomness is seeded.

# Dataset Documentation

## What the raw file contains

The export `Survey_Results_UC.csv` holds **96 response rows** and **61 columns**: one response ID
plus **60 opinion statements**. The statements are organised into four thematic blocks of fifteen,
identifiable from their column prefixes:

| Code | Domain | Items | A statement from the block |
|:--|:--|--:|:--|
| T | Technology | 15 | *"AI will improve society more than it will create problems."* |
| E | Education | 15 | *"Project-based learning develops deeper understanding than exams."* |
| S | Ethics & Society | 15 | *"Diverse teams generally make better decisions."* |
| V | Environment | 15 | *"Climate change requires immediate global action."* |

Every cell is one of seven strings. Five are Likert points; two are non-answers.

## Interpretation decisions

**1. Likert points become a symmetric integer scale.** `Strongly Disagree → −2`, `Disagree → −1`,
`Neutral → 0`, `Agree → +1`, `Strongly Agree → +2`. The scale is centred on Neutral so that
"no opinion" is numerically zero and disagreement is genuinely the negative of agreement — this
matters because every similarity we compute is an inner product of these vectors.

**2. "No Comments" is treated as missing, not as Neutral.** The survey offered *Neutral* and
*No Comments* as separate choices, so a respondent who picked *No Comments* (37 cells) deliberately
declined rather than taking a middle position. Coding it as 0 would manufacture agreement between
everyone who skipped a question. It is coded as missing instead.

**3. Blank cells (505, or 8.8% of the 5,760-cell grid) are missing.**

**4. Non-participants are removed, partial participants are kept.** Missingness is not scattered
— it is *block-wise*, exactly what one sees when respondents abandon a form (Figure 1, right).
Five respondents answered nothing at all, four stopped after the 15-item Technology block, one
stopped after 45 items and one after 50. We therefore drop any respondent who answered fewer than
**50% of the 60 items**, which removes exactly the nine people who never really participated
(IDs 44, 60, 68, 73, 78, 30, 39, 77, 87) and keeps everyone who worked through at least three
blocks. This leaves **87 respondents**, our network's nodes, with a residual missing rate of only
**1.19%** (62 cells). Those 62 cells are never imputed: every similarity is computed on the items
a given *pair* both answered (pairwise-complete), with a minimum overlap of 20 items.

## What the cleaned data looks like

![The cleaned response matrix. Left: agreement dominates every domain, and the
Environment block is the most consensual. Right: dropout happens at block boundaries, which is why
a completeness cut-off removes non-participants rather than opinions.](../figures/fig1_data_profile.png){width=82%}

After cleaning, **80.0%** of all answers are *Agree* or *Strongly Agree*, 12.9% are *Neutral*, and
only **7.1%** are any form of disagreement. The grand mean is **+1.14** on a scale running to +2.
Fifty-eight of the 60 statements have a positive mean; only two are net-negative.

This single fact drives the entire methodology that follows. **A class this agreeable cannot be
networked on raw agreement** — if we connected respondents whose answers simply look alike, we
would connect everyone to everyone, and the network would describe the questionnaire rather than
the people answering it. The next section addresses this directly.

# Pipeline Followed

## The network we build

**Nodes** are the 87 retained respondents. **Edges** are undirected and weighted: an edge
$(u,v)$ exists when $u$ and $v$ agree with each other *more than the class as a whole agrees
about anything*, and its weight is how much more. The network is therefore a map of **residual
opinion similarity** — who shares an outlook once the shared class consensus has been subtracted.

## Step 1 — Encode and clean

As documented above: map to $\{-2,\dots,+2\}$, treat both non-answer tokens as missing,
drop the nine non-participants. Output: an $87 \times 60$ matrix $X$ with NaNs.

## Step 2 — Remove the class consensus

Each item $j$ is standardised across respondents,
$$ z_{ij} \;=\; \frac{x_{ij} - \mu_j}{\sigma_j}, \qquad
\mu_j,\sigma_j \text{ computed over the respondents who answered item } j. $$

$\mu_j$ is the class consensus on item $j$; $z_{ij}$ is respondent $i$'s **deviation** from it.
Dividing by $\sigma_j$ gives every item equal say, so that a near-unanimous item (V15, sd 0.55)
cannot outvote a genuinely contested one (E04, sd 1.16).

Figure 2 (left) shows why this step is not cosmetic. Before standardisation, **91.2%** of all
3,741 respondent pairs have a *positive* correlation (mean $+0.264$): everybody endorses climate
action and everybody is cool on written exams, so everybody looks alike. After standardisation the
distribution is centred on zero (mean $-0.011$), and similarity finally means something specific.

## Step 3 — Measure pairwise similarity

For each pair we take the Pearson correlation of their standardised vectors over the items they
both answered:
$$ S_{uv} \;=\; \mathrm{corr}\!\left(z_{u,\mathcal{I}_{uv}},\; z_{v,\mathcal{I}_{uv}}\right),
\qquad \mathcal{I}_{uv} = \{j : \text{both answered } j\},\quad |\mathcal{I}_{uv}| \ge 20. $$
Pearson (rather than cosine) re-centres each respondent, which removes any residual
*acquiescence bias* — a person who agrees enthusiastically with everything no longer looks
similar to every other enthusiast purely on that basis.

## Step 4 — Decide which similarities are real

A threshold picked by eye is the weakest point of most similarity networks, so we derive one from
a **null model**. We permute each item independently across respondents: this preserves every
item's marginal distribution exactly while destroying any genuine respondent-to-respondent
coupling. Re-running steps 2–3 on 60 such permutations yields **224,460 null pair similarities**
(mean $-0.012$, sd $0.130$).

We keep an edge when its similarity exceeds the **99th percentile of that null**,
$\tau = 0.299$. This gives **88 edges** where 37 would be expected by chance alone — a 2.4-fold
enrichment. Figure 2 (right) overlays the observed and null distributions.

![Left: subtracting the class consensus moves the similarity distribution from
"almost everything is positive" to centred on zero. Right: the edge threshold is the 99th
percentile of a permutation null, not a hand-picked number.](../figures/fig2_similarity_threshold.png){width=82%}

Because any threshold is a judgement call, we swept it from 0.05 to 0.525 and recorded how the
network responds (Figure 3). The sweep shows a clear regime change: below $\approx 0.25$ the graph
is a single connected blob of 200+ edges with nothing to say; above $\approx 0.40$ it shatters into
dust. Our null-derived $\tau = 0.299$ lands inside the informative band, just past the point where
the giant component detaches from the periphery.

![Threshold sensitivity. The null-derived $\tau$ sits in the informative regime
between "one blob" and "no network".](../figures/fig3_threshold_sweep.png){width=80%}

## Step 5 — Analyse

We then compute global structure, centralities, and communities (Louvain on the weighted graph,
best of 50 random restarts), compare against 200 Erdős–Rényi and 200 configuration-model graphs,
and — as a robustness check on every structural claim — rebuild the whole network a second way, as
a **mutual 5-nearest-neighbour graph** (156 edges, connected by construction, no threshold at all).

# Analysis and Visualizations

## The network

![The opinion network. Seven Louvain communities occupy the 61-node giant component;
the remaining 26 respondents form small fragments or sit alone.](../figures/fig4_network.png){width=58%}

\Needspace{20\baselineskip}

## Global metrics

| Metric | Opinion network $G$ | Mutual-5NN check | Erdős–Rényi | Configuration model |
|:---|---:|---:|---:|---:|
| Nodes | 87 | 87 | 87 | 87 |
| Edges | 88 | 156 | 88 | 88 |
| Density | 0.0235 | 0.0417 | 0.0235 | 0.0235 |
| Average degree | 2.02 | 3.59 | 2.02 | 2.02 |
| Max / median degree | 7 / 1 | 5 / 4 | — | — |
| Isolates | 13 | 0 | — | — |
| Connected components | 19 | 1 | — | — |
| Giant component | 61 (70.1%) | 87 (100%) | — | — |
| Average clustering $C$ | **0.1009** | 0.1502 | 0.0154 | 0.0168 |
| Transitivity | 0.1806 | 0.1623 | — | — |
| Avg. path length $L$ (giant) | 5.37 | 4.03 | 5.30 | 4.63 |
| Diameter (giant) | 12 | 8 | — | — |
| Degree assortativity | +0.125 | +0.140 | ~0 | ~0 |
| Mean edge weight | 0.371 | 0.322 | — | — |

The headline structural result is the **small-world signature**: clustering is
$C/C_{\text{rand}} = 6.6\times$ the random-graph value while the average path length is
$L/L_{\text{rand}} = 1.01\times$ it, giving $\sigma = 6.5$. Opinion ties are strongly triadic —
if you agree with two people, those two tend to agree with each other — yet the graph is no harder
to traverse than a random one of the same density. Degree is also over-dispersed relative to the
Poisson expectation (Figure 5, left): more isolates *and* more high-degree hubs than chance allows.

![Left: the observed degree distribution against the Poisson curve for a random graph
of the same density. Right: both structural quantities expressed as a ratio to their random-graph
expectation — the small-world signature.](../figures/fig5_structure.png){width=82%}

## Central respondents

| Respondent | Degree | Strength | Betweenness | Closeness | Eigenvector |
|:---|---:|---:|---:|---:|---:|
| #110 | 7 | 2.70 | 0.150 | 0.197 | 0.379 |
| #104 | 6 | 2.27 | 0.111 | 0.186 | 0.359 |
| #49 | 6 | 2.27 | 0.095 | 0.185 | 0.357 |
| #119 | 6 | 2.39 | 0.090 | 0.129 | 0.022 |
| #95 | 6 | 2.25 | 0.093 | 0.160 | 0.201 |

The five centrality measures agree closely (Spearman $\rho$ of degree with strength 0.97, with
betweenness 0.89, with closeness 0.78, with eigenvector 0.74), so there is a single, unambiguous
core rather than competing notions of importance. Respondent **#110** is the class's "median
voice": the person whose opinion profile is closest to the largest number of classmates. Note that
#119 has high degree but near-zero eigenvector centrality — it is a hub of an outlying branch, not
of the core.

## Communities

Louvain finds **six communities** partitioning the 61-node giant component, with modularity
$Q = 0.677$ measured on that component, plus a seventh group (**C6**, four people) that is a
connected component entirely of its own. The partition is highly stable: across 50 random restarts
the mean pairwise adjusted Rand index is **0.94**. The blocks are visible
directly in the similarity matrix (Figure 6).

![The 87×87 similarity matrix with rows ordered by community. Red diagonal blocks
are the communities; the unclustered periphery shows no block structure.](../figures/fig6_heatmap.png){width=37%}

Homophily is real but moderate: mean similarity **within** a community is $+0.108$ against
$-0.029$ **between** communities, a gap of $0.137$ (Cohen's $d = 0.78$).

| # | Size | Mean score | $z_T$ | $z_E$ | $z_S$ | $z_V$ |
|:--|--:|--:|--:|--:|--:|--:|
| C0 | 14 | 0.85 | −0.10 | −0.30 | −0.47 | **−0.75** |
| C1 | 13 | 1.20 | +0.15 | +0.06 | +0.01 | +0.11 |
| C2 | 12 | 1.39 | +0.01 | +0.34 | +0.46 | +0.45 |
| C3 | 10 | 1.37 | **+0.29** | +0.10 | +0.43 | +0.38 |
| C4 | 7 | 1.02 | +0.07 | −0.33 | −0.18 | −0.20 |
| C5 | 5 | 1.29 | −0.11 | +0.02 | +0.37 | **+0.47** |
| C6 | 4 | 0.78 | **−0.60** | −0.10 | −0.39 | −0.68 |

Read against the class mean, each is a coherent stance. **C0**, the *reserved* group, is uniformly restrained and coolest
on environmental commitment (V05 $z=-1.19$). **C1**, *campus-first technologists*, sit near the class average but back AI
education (T04 +0.66) while rejecting online learning (E04 −0.77). **C2** are *research-minded reformers* (E06 +0.90) who are notably cool on AI optimism (T01 −0.55). **C3**, the *regulation-minded progressives*, pair the strongest
enthusiasm for technology with the strongest demand for regulating it (T12 +0.74, T10 +0.70).
**C4**, *exam-hall traditionalists*, defend examinations and attendance while rejecting project-based learning outright
(E01 −1.62). **C5**, *techno-optimist greens*, back environmental goals but dismisses data-protection-first thinking
(T10 −1.58). **C6**, four *contrarians* forming their own component, reject the platform-harm narrative
entirely (T14 −2.26, S06 −1.72).

![Left: how far each community departs from the class mean, by domain. Right: the
same communities plotted on the two dominant latent opinion axes.](../figures/fig7_community_profiles.png){width=82%}

The communities are not an artefact of the clustering algorithm. Projecting respondents onto the
principal axes of the opinion matrix (Figure 7, right) separates them significantly:
$F(6,58) = 9.74$ on PC1 and $F(6,58) = 4.15$ on PC2. **PC1 (19.8% of variance)** is a *breadth of
endorsement* axis — its strongest loadings are S09 (inclusive debate), V09/V10 (reuse and
conscious consumption), S14 (ethical leadership) and E10 (innovation over rote), opposed by E03
(compulsory attendance) and E02 (exams measure knowledge). **PC2 (5.5%)** is an *AI-governance
concern* axis: T12 (stricter AI regulation), T14 (algorithmic polarisation), T10 (data
protection). PC1 correlates with eigenvector centrality at $\rho = +0.47$ — the core of the
network is made of broad endorsers, and scepticism pushes a respondent to the periphery.

## Where the class actually divides

![Top: every item positioned by mean opinion and disagreement. Bottom: the ten most
contested statements.](../figures/fig8_items.png){width=55%}

Disagreement is concentrated in **Education**: the three most contested items of all are E-block
statements, and the Education block has the second-highest mean item sd (0.881) behind Technology
(0.916), against Environment's 0.719. Only two statements in the whole survey have a negative mean
— E03 (compulsory attendance, −0.94, rejected 70:13) and E02 (written exams measure knowledge,
−0.16, rejected 43:32). A third, T08 (routine AI diagnosis), sits essentially at zero (+0.07) with
the class split 28:37.

## Do the survey's four themes exist in the data?

We built a second network with the 60 **items** as nodes, joining two items when responses to them
co-vary in the top decile of all item pairs. If the survey's T/E/S/V design reflected how students
actually think, the detected clusters should reproduce it.

They do not. Five clusters emerge (sizes 14, 14, 13, 4, 3) and every one of them mixes themes;
agreement with the survey's own domains is **ARI = 0.12**, barely above the 0 of an unrelated
partition.

![Left: the issue network. Right: what each detected cluster is made of — no cluster
corresponds to a survey theme.](../figures/fig9_issue_network.png){width=82%}

# Results and Discussion

**1. The class is overwhelmingly consensual, and that is the primary finding.** Eighty percent of
all answers are agreement and the grand mean is +1.14. On 57 of 60 statements the class agrees, on
average, with the proposition. There is no polarisation here in the political-science sense: there
is a broad shared value system — sustainability, ethics, inclusion, lifelong learning — and the
network structure lives in the thin residual layer above it.

**2. Once consensus is removed, genuine but weak structure remains.** Observed pair similarities
have sd 0.155 against the null's 0.130. Since $0.155^2 - 0.130^2 = 0.0071$, the *true* similarity
between two classmates has a standard deviation of only about **0.084** — real, reproducible, and
small. Any honest reading of this network must start from that number: 88 edges is 2.4× what chance
gives, but it is not a densely woven social fabric. The 60-item survey simply does not have enough
resolution to measure agreement between two individuals precisely, which is exactly why we
threshold against a null rather than trusting individual pair values.

**3. The network is a small world with a large periphery.** Clustering is 6.6× random while path
length is unchanged ($\sigma = 6.5$), and degree assortativity is positive (+0.125) — well-connected
respondents connect to each other, forming the core visible in Figure 4. But **13 respondents have
no significant tie at all** and 26 (30% of the class) sit outside the giant component. These are
not people with no opinions; they are people whose *combination* of opinions is unusual enough that
nobody else in the class shares it. In an opinion-dynamics reading, they are the individuals least
exposed to reinforcement and most likely to shift.

**4. Seven stable communities, differing in degree rather than in direction.** $Q = 0.677$ with a
94% stable partition across restarts, and within-community similarity exceeds between-community
similarity by 0.137 ($d = 0.78$). Crucially, Figure 7 shows the communities separating mostly along
*how strongly* they endorse the shared value system (PC1, 19.8% of variance), not along opposing
camps. The exceptions are the interesting ones: **C4** rejects project-based and collaborative
learning while defending exams and attendance — a coherent traditionalist position; **C6**, four
respondents, rejects the entire platform-harm narrative (T14 at $z = -2.26$); **C3** pairs
enthusiasm for technology with the strongest demand for regulating it; and **C5** takes the
opposite tack, backing environmental goals while dismissing data-protection-first thinking. These
are recognisable, internally consistent worldviews, not noise.

**5. Education is where the class genuinely argues.** The contested items are pedagogical, not
ethical: compulsory attendance (70% against), the validity of written examinations (43% against),
online learning (28/54 split), and AI-assisted medical diagnosis (28/37 split). Students hold a
shared ethical and environmental outlook but disagree sharply about how they should be taught —
and the traditionalist/reformist split on that question is the single strongest axis in the data.

**6. The survey's thematic blocks are not how students actually organise their opinions.** With
ARI = 0.12, the empirically detected issue clusters cut across Technology / Education / Ethics /
Environment. The largest cluster joins seven Environment items with four Ethics items, two
Education items and one Technology item — a "collective responsibility" dimension that the survey's
own structure splits across three sections. Opinion is organised by underlying value, not by topic
heading.

## Robustness

Every structural claim survives rebuilding the network a completely different way. The mutual-5NN
graph — no threshold, no null model, connected by construction — reproduces the elevated clustering
(0.150), the positive assortativity (+0.140), and a community partition that agrees with the main
one at ARI = 0.42 despite having 10 communities instead of 7 and including the 26 peripheral
respondents the threshold network leaves out. The threshold sweep also shows the clustering excess
is not an artefact of $\tau$: measured against random graphs of matching size at each cut-off, it
is present everywhere and grows monotonically — 1.5× at $\tau=0.10$, 2.4× at 0.20, 5.1× at 0.25,
7.7× at 0.30 and 10.4× at 0.40. Loosening the threshold weakens the result but never reverses it.

## Limitations

*Sixty items is few* for estimating a correlation between two individuals — the null sd of 0.130 is
essentially $1/\sqrt{60}$, and this measurement noise, not methodology, is what caps the density of
the network. *The response scale is severely skewed*, so the effective dynamic range is two points
(Agree / Strongly Agree) rather than five. *Edges represent opinion similarity, not social
contact*: this network says who thinks alike, and nothing about who talks to whom, so it cannot by
itself distinguish homophily from social influence. *A 42% false-discovery rate* (37 of 88 edges
expected by chance) means individual edges should not be over-interpreted; only aggregate structure
is trustworthy, which is why every conclusion above rests on distributions and communities rather
than on particular ties.

# Individual Contribution

| Member | Tasks completed |
|:---|:---|
| Devansh Varshney | [e.g. data interpretation and cleaning decisions (`prepare_data.py`); similarity design, permutation null model and thresholding (`build_network.py`); Dataset Documentation and Pipeline sections] |
| Vansh Agarwal | [e.g. network metrics, centralities, community detection and random-graph baselines (`analyze.py`); Analysis and Visualizations section] |
| Aasrith Reddy Vedanaparti | [e.g. all nine figures and the shared chart styling (`visualize.py`, `viz_style.py`); issue-network analysis; Results and Discussion section] |

*All three members jointly reviewed the methodology, agreed the interpretation of the seven
communities, and proof-read the final report.*
