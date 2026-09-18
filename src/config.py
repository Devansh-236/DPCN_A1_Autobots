"""Shared configuration for the Opinion Network Formation pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RAW_CSV = DATA / "Survey_Results_UC.csv"

# --- Data preparation -------------------------------------------------------
# 5-point Likert -> symmetric integer scale centred on Neutral.
LIKERT_MAP = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}
# "No Comments" is an explicit refusal to answer, not a midpoint opinion.
MISSING_TOKENS = {"No Comments", "", "NA", "N/A"}

# A respondent must have answered at least this share of the 60 items to be a node.
MIN_COMPLETENESS = 0.50
# Two respondents need at least this many commonly-answered items to get a weight.
MIN_OVERLAP = 20

DOMAINS = {"T": "Technology", "E": "Education", "S": "Ethics & Society", "V": "Environment"}

# --- Network construction ---------------------------------------------------
NULL_PERMUTATIONS = 500      # size of the permutation null used to pick the threshold
NULL_PERCENTILE = 99.0       # keep edges stronger than this percentile of the null
KNN_K = 5                    # k for the mutual-kNN robustness check
RANDOM_SEED = 42

# --- Palette (validated categorical slots, light surface) -------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e5e4e0"
DIVERGING = ("#2a78d6", "#f0efec", "#d03b3b")   # blue <-> neutral gray <-> red
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
