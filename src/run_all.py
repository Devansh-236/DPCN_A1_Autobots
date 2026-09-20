"""Run the whole pipeline end to end: prepare -> build -> analyse -> visualise.

    python src/run_all.py

Each stage writes its outputs to results/ (numbers) and figures/ (charts), so any
stage can also be re-run on its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import prepare_data
import build_network
import analyze

FIGURES = ["fig_data_profile", "fig_similarity_and_threshold", "fig_threshold_sweep",
           "fig_network", "fig_structure", "fig_heatmap", "fig_community_profiles",
           "fig_items", "fig_issue_network", "fig_benchmarks"]


def main():
    for label, fn in [("1/4  preparing data", prepare_data.main),
                      ("2/4  building network", build_network.main),
                      ("3/4  analysing network", analyze.main)]:
        print(f"\n=== {label} " + "=" * max(4, 48 - len(label)))
        fn()

    # imported only now: it reads the files the stages above have just written
    print("\n=== 4/4  drawing figures " + "=" * 24)
    import visualize
    for name in FIGURES:
        getattr(visualize, name)()

    print("\nDone. Numbers in results/, charts in figures/.")


if __name__ == "__main__":
    main()
