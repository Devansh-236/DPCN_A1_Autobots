"""Step 1 - Read the raw survey export and turn it into a clean numeric opinion matrix.

Outputs
-------
results/opinion_matrix.csv   respondents x 60 items, values in {-2..2}, NaN = no answer
results/item_metadata.csv    item code, domain, full question text
results/prep_summary.json    every number quoted in the "Dataset Documentation" section
"""
import json
import re
import pandas as pd
import numpy as np

from config import RAW_CSV, RESULTS, LIKERT_MAP, MISSING_TOKENS, MIN_COMPLETENESS, DOMAINS


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def split_header(col: str):
    """'T01. Artificial Intelligence will ...' -> ('T01', 'Artificial Intelligence will ...')"""
    code, text = col.split(".", 1)
    return code.strip(), text.strip()


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = load_raw()

    id_col = raw.columns[0]
    item_cols = list(raw.columns[1:])
    codes, texts = zip(*[split_header(c) for c in item_cols])

    meta = pd.DataFrame({
        "item": codes,
        "domain_code": [c[0] for c in codes],
        "domain": [DOMAINS[c[0]] for c in codes],
        "question": texts,
    })

    df = raw[item_cols].copy()
    df.columns = list(codes)
    df.index = pd.Index([re.sub(r"\D", "", str(v)) for v in raw[id_col]], name="respondent")

    # --- audit the raw response vocabulary before mapping -------------------
    vocab = pd.Series(df.values.ravel()).fillna("").value_counts().to_dict()

    # --- map to numbers; everything not on the Likert scale becomes NaN -----
    def to_score(v):
        if pd.isna(v):
            return np.nan
        return LIKERT_MAP.get(str(v).strip(), np.nan)   # 'No Comments' falls through to NaN

    clean = df.map(to_score)

    n_items = clean.shape[1]
    completeness = clean.notna().sum(axis=1) / n_items
    keep = completeness >= MIN_COMPLETENESS
    dropped = completeness[~keep].sort_values()

    kept = clean.loc[keep]

    summary = {
        "raw_respondents": int(clean.shape[0]),
        "items": int(n_items),
        "raw_response_vocabulary": vocab,
        "no_comments_cells": int((df == "No Comments").sum().sum()),
        "blank_cells": int(df.isna().sum().sum()),
        "min_completeness": MIN_COMPLETENESS,
        "dropped_respondents": {str(k): round(float(v), 3) for k, v in dropped.items()},
        "n_dropped": int((~keep).sum()),
        "n_nodes": int(keep.sum()),
        "missing_cells_after_filter": int(kept.isna().sum().sum()),
        "missing_rate_after_filter": round(float(kept.isna().mean().mean()), 4),
        "fully_complete_respondents": int((kept.notna().all(axis=1)).sum()),
        "items_per_domain": {d: int((meta.domain_code == d).sum()) for d in DOMAINS},
        "response_share_after_filter": {
            k: round(float((kept == v).sum().sum()) / float(kept.notna().sum().sum()), 4)
            for k, v in LIKERT_MAP.items()
        },
        "grand_mean": round(float(np.nanmean(kept.values)), 4),
    }

    kept.to_csv(RESULTS / "opinion_matrix.csv")
    meta.to_csv(RESULTS / "item_metadata.csv", index=False)
    (RESULTS / "prep_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"nodes kept        : {summary['n_nodes']} / {summary['raw_respondents']}")
    print(f"dropped           : {summary['dropped_respondents']}")
    print(f"missing after cut : {summary['missing_cells_after_filter']} cells "
          f"({summary['missing_rate_after_filter']:.2%})")
    print(f"response share    : {summary['response_share_after_filter']}")
    print(f"grand mean        : {summary['grand_mean']}")


if __name__ == "__main__":
    main()
