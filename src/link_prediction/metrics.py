from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def rank_candidates(
    labels: Sequence[int],
    scores: Sequence[float],
    candidate_ids: Sequence[int],
) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "label": labels,
            "score": scores,
        }
    )

    if dataframe.empty:
        raise ValueError("Candidate table cannot be empty.")

    if dataframe["candidate_id"].duplicated().any():
        raise ValueError("candidate_id values must be unique.")

    if not np.isfinite(
        dataframe["score"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Scores must be finite.")

    labels_set = set(
        dataframe["label"].astype(int).unique()
    )

    if not labels_set <= {0, 1}:
        raise ValueError(
            "Labels must be binary values 0 or 1."
        )

    return dataframe.sort_values(
        ["score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def ndcg_at_k(
    ranked_labels: Sequence[int],
    k: int,
) -> float:
    labels = np.asarray(
        ranked_labels,
        dtype=int,
    )

    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    k = min(
        k,
        len(labels),
    )

    top_labels = labels[:k]

    discounts = (
        1.0
        / np.log2(
            np.arange(
                2,
                k + 2,
            )
        )
    )

    dcg = float(
        np.sum(
            top_labels
            * discounts
        )
    )

    ideal_positive_count = min(
        int(labels.sum()),
        k,
    )

    if ideal_positive_count == 0:
        return 0.0

    idcg = float(
        np.sum(
            discounts[
                :ideal_positive_count
            ]
        )
    )

    return dcg / idcg


def tie_diagnostics(
    ranked: pd.DataFrame,
    cutoff: int,
) -> dict[str, float | int | bool]:
    score_counts = (
        ranked.groupby(
            "score",
            sort=False,
        )
        .size()
    )

    tied_score_counts = score_counts[
        score_counts > 1
    ]

    cutoff_score = float(
        ranked.iloc[
            cutoff - 1
        ][
            "score"
        ]
    )

    cutoff_tie = ranked[
        ranked[
            "score"
        ]
        == cutoff_score
    ]

    cutoff_tie_positions = (
        cutoff_tie.index.to_numpy(
            dtype=int
        )
    )

    cutoff_tie_start = int(
        cutoff_tie_positions.min()
    )

    cutoff_tie_end = int(
        cutoff_tie_positions.max()
    )

    cutoff_slots_in_tie = (
        cutoff
        - cutoff_tie_start
    )

    return {
        "distinct_score_count":
            len(score_counts),
        "tie_group_count":
            len(
                    tied_score_counts
                ),
        "tied_candidate_count":
            int(
                tied_score_counts.sum()
            ),
        "tied_candidate_ratio":
            float(
                tied_score_counts.sum()
                / len(ranked)
            ),
        "largest_tie_group":
            int(
                tied_score_counts.max()
                if not tied_score_counts.empty
                else 1
            ),
        "cutoff_score":
            cutoff_score,
        "cutoff_tie_size":
            len(cutoff_tie),
        "cutoff_tie_positive_count":
            int(
                cutoff_tie[
                    "label"
                ].sum()
            ),
        "cutoff_tie_negative_count":
            int(
                len(cutoff_tie)
                - cutoff_tie[
                    "label"
                ].sum()
            ),
        "cutoff_slots_in_tie":
            int(
                cutoff_slots_in_tie
            ),
        "cutoff_tie_crosses_boundary":
            bool(
                cutoff_tie_start
                < cutoff
                <= cutoff_tie_end
            ),
    }


def evaluate_ranking(
    labels: Sequence[int],
    scores: Sequence[float],
    candidate_ids: Sequence[int],
    cutoff: int | None = None,
) -> dict[str, float | int | bool]:
    ranked = rank_candidates(
        labels=labels,
        scores=scores,
        candidate_ids=candidate_ids,
    )

    ranked_labels = ranked[
        "label"
    ].to_numpy(
        dtype=int
    )

    raw_labels = np.asarray(
        labels,
        dtype=int,
    )

    raw_scores = np.asarray(
        scores,
        dtype=float,
    )

    positive_count = int(
        raw_labels.sum()
    )

    negative_count = int(
        len(raw_labels)
        - positive_count
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        raise ValueError(
            "ROC-AUC requires both positive "
            "and negative candidates."
        )

    if cutoff is None:
        cutoff = positive_count

    if (
        cutoff < 1
        or cutoff > len(ranked_labels)
    ):
        raise ValueError(
            "cutoff must be between 1 "
            "and the number of candidates."
        )

    diagnostics = tie_diagnostics(
        ranked=ranked,
        cutoff=cutoff,
    )

    top_labels = ranked_labels[
        :cutoff
    ]

    true_positives = int(
        top_labels.sum()
    )

    precision = (
        true_positives
        / cutoff
    )

    recall = (
        true_positives
        / positive_count
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall > 0
        else 0.0
    )

    return {
        "average_precision":
            float(
                average_precision_score(
                    raw_labels,
                    raw_scores,
                )
            ),
        "roc_auc":
            float(
                roc_auc_score(
                    raw_labels,
                    raw_scores,
                )
            ),
        "precision":
            float(precision),
        "recall":
            float(recall),
        "f1":
            float(f1),
        "ndcg":
            float(
                ndcg_at_k(
                    ranked_labels,
                    cutoff,
                )
            ),
        "cutoff":
            int(cutoff),
        "positive_count":
            positive_count,
        "negative_count":
            negative_count,
        "candidate_count":
            len(
                ranked_labels
            ),
        **diagnostics,
    }