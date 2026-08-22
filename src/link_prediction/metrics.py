from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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


def average_precision_from_ranking(
    ranked_labels: Sequence[int],
) -> float:
    labels = np.asarray(
        ranked_labels,
        dtype=int,
    )

    positive_count = int(
        labels.sum()
    )

    if positive_count == 0:
        raise ValueError(
            "Average Precision requires "
            "at least one positive candidate."
        )

    cumulative_positives = np.cumsum(
        labels
    )

    ranks = np.arange(
        1,
        len(labels) + 1,
    )

    precisions = (
        cumulative_positives / ranks
    )

    return float(
        (precisions * labels).sum()
        / positive_count
    )


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


def evaluate_ranking(
    labels: Sequence[int],
    scores: Sequence[float],
    candidate_ids: Sequence[int],
    cutoff: int | None = None,
) -> dict[str, float | int]:
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
            average_precision_from_ranking(
                ranked_labels
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
    }