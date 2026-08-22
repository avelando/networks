import networkx as nx
import pandas as pd

DEGREE_BASED_METHODS = (
    "pa",
)


def score_degree_based_candidates(
    graph: nx.Graph,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if graph.is_directed():
        raise ValueError(
            "Degree-based methods "
            "require an undirected graph."
        )

    required_columns = {
        "source",
        "target",
    }

    missing_columns = (
        required_columns
        - set(candidates.columns)
    )

    if missing_columns:
        raise ValueError(
            "Candidate table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    degrees = dict(
        graph.degree()
    )

    scores = []

    for (
        source,
        target,
    ) in candidates[
        [
            "source",
            "target",
        ]
    ].itertuples(
        index=False,
        name=None,
    ):
        if (
            source not in degrees
            or target not in degrees
        ):
            raise ValueError(
                "Candidate pair contains "
                "node absent from training graph: "
                f"({source}, {target})"
            )

        scores.append(
            float(
                degrees[source]
                * degrees[target]
            )
        )

    return pd.DataFrame(
        {
            "pa": scores,
        }
    )