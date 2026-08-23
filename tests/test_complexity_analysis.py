from link_prediction.complexity_analysis import (
    build_method_complexity_table,
    run_complexity_analysis,
)


def test_build_method_complexity_table():
    table = (
        build_method_complexity_table()
    )

    assert len(table) == 27

    assert table[
        "method_id"
    ].is_unique

    assert not table[
        "complexity"
    ].isna().any()

    assert not table[
        "analysis_family"
    ].isna().any()

    ora_cni = table[
        table[
            "method_id"
        ]
        == "ora_cni"
    ].iloc[0]

    assert ora_cni[
        "complexity"
    ] == "O(v k^6)"

    pfp = table[
        table[
            "method_id"
        ]
        == "pfp"
    ].iloc[0]

    assert pfp[
        "complexity"
    ] == "O(v l k^l)"


def test_run_complexity_analysis_writes_output(
    tmp_path,
):
    table = run_complexity_analysis(
        output_dir=
            tmp_path,
    )

    path = (
        tmp_path
        / "revision_method_complexity.csv"
    )

    assert path.exists()

    assert len(table) == 27

    assert not list(
        tmp_path.glob(
            "*.part"
        )
    )