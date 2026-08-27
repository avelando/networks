# Comparative Evaluation of Local and Quasi-Local Topological Methods for Link Prediction

Experimental benchmark comparing **19 local and quasi-local topological methods** for link
prediction across **eight real-world complex networks** from four domains (social, scientific
collaboration, e-mail communication, and electrical infrastructure).

This repository contains the source code and experimental materials for the paper:

> **Comparative Evaluation of Local and Quasi-Local Topological Methods for Link Prediction in Complex Networks**
> Avelar Rodrigues de Sousa, Maykon Willyam de Sousa Ferreira
> Institute of Mathematics and Computer Science (ICMC), University of São Paulo (USP)
> *Symposium on Knowledge Discovery, Mining and Learning (KDMiLe), 2026.*

The main contribution is not a new index, but a uniform empirical comparison: all methods are
evaluated under a single protocol (five-fold cross-validation, spanning-tree connectivity
preservation, and balanced negative sampling) across every network, so that differences in
performance reflect the methods and network structure rather than heterogeneous experimental
setups.

## Key findings

- **SRW-l3** (Superposed Random Walk, 3 steps) obtained the best overall mean Average Precision (**0.9220**).
- Followed by **LPI-beta-0.001** (0.9180) and **RA-CNI** (0.9086).
- Quasi-local methods based on short paths and walks led the ranking, though simple local
  indices (RA, AA, CN) remained competitive.
- A Friedman test (p < 0.001) with Wilcoxon signed-rank pairwise comparisons and Holm correction
  found SRW-l3 significantly superior to PFP-l3 and RA-CNI, but its advantage over LPI-beta-0.001
  was **not** statistically significant.
- Performance was highest on `ego-Facebook` and the scientific collaboration networks, and lowest
  on the electrical infrastructure networks, where low degree and clustering offer weaker
  topological signal.

## Methods evaluated

| Family | Methods |
|---|---|
| Local similarity | CN, AA, RA, JA, SA, SO, HPI, HDI, LLHN |
| Degree-based local | PA |
| Local Bayesian | LNB-CN, LNB-AA, LNB-RA |
| Enhanced local | RA-CNI, IA1, IA2, CAR-CN, CAR-AA, CAR-RA, FSW, LIT-i2 |
| Quasi-local (paths) | LPI (beta = 0.001), ORA-CNI, FL-l3 |
| Quasi-local (walks) | LRW-l3, SRW-l3, PFP-l3 |

The exact set of methods, families, hyperparameters, and complexity classes is declared in
[`configs/methods.yaml`](configs/methods.yaml).

## Networks

| Network | Domain | Source |
|---|---|---|
| ego-Facebook | Social | SNAP |
| socfb-Middlebury45 | Social | Network Data Repository |
| ca-GrQc | Scientific collaboration | SNAP |
| ca-HepTh | Scientific collaboration | SNAP |
| email-Eu-core | Communication | SNAP |
| email-univ | Communication | Network Data Repository |
| power-1138-bus | Infrastructure | Network Data Repository |
| power-grid | Infrastructure | Netzschleuder |

Download URLs, parsers, and preprocessing options for each network are declared in
[`configs/networks.yaml`](configs/networks.yaml). All networks are standardized as simple
undirected graphs, with self-loops removed and the largest connected component selected.

## Repository structure

```
.
├── configs/                # YAML configuration (experiment, networks, methods)
│   ├── experiment.yaml     # Seed, folds, metrics, negative sampling, execution limits
│   ├── networks.yaml       # Network sources, parsers, preprocessing
│   └── methods.yaml        # Methods, families, hyperparameters, complexity classes
├── src/link_prediction/    # Core library
│   ├── datasets.py         # Download and parse network files
│   ├── preprocessing.py    # Graph standardization
│   ├── profiling.py        # Structural profiling of networks
│   ├── folds.py            # Connectivity-preserving fold construction
│   ├── sampling.py         # Balanced negative sampling
│   ├── evaluation.py       # Family benchmark runner
│   ├── metrics.py          # AP, ROC-AUC, Precision/Recall/F1, NDCG, tie diagnostics
│   ├── methods/            # Scoring functions per method family
│   ├── sensitivity.py      # Parameter sensitivity (beta, walk steps)
│   ├── robustness.py       # Negative-sampling ratio robustness
│   ├── statistical_analysis.py  # Friedman + Wilcoxon/Holm tests
│   ├── (structural / tie / runtime / complexity analysis modules)
│   └── reproducibility.py  # Environment and git metadata capture
├── notebooks/              # Ordered pipeline (01 -> 15)
├── tests/                  # 25 pytest test modules
├── data/                   # raw / processed / folds (downloaded, git-ignored)
└── results/                # summaries, reports, figures
```

## Installation

Requires **Python >= 3.10** (developed on 3.14). Using a virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package with development extras (JupyterLab, pytest, ruff)
pip install -e ".[dev]"
```

For an exactly pinned environment, use the lockfile instead:

```bash
pip install -r requirements-lock.txt
```

## Running the experiments

The notebooks in [`notebooks/`](notebooks/) are numbered and meant to be run in order. Each one
is a thin wrapper over the library in `src/link_prediction/`.

| Notebook | Purpose |
|---|---|
| `01_download_and_profile_networks` | Download networks and compute structural properties |
| `02_build_evaluation_folds` | Build connectivity-preserving 5-fold splits with balanced negatives |
| `03_local_similarity_methods` | Benchmark local similarity indices |
| `04_degree_based_methods` | Benchmark degree-based (PA) |
| `05_enhanced_local_methods` | Benchmark enhanced local indices |
| `06_local_bayesian_methods` | Benchmark local Bayesian indices |
| `07_quasi_local_path_methods` | Benchmark path-based quasi-local methods |
| `08_quasi_local_walk_methods` | Benchmark walk-based quasi-local methods |
| `09_parameter_sensitivity` | Sensitivity to beta and walk steps |
| `10_negative_sampling_robustness` | Robustness across 1:1, 1:5, 1:10 negative ratios |
| `11_statistical_tests` | Friedman + Wilcoxon (Holm) significance tests |
| `12_structural_analysis` | Relate structural properties to task difficulty |
| `13_tie_analysis` | Score-tie diagnostics at the cutoff |
| `14_runtime_analysis` | Empirical runtime vs. theoretical complexity |
| `15_generate_results` | Consolidate tables and figures into `results/` |

Start JupyterLab and run them top to bottom:

```bash
jupyter lab
```

Notebooks 01 and 02 populate `data/`; the method notebooks (03-08) write per-fold metrics and
network summaries; the analysis notebooks (09-15) produce the tables and figures under
`results/`.

## Experimental protocol

The full configuration lives in [`configs/experiment.yaml`](configs/experiment.yaml). Key settings:

- **Random seed:** 42
- **Cross-validation:** 5 folds
- **Connectivity:** a spanning tree is kept fixed in the training graph (seeded randomized
  Kruskal); only the remaining edges are distributed across test folds, so training graphs never
  disconnect. Spanning-tree edges are never used as positive test instances.
- **Negative sampling:** balanced 1:1 for the main experiment; 1:1, 1:5, and 1:10 for robustness.
- **Primary metric:** Average Precision (link prediction is treated as a ranking task).
- **Additional metrics:** ROC-AUC, Precision, Recall, F1, and NDCG, evaluated at a cutoff of
  K = |E+| (number of removed positive edges). Under the balanced setting Precision, Recall, and
  F1 coincide at this cutoff.
- **Ties:** resolved deterministically using the fixed seed and a stable sort.

Fixed hyperparameters (following the literature): beta = 0.001 for LPI, and l = 3 steps for the
walk-based methods (LRW, SRW, PFP). Broader sweeps are provided in the parameter-sensitivity
notebook.

## Tests

```bash
pytest
```

The suite (25 modules under `tests/`) covers dataset parsing, preprocessing, fold construction,
sampling, metrics, each method family, the statistical and structural analyses, and
reproducibility.

## Reproducibility

Randomness is controlled by a single fixed seed (42). Fold construction, negative sampling, and
tie-breaking are all deterministic. `reproducibility.py` captures package versions and the git
commit/dirty state alongside results so that runs can be traced to a specific environment. For
byte-for-byte reproduction of the reported environment, install from `requirements-lock.txt`.

## Citation

If you use this code or the accompanying materials, please cite:

```bibtex
@inproceedings{sousa2026linkprediction,
  title     = {Comparative Evaluation of Local and Quasi-Local Topological Methods
               for Link Prediction in Complex Networks},
  author    = {Sousa, Avelar Rodrigues de and Ferreira, Maykon Willyam de Sousa},
  booktitle = {Symposium on Knowledge Discovery, Mining and Learning (KDMiLe)},
  year      = {2026}
}
```

## Acknowledgments

Developed during the course **SME5924 (Dynamical Processes in Complex Networks)**, taught by
Prof. Francisco Aparecido Rodrigues (ICMC/USP).

## License

Released under the MIT License. See [LICENSE](LICENSE).
