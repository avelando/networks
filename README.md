# Link Prediction Benchmark

This repository contains the experimental pipeline used to evaluate local and quasi-local topological methods for link prediction in complex networks.

## Experimental scope

The benchmark focuses on local and quasi-local topological link prediction methods. The method taxonomy and part of the theoretical organization follow prior literature, including the survey by Martínez, Berzal, and Cubero.

The revised benchmark preserves all five networks used in the original KDMiLe 2026 study and extends the evaluation with three additional networks.

The final benchmark contains eight real-world networks distributed evenly across four domains:

- social;
- scientific collaboration;
- communication;
- infrastructure.

Two networks are evaluated for each domain.

The additional networks are selected from the Network Data Repository and are included to improve domain coverage without changing the original experimental scope.

## Experimental protocol

The experiments use five-fold cross-validation while preserving training-graph connectivity through a fixed spanning tree. The primary evaluation uses balanced negative sampling, with additional robustness experiments using larger negative sampling ratios.

Average Precision is the primary evaluation metric.