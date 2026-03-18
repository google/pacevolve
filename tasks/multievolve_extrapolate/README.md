# MULTI-evolve Extrapolation Benchmark

This task implements the following from the MULTI-evolve paper as a public-data benchmark:

- train on singles + doubles
- test on unseen 3-10 mutants
- optimize Pearson R and top-5% precision

It is designed to be reproducible from the public benchmark data release referenced by the MULTI-evolve repo.


## Benchmark structure

The task uses a representative `lite` subset by default to keep repeated evolutionary evaluation tractable. The task also ships a `full` manifest covering all 12 benchmark assays from the MULTI-evolve summary.

### Default split

For each dataset:

- Train: wild type, singles, and doubles
- Test: 3-10 mutants
- Filter: keep only multimutants whose constituent mutations appear as singles in the same dataset

This follows the paper-aligned low-order-to-high-order extrapolation setup and the preprocessing logic in the public MULTI-evolve repo.

## Metrics

Per dataset:

- `pearson_r`
- `precision_top5`

Final benchmark score:

- `combined_score = 0.7 * mean_pearson_r + 0.3 * mean_precision_top5`

## Evolvable surface

Your editable code lives in [multievolve_extrapolate_1.py](/Users/minghao/PACE-RL/pacevolve/tasks/multievolve_extrapolate/src/multievolve_extrapolate_1.py).

The intended search surface includes:

- mutation featurization
- pairwise interaction modeling
- regularization and calibration
- sample weighting
- lightweight ensembling

## Public data setup

### Option 1: Download from Zenodo

```bash
python pacevolve/tasks/multievolve_extrapolate/data/download_public_data.py --benchmark-level lite
python pacevolve/tasks/multievolve_extrapolate/data/prepare_public_benchmark.py --benchmark-level lite
```

### Option 2: Copy from an existing local MULTI-evolve data folder

If you already downloaded the CSVs into `MULTI-evolve/data/benchmark/datasets`:

```bash
python pacevolve/tasks/multievolve_extrapolate/data/download_public_data.py \
  --benchmark-level lite \
  --source-dir /path/to/MULTI-evolve/data/benchmark/datasets
python pacevolve/tasks/multievolve_extrapolate/data/prepare_public_benchmark.py --benchmark-level lite
```

Prepared artifacts are written under:

- `pacevolve/tasks/multievolve_extrapolate/data/raw`
- `pacevolve/tasks/multievolve_extrapolate/data/prepared/lite`

## Running the evaluator

```bash
python pacevolve/tasks/multievolve_extrapolate/eval/evaluate_multievolve_extrapolate.py \
  --candidate_path pacevolve/tasks/multievolve_extrapolate/src/multievolve_extrapolate_1.py \
  --data_dir pacevolve/tasks/multievolve_extrapolate/data \
  --benchmark_level lite
```

For syntax-only validation:

```bash
python pacevolve/tasks/multievolve_extrapolate/eval/evaluate_multievolve_extrapolate.py \
  --candidate_path pacevolve/tasks/multievolve_extrapolate/src/multievolve_extrapolate_1.py \
  --data_dir pacevolve/tasks/multievolve_extrapolate/data \
  --benchmark_level lite \
  --syntax_only
```

## Relationship to the original MULTI-evolve repo

This task is based on the public benchmark design and split style from the local [MULTI-evolve](../../../../MULTI-evolve/README.md) clone, but the evaluator is intentionally lighter than their full training stack so that evolutionary search can run repeatedly without WandB sweeps or the full package environment.
