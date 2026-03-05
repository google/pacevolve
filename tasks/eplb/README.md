# Expert Parallelism Load Balancer (EPLB)

This example demonstrates how to use OpenEvolve to optimize the Expert Parallelism Load Balancer (EPLB) algorithm.

## Setup

Install PyTorch:

```bash
uv pip install torch
```

Download the workload file from [Hugging Face](https://huggingface.co/datasets/abmfy/eplb-openevolve):

```bash
mkdir -p pacevolve/tasks/eplb/data
wget -O pacevolve/tasks/eplb/data/expert-load.json \
  https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json
```

### 3. Run with PACEvolve

From the project root:

```bash
bash run_eplb.sh
```

Ensure `IS_TRAINING=True` in the script for full training. Edit the configuration section in `run_eplb.sh` (SAVE_PATH, model, wandb, etc.) before running.

## Task structure

- `config/config_1.yaml` - PACE-format config
- `config/prompts.py` - Prompts and initial algorithm
- `src/eplb_1.py` - Initial program (editable block between PACE tags)
- `eval/evaluate_eplb.py` - CLI evaluator
- `eval/eval_utils.py` - recompile_library, evaluate_dataset, parse_eval_results
- `data/` - Place `expert-load.json` here

## Evaluation metrics

- **balancedness_score**: Load balance quality (higher is better)
- **speed_score**: Inverse of algorithm runtime (higher is better)
- **combined_score**: `(balancedness_score + speed_score) / 2` - primary metric
