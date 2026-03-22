# Parameter Golf Task

PACEvolve task for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf): train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100.

## Evolvable Surface

The entire training script (`src/train_gpt.py`) between `RegexTagCustomPruningAlgorithmStart` and `RegexTagCustomPruningAlgorithmEnd` is evolvable. This includes:

- Model architecture (layers, dimensions, attention, MLP, normalization)
- Optimizer (Muon, Adam, learning rates, schedules)
- Training loop (batch size, gradient accumulation, warmup/warmdown)
- Quantization/compression pipeline
- Tokenizer configuration (vocabulary size)
- Any other aspect of training

## Evaluation Guardrails

Candidates must preserve the protected evaluation semantics from `src/train_gpt_ref.py`.
That includes shard ordering, validation token loading, BPB computation, quantized round-trip evaluation, and the meaning of the final metric reporting lines.
PACEvolve now runs an additional model-side review step before compile/eval and gives failed candidates up to two regeneration attempts to avoid reward hacking.

## Setup

### Data

Download the FineWeb dataset with the 1024-token SentencePiece vocabulary:

```bash
cd /workspace/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This populates `data/datasets/fineweb10B_sp1024/` (80 train shards, ~8B tokens) and `data/tokenizers/`.

### Running

```bash
# Syntax check only
./eval/run.sh pgolf /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model 8 syntax

# Full training run (8 GPUs)
./eval/run.sh pgolf /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model 8 train

# Single-GPU test (slower, for debugging)
./eval/run.sh pgolf /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model 1 train
```

### Output

The training script outputs:

```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:Xms
Candidate val_bpb:X.XXXXXXXX artifact_bytes:X size_limit:16000000 size_ok:True train_time_ms:X
```
