# Parameter Golf Triton Task

PACEvolve task for a Triton-focused Parameter Golf variant: keep the original 16MB artifact and 10-minute budget, but search only for opportunities where Triton kernels or equivalent kernel-level acceleration can lower BPB by fitting more real work into the same wallclock budget.

## Evolvable Surface

The entire training script (`src/train_gpt.py`) between `RegexTagCustomPruningAlgorithmStart` and `RegexTagCustomPruningAlgorithmEnd` is evolvable, but the prompt explicitly restricts the search agenda to:

- Triton kernels or Triton-backed fusion
- Kernel launch reduction
- Better memory layout / buffer reuse
- Faster quantization / evaluation kernels
- Other kernel-level throughput improvements that preserve the same benchmark semantics

## Evaluation Guardrails

Candidates must preserve the protected evaluation semantics from `src/train_gpt_ref.py`.
That includes shard ordering, validation token loading, BPB computation, quantized round-trip evaluation, and the meaning of the final metric reporting lines.
PACEvolve runs an additional model-side review step before compile/eval and gives failed candidates up to two regeneration attempts to avoid reward hacking.

## Setup

Use the same FineWeb and tokenizer assets as the original Parameter Golf task. The default config points to:

- `~/pacevolve/tasks/parameter_golf/data/datasets/fineweb10B_sp1024`
- `~/pacevolve/tasks/parameter_golf/data/tokenizers/fineweb_1024_bpe.model`
