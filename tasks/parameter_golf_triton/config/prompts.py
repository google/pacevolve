# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tasks.parameter_golf.config import prompts as base

PARAMETER_GOLF_TRITON = """
This training script is a baseline for a Triton-focused Parameter Golf variant.
The task is identical to Parameter Golf except the research agenda is constrained:
improve the score by finding places where custom Triton kernels or kernel-level
performance engineering can make training/evaluation faster, so the run can fit
more real optimization steps into the same 10-minute budget and achieve lower BPB.

The baseline still targets ~1.2244 val_bpb with a ~15.9MB compressed artifact.
"""

CODING_REQ = """
While completing your task, you MUST:
- Enclose your code in triple backticks to properly format the code in Markdown.
- Your code will replace the content between RegexTagCustomPruningAlgorithmStart and RegexTagCustomPruningAlgorithmEnd in train_gpt.py.
- Your code MUST be a complete, self-contained training script that can be run via `torchrun --standalone --nproc_per_node=N train_gpt.py`.
- Your code MUST print a final summary line in exactly this format:
  `Candidate val_bpb:<float> artifact_bytes:<int> size_limit:<int> size_ok:<bool> train_time_ms:<int>`
- The total artifact size (code bytes + int8+zlib compressed model bytes) MUST be <= 16,000,000 bytes (16 MB).
- Training MUST complete within 600 seconds (10 minutes) of wallclock time.
- You MUST preserve the int8 quantization + zlib compression pipeline and the BPB evaluation logic.
- You MUST preserve the protected evaluation semantics from `train_gpt_ref.py`.
- In particular, do NOT change the train/validation shard discovery or ordering, the `TokenStream` / `load_validation_tokens` behavior, the BPB computation path in `eval_val` / `eval_val_sliding`, the quantized round-trip evaluation semantics, or the meaning/format of the final metric reporting lines.
- Any improvement must come from genuine kernel acceleration or equivalent low-level performance engineering, not from shortcutting evaluation, reordering or subsampling data, reusing stale metrics, or altering the final metric reporting logic.
- Restrict your changes to opportunities where Triton kernels, Triton-backed fusion, launch reduction, memory-layout improvements, or equivalent kernel-level execution changes can make the same training objective run faster.
- Do NOT introduce architecture searches, tokenizer changes, objective changes, or unrelated hyperparameter-only ideas unless they are strictly required to expose or exploit a faster kernel while preserving the same task semantics.
- When possible, use the saved wallclock budget to run more legitimate optimization steps within 600 seconds.
- You MUST NOT access any external data, downloads, or network during evaluation. The artifact must be self-contained.
- Keep the script under 1500 lines.
"""

CANDIDATE_VALIDATION_REFERENCE_FILE = base.CANDIDATE_VALIDATION_REFERENCE_FILE
CANDIDATE_VALIDATION_MAX_RETRIES = base.CANDIDATE_VALIDATION_MAX_RETRIES

RJCH_DOCS = f"""
### Codebase documentation
Your code trains a language model for the Parameter Golf challenge.
The challenge objective: minimize bits-per-byte (BPB) on the FineWeb validation set,
subject to a 16MB artifact size limit and 10-minute training time cap on 8xH100.

{PARAMETER_GOLF_TRITON}
"""

BACKGROUND = """
### Background on Triton-Focused Parameter Golf

This variant keeps the exact same benchmark and scoring pipeline as Parameter Golf,
but narrows the research space to throughput improvements driven by kernels.

You should search for places where the current implementation is leaving GPU
performance on the table:
- fused Triton kernels for RMSNorm, residual + norm + projection, relu^2 MLP blocks
- attention-side kernels or epilogues that reduce memory traffic or launch overhead
- optimizer/update kernels, especially when many small tensor ops are launch-bound
- quantization or dequantization kernels that shorten post-training evaluation time
- validation-side kernels or batching changes that preserve semantics while speeding up scoring
- kernel scheduling, autotuning, memory layout, or buffer reuse that preserves math

The target outcome is not "different math"; it is "same benchmark semantics, more real work in 10 minutes".
"""

KNOWLEDGE_BASE = f"""
{base.KNOWLEDGE_BASE}

# Triton-specific guidance
- Prefer kernel ideas that preserve the same mathematical result but reduce launch count, memory traffic, synchronization, or Python overhead.
- Good targets include RMSNorm, RoPE application, relu^2 MLP paths, residual mixing, optimizer transforms, de/quantization, and validation kernels.
- The most valuable improvements are ones that create enough wallclock headroom to increase the number of real training steps inside the fixed 600-second budget.
- Do not claim wins from altered validation windows, different data order, skipped quantized round-trip evaluation, or modified summary reporting.
"""


def construct_mutation_prompt(sota_algorithm, ablation_list):
    ablation_descriptions = "\n".join(ablation_list)
    prompt = f"""
We are conducting an evolutionary optimization process for a Triton-focused Parameter Golf challenge.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

{KNOWLEDGE_BASE}
{ablation_descriptions}

## Your Task

You will improve BPB only by discovering kernel-level acceleration opportunities.
Your proposed candidate must preserve the same evaluation semantics while making the
training/evaluation path faster, ideally so the run can complete more legitimate
optimization steps within the same 10-minute budget.

Allowed directions:
- Triton kernels or Triton-backed fusion for existing math
- Faster memory layouts, buffer reuse, launch reduction, or kernel scheduling
- Kernel-level speedups in training, validation, quantization, or round-trip evaluation

Disallowed directions unless absolutely required to expose a faster kernel:
- Broad architecture redesigns
- Objective or loss changes
- Data ordering changes
- Evaluation shortcuts or reporting changes
- Pure hyperparameter fishing unrelated to throughput

You must consider past experiment results when choosing where a kernel speedup is most likely to matter.

{CODING_REQ}

Please follow these steps:

1. Explain the current bottlenecks and likely kernel opportunities.
2. Brainstorm several kernel-focused ideas. Provide reasoning for each idea.
3. Select the single most promising idea and explain why it should improve throughput enough to lower BPB by enabling more real work in 10 minutes.
4. Implement the candidate in code.
"""
    return prompt


def construct_idea_gen_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for a Triton-focused Parameter Golf challenge.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

{KNOWLEDGE_BASE}

### Idea Repo
Idea repos contain ideas that we have generated so far, and experiments we have run to test these hypotheses.

{idea_repo}

## Your Task

Propose 3 different kernel-focused research options. Each option must identify a concrete
place where Triton or low-level kernel engineering could make the same benchmark semantics run faster.

For each option, explain:
* The exact bottleneck or kernel opportunity.
* Why the speedup should translate into more legitimate training/eval work inside 600 seconds.
* Why the change should preserve the protected evaluation semantics.
* Why the experiment is worth spending one full 8xH100 run on.

Go through the idea and experiment history carefully, and DO NOT re-propose a kernel idea that has already been well tested.

Use the following format:
** Idea 1 **
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>

** Idea 2 **
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>

...

** Idea N **
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>
"""
    return prompt


def construct_idea_select_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for a Triton-focused Parameter Golf challenge.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

{KNOWLEDGE_BASE}

### Idea Repo
Idea repos contain ideas that we have generated so far, and experiments we have run to test these hypotheses.

{idea_repo}

## Your Task

Select one concrete kernel-focused experiment from the idea repo to implement next.
Pick the option that has the best chance of lowering BPB by creating enough real throughput
headroom to fit more legitimate work in the 10-minute budget.

Do NOT select ideas that depend on changing evaluation semantics, data ordering, or final reporting.
Do NOT propose an experiment that has already been tested thoroughly.

Use the following format:

Idea ID: <Idea ID>
Experiment description: <Provide a concrete but concise kernel-focused experiment description. DO NOT HALLUCINATE IDEA ID.>

Once your reasoning is finished, write code using the same guardrails below:

{CODING_REQ}
"""
    return prompt


SUMMARIZE_EVAL_PROMPT = base.SUMMARIZE_EVAL_PROMPT
EVAL_DESCRIPTION_PROMPT = base.EVAL_DESCRIPTION_PROMPT
HPARAM_PROMPT = """
## Hyperparameter tuning
Hyperparameter-only changes are out of scope for this task unless they are directly justified by a kernel throughput change.

If a tiny schedule tweak is strictly necessary to exploit a faster kernel, explain why. Otherwise respond "No."
"""

HPARAM_IMPLEMENT_PROMPT = f"""
### Hyperparameter implementation
Only write code if the hyperparameter change is inseparable from a kernel-speedup change.

{CODING_REQ}
"""

UPDATE_BASELINE_PROMPT = f"""
Should we update the baseline algorithm? Please answer yes or no then explain your reasoning. If the answer is yes, respond with a code block containing the candidate that we should use as the new baseline algorithm. Prefer updates that demonstrate real kernel-driven throughput gains while preserving evaluation semantics. If no, simply respond "No."

{CODING_REQ}
"""


def construct_candidate_validation_prompt(candidate_code: str, reference_code: str) -> str:
    return base.construct_candidate_validation_prompt(candidate_code, reference_code)
