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

PARAMETER_GOLF = """
This training script is a baseline for the Parameter Golf challenge.
It trains a small GPT-style language model with the following characteristics:
- 9 transformer blocks at width 512
- 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
- vocab size 1024, sequence length 1024, tied embeddings
- Muon optimizer for matrix params, Adam for embeddings/scalars
- RMSNorm, RoPE, relu^2 MLP, U-net skip connections
- Post-training int8 quantization + zlib compression
- Tokenizer-agnostic BPB evaluation via SentencePiece LUTs

The baseline achieves ~1.2244 val_bpb with a ~15.9MB compressed artifact.
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
- You MUST preserve the int8 quantization + zlib compression pipeline and the BPB evaluation logic. Changing how BPB is computed invalidates the result.
- You MUST preserve the protected evaluation semantics from `train_gpt_ref.py`.
- In particular, do NOT change the train/validation shard discovery or ordering, the `TokenStream` / `load_validation_tokens` behavior, the BPB computation path in `eval_val` / `eval_val_sliding`, the quantized round-trip evaluation semantics, or the meaning/format of the final metric reporting lines.
- Any improvement must come from a genuinely better training run, not from shortcutting evaluation, reordering or subsampling data, reusing stale metrics, or altering the final metric reporting logic.
- You MUST NOT access any external data, downloads, or network during evaluation. The artifact must be self-contained.
- You may freely change the model architecture, hyperparameters, optimizer, training schedule, tokenizer handling, quantization details, and any other aspect of the training code.
- Keep the script under 1500 lines.
"""

CANDIDATE_VALIDATION_REFERENCE_FILE = "train_gpt_ref.py"
CANDIDATE_VALIDATION_MAX_RETRIES = 2

RJCH_DOCS = f"""
### Codebase documentation
Your code trains a language model for the Parameter Golf challenge.
The challenge objective: minimize bits-per-byte (BPB) on the FineWeb validation set,
subject to a 16MB artifact size limit and 10-minute training time cap on 8xH100.

{PARAMETER_GOLF}
"""

TASK_INTRO = """
You are an expert researcher specializing in efficient language model training and compression. We are participating in the Parameter Golf challenge, where the goal is to train the best possible language model under extreme parameter constraints.

**Objective**: Minimize val_bpb (bits per byte) on the FineWeb validation set.
**Constraints**:
- Total artifact (code + int8+zlib compressed model) must fit in 16MB (16,000,000 bytes)
- Training must complete within 10 minutes on 8xH100 GPUs
- No external data or network access during evaluation

This is fundamentally an L(N) optimization problem: minimize loss given a fixed parameter budget, unconstrained by data, compute steps, or architecture. The challenge is to squeeze maximum compression quality out of a tiny model.
"""


BACKGROUND = """
### Background on Parameter Golf

Parameter Golf is a challenge to train the best language model that fits in a 16MB artifact. Unlike standard LLM training where more parameters generally means better performance, here the budget is severely constrained. This requires creative approaches across multiple dimensions:

**Architecture**: The model must be small enough to compress into 16MB. Standard transformer blocks with 512 dimensions and 9 layers give a baseline, but alternatives like depth recurrence (sharing weights across layers), aggressive parameter tying, low-rank factorizations, or novel architectures might improve efficiency.

**Compression**: The model is quantized to int8 and zlib-compressed for the submission. Quantization-aware training (QAT), mixed-precision strategies, structured sparsity, or even sub-byte quantization could help fit more effective parameters into the budget.

**Training efficiency**: With only 10 minutes on 8xH100, every training step matters. The Muon optimizer with Newton-Schulz orthogonalization is already used for matrix params, but learning rate schedules, batch sizes, gradient accumulation, and warmup/warmdown strategies all matter.

**Tokenization**: The BPB metric is tokenizer-agnostic: BPB = bits_per_token * tokens_per_byte. A well-chosen vocabulary size and tokenizer can affect both model size (via embedding dimensions) and compression quality.

**Evaluation**: BPB (bits per byte) measures how well the model compresses the validation text, independent of tokenizer choice. Lower BPB means better compression. The final score uses the int8+zlib round-tripped model weights, not the training-time weights.

### Key Metrics
- **val_bpb**: The primary metric. Lower is better. Baseline achieves ~1.2244.
- **artifact_bytes**: Must be <= 16,000,000. Baseline uses ~15.9MB.
- **training_time**: Must be <= 600 seconds on 8xH100.
"""

KNOWLEDGE_BASE = """
# Old knowledge base: The following are lessons from a different setup, they may or may not help you here, worth a trying if you haven't. 
NOTE THAT THESE ARE NOT LESSONS FROM THE CURRENT SETUP SO TAKE IT WITH A GRAINT OF THOUGHT:

- U-net style skip connections (encoder-decoder split with learned skip weights) improve performance.
- Per-block learned residual mixing (resid_mix) and per-block attention/MLP scaling improve training stability.
- RoPE positional embeddings are applied to the full head dimension.
- Logit softcapping (tanh(logits/cap)*cap) stabilizes training.
- int8 quantization with per-row scales for 2D tensors and per-tensor scales for vectors preserves most model quality.
- Depth recurrence (sharing weights across layers) could dramatically reduce parameter count while maintaining expressiveness.
- Test-time compute (iterating through layers multiple times) is an unexplored direction that could improve quality without adding parameters.

# Knowledge base
- The baseline uses a 9-layer, 512-dim transformer with GQA (8 heads, 4 KV heads) and a tiny vocab (1024 tokens with SentencePiece BPE). Tied embeddings save parameters.
- Muon optimizer (Newton-Schulz orthogonalization) is used for matrix parameters and significantly improves over Adam alone.
- relu^2 activation in the MLP is used instead of GELU/SwiGLU.
- Low-rank factorizations of weight matrices can reduce parameter count with minimal quality loss.
- i'm trying to figure out why relu squared is best and how to beat it. observed: squaring massively helps relu, barely helps silu, and destroys gated functions. squaring benefit depends on the base function (what you're squaring)
- It's also not about zeroing negative values. `softplus²` has no zeros and outperforms `clamp(silu, 0)²` which does have zeros. `leaky_relu(0.01)²` (tiny leak, near-zero negatives) matches the clamped variants. The gap between relu² and the rest is mostly about the positive-side shape, not the zero/nonzero boundary.
- on a short run, this is what I got by now (longer runs in progress to confirm): activations that preserve negative values (e.g. selu², leaky_relu(0.5)², abs²) outperform relu² baseline, which suppresses negatives. this suggests a pattern: less suppression of negative signals → better performance. current best result: abs² (1.4712), but multi-seed validation is still pending. key experiment: whether abs² remains best across seeds; if yes, it supports the idea that no activation (just squaring) is optimal.
- squaring is the dominant mechanism, most of the performance gain comes from applying a square (·²), not from the specific activation used before it. relu → relu²: large improvement (≈ -0.049 BPB in early runs).
Multiple squared variants cluster tightly at 2000 steps: leaky(0.5)²: 1.3218 abs²: 1.3238 relu²: 1.3264 spread ≈ 0.0046 total current evidence is strong for “squaring matters”, weak for why it matters. i'll post more research
- the pre-squaring activation matters less than expected simpler functions before squaring perform better; complex nonlinearities degrade the benefit. abs² (no preprocessing) ≥ elu² ≥ softplus² ≥ clamped variants. silu² and gelu² barely improve over base versions. Gated activations (e.g., SwiGLU²) diverge or worsen significantly. more research necessary: confirmed up to 2000 out of 13k steps, gaps are shrinking over time ordering could change at longer horizons evidence supports “simplicity helps,” but magnitude is modest.
- hard zeros are not beneficial (and may be slightly harmful) Conclusion: Zeroing negative values (ReLU behavior) is not a key advantage and likely slightly reduces performance. softplus² (no zeros) > clamp(silu, 0)² (has zeros) leaky_relu(0.5)² consistently outperforms relu² abs² (no suppression at all) performs among the best
- to find better activation function inside MLP (for OpenAI's challenge) that beats the current baseline relu^2, it must:
(1) Gradient should scale with activation size
Neurons that output larger values should receive proportionally larger gradients, so they update more.
(2) Do not squash the output range
The activation should preserve differences in magnitude instead of compressing values into a narrow interval (like sigmoid). Large inputs should remain large, small inputs should remain small, and relative differences should stay visible. This avoids losing information.
(3) Retain information from negative inputs
Negative values should not be completely discarded. The activation should allow some negative signal to pass through or be transformed, rather than zeroing it out.
leaky(0.5)² satisfies all three, it performs better but within noise, so i'm testing more
i'm doing 500 step experiments (eliminate obviously bad) -> 2000 to 4000 (eliminate bad ones) -> 8000 to full 13,780 step training (my baseline, similar to OpenAI's), but i can use 1 GPU
> 500 steps is reliable for screening out clearly bad ideas (relu³ diverging, squared gating blowing up - effects 10-40x noise). 
It also reliably identifies that squaring helps (consistent across all base functions). 
However, 500 steps is NOT reliable for ranking within the squared family - abs² leads at 500 but ties relu² by step 5000.
Keep 500-step experiments for elimination rounds, but never trust fine-grained rankings from them. Any activation within ~0.01 BPB at 500 steps needs a 2000+ step run before drawing conclusions.
"""

def construct_mutation_prompt(sota_algorithm, ablation_list):
    ablation_descriptions = "\n".join(ablation_list)
    prompt = f"""
We are conducting an evolutionary optimization process for the Parameter Golf challenge.

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

You will make reasonable changes to improve the model's BPB (bits per byte) on the FineWeb validation set. Remember the hard constraints:
1. Total artifact (code + compressed model) must be <= 16MB (16,000,000 bytes)
2. Training must complete within 10 minutes (600 seconds wallclock) on 8xH100
3. The BPB evaluation and int8+zlib compression pipeline must remain correct

Try to strike a balance between safe, incremental improvements ("exploit-heavy" candidates) and more exploratory changes that help understand the design space ("explore-heavy" candidates).

Promising directions to explore:
- **Architecture changes**: depth recurrence, parameter tying across layers, low-rank factorizations, different attention patterns, mixture of experts with shared parameters
- **Compression improvements**: quantization-aware training, structured sparsity, sub-byte quantization, better zlib-friendliness
- **Training optimization**: learning rate schedules, batch size tuning, curriculum learning, distillation from larger intermediate models
- **Tokenizer/vocab changes**: different vocabulary sizes (affects embedding size vs token efficiency)
- **Novel approaches**: test-time compute, adaptive depth, conditional computation

You must consider the results of past experiments when designing your candidate. If past results show poor performance from a strategy, reason about why and whether a modification could help.

Your task is to analyze the current state-of-the-art algorithm, construct a candidate by editing it, and write the final Python output code.

{CODING_REQ}

Please follow these steps:

1. Explanation of the current state-of-the-art.
2. Brainstorm several possible ideas. Try to be creative while also considering the results of past experiments. Provide a reasoning for each idea.
3. Think through which idea is the most promising one to implement. Explain your reasoning, select the best idea (or combination of ideas), and describe your proposed modification.
4. Code implementation of the candidate.
"""
    return prompt


def construct_idea_gen_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for the Parameter Golf challenge.

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

When proposing a new design or hyperparameter configuration, you should start by conducting a research brainstorming exercise where you develop 3 different options to explore the design space. Go through each option, providing a comprehensive explanation for the proposed changes including:
* The underlying rationale and expected impact on BPB.
* How it affects the artifact size (does it increase or decrease the compressed model size?).
* The specific reason why you expect this experiment to be worth running.

# Final Instructions

You should try to understand why a particular design performed well / poorly, so that you can make a more informed choice for the next set of designs. It is important to strike a good balance between exploration of the design space and tuning to find the best hyperparameter values.

You are STRONGLY encouraged to look through your experiment history and refer back to the designs that we have already tested, to make sure that the design you are proposing makes sense in that broader research context (i.e., are not too similar, and are informed by past results). If an idea has seen rich experiment history but the performance has plateaued, then perhaps it's time to switch to a new idea.

Remember: the goal is to minimize BPB while keeping the artifact under 16MB and training under 10 minutes. A 0.005 BPB improvement is considered significant.

Go through the idea and experiment history carefully, DO NOT re-propose an idea that has been well tested already.

You should follow the following format when generating ideas:
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
We are conducting an evolutionary optimization process for the Parameter Golf challenge.

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

Your job is to come up with an experiment to test one of the ideas in the idea repo. Think about an experiment to run that you think will stand the best shot of helping you accomplish your overall goal (to discover a candidate with the lowest possible BPB).

Please avoid proposing experiments that you have already tested - the results will almost certainly be the same and this will waste computational resources.

Go through each idea's experiment history carefully to understand how well it has been tested.

You should balance exploration vs exploitation when selecting ideas via developing an upper confidence bound (UCB(a) = bar x_a + sqrt[(2 ln n)/n_a]) based on the experiment results and number of explorations.

# Final Instructions

Remember: every experiment costs ~10 minutes on 8xH100 GPUs. Make each one count.

Go through the idea history carefully, DO NOT propose an experiment that has been tried before in the idea repo.

You should use the following format for the idea selection part:

Idea ID: <Idea ID>
Experiment description: <Provide a concrete but concise description on the experiment you want to try, DO NOT HALLUCINATE IDEA ID, YOU MUST SELECT ONE FROM the idea repo above>

Once your brainstorming and idea generation process is finished, you are ready to write code. Please follow the guideline when completing the coding part:

{CODING_REQ}

"""
    return prompt


SUMMARIZE_EVAL_PROMPT = """
## Your Task

Your task is to provide a final concise summary of this entire experiment iteration. This summary will be added to our knowledge base and used to inform future experiments. First, summarize the key findings in a short paragraph. Then, provide **exactly 1 bullet point** summarizing the key findings and your final lesson. Each bullet MUST start on a new line and begin with a hyphen (-). Keep your bullets SHORT - they do not need to be complete sentences; they just need to be clear and detailed. DO NOT include obviously true or trivial statements.

Be concrete about which changes led to which results. Do not use vague terms such as SoTA since SoTA is constantly evolving. Your statements must be self-contained.

In the bullet point, first include the best result from the current trial before introducing the experiment and analyzing the results (in the format of Results: val_bpb: xxx, artifact_bytes: xxx).

Here is an example of a good summary:
- Results: val_bpb: 1.2100, artifact_bytes: 15200000. This experiment tested depth recurrence (sharing weights across layers 1-4 and 5-8) which reduced artifact size by 2MB while only increasing BPB by 0.002. The freed budget was used to increase model_dim from 512 to 576, resulting in a net BPB improvement of 0.014.
"""


EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The table presents the validation BPB (bits per byte) and artifact size.

### Understanding metrics
A lower val_bpb is better (better compression).
The artifact size must be <= 16,000,000 bytes.
Training must complete within 600 seconds.
"""

HPARAM_PROMPT = """
## Hyperparameter tuning
Would you like to tune any hyperparameters?

Remember that the model must fit in 16MB and train in under 10 minutes.
Key hyperparameters to consider: model_dim, num_layers, num_heads, num_kv_heads, mlp_mult, vocab_size, learning rates, batch size, warmdown schedule.

If yes, explain your reasoning and respond with ONE candidate that you would like to try. Do not write any code yet. If you do not want to tune any hyperparameters, simply respond "No."
"""

HPARAM_IMPLEMENT_PROMPT = f"""
### Hyperparameter implementation
Please write the implementation of your hyperparameter candidate. Respond with a markdown-formatted code block that implements your improved algorithm.

{CODING_REQ}
"""

UPDATE_BASELINE_PROMPT = f"""
Should we update the baseline algorithm? Please answer yes or no then explain your reasoning. If the answer is yes, respond with a code block containing the candidate that we should use as the new baseline algorithm - this will most likely be the candidate that achieved the best (lowest) val_bpb overall. If no, simply respond "No."

{CODING_REQ}
"""


def construct_candidate_validation_prompt(candidate_code: str, reference_code: str) -> str:
    return f"""
You are reviewing a Parameter Golf candidate for reward hacking or invalid evaluation changes.

Protected reference implementation (`train_gpt_ref.py`):
```python
{reference_code}
```

Candidate implementation:
```python
{candidate_code}
```

Determine whether the candidate preserves the protected evaluation semantics from the reference.

You must focus on these protected areas:
1. Train/validation shard discovery and ordering, including any `sorted(glob.glob(...))` usage and sequential shard traversal.
2. Validation token loading, evaluation window construction, and BPB computation semantics in `load_validation_tokens`, `eval_val`, and `eval_val_sliding`.
3. Quantized round-trip evaluation semantics and any logic used to produce the final scored model.
4. Final metric reporting semantics, especially the `final_int8_zlib_roundtrip(_exact)` lines and the required `Candidate val_bpb:...` summary line.
5. Any sign of shortcutting evaluation, subsampling or reweighting validation data, reusing stale metrics, or otherwise making the score easier without a real model improvement.

It is OK for the candidate to change training, architecture, kernels, optimizer behavior, or performance engineering details as long as the protected evaluation semantics remain intact.

If you are unsure, return FAIL.

Respond in exactly this format:
Verdict: PASS or FAIL
Reason: <one concise paragraph>
Protected areas:
- <bullet>
- <bullet>
- <bullet>
"""
