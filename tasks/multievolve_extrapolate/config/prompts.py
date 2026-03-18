"""Prompts for the MULTI-evolve extrapolation benchmark."""

from __future__ import annotations

from pathlib import Path
import re


def _load_editable_block() -> str:
    src_path = Path(__file__).resolve().parents[1] / "src" / "multievolve_extrapolate_1.py"
    source = src_path.read_text(encoding="utf-8")
    match = re.search(
        r"# RegexTagCustomPruningAlgorithmStart\n(.*?)\n# RegexTagCustomPruningAlgorithmEnd",
        source,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not locate editable block in multievolve_extrapolate_1.py")
    return match.group(1).strip()


MULTIEVOLVE_EXTRAPOLATE = _load_editable_block()

CODING_REQ = """
While completing your task, you MUST:
- Enclose your code in triple backticks to properly format the code in Markdown.
- Your code will replace the content between RegexTagCustomPruningAlgorithmStart and RegexTagCustomPruningAlgorithmEnd in multievolve_extrapolate_1.py.
- You MUST define `fit_and_predict(train_df, test_df, dataset_context)`.
- `train_df` and `test_df` are pandas DataFrames with at least `mutant`, `fitness`, and `num_mutations`.
- Your function MUST return one numeric prediction per row of `test_df`.
- Do not read files or call the network inside the candidate.
- Each triple-backtick enclosed code block must contain valid Python and a valid implementation.
"""

BACKGROUND = """
### Background on the MULTI-evolve extrapolation benchmark

This task is derived from the MULTI-evolve paper's multimutant extrapolation benchmark.

The paper-style setup is:
- train on low-order variants, especially singles and doubles
- test on higher-order variants
- measure how well the model extrapolates from local epistatic observations to unseen multimutants

This repo task uses public DMS datasets from the MULTI-evolve benchmark data release and evaluates:
- mean Pearson R on held-out 3-10 mutants
- mean top-5% precision, where the predicted top 5% is compared with the truly best 5%

The final benchmark score is `0.7 * mean_pearson_r + 0.3 * mean_precision_top5`.
"""

TASK_INTRO = """
You are optimizing a public-data multimutant fitness extrapolator. The key challenge is to generalize from single and double mutation measurements to unseen higher-order combinations.
"""


def construct_mutation_prompt(sota_algorithm, ablation_list):
    ablation_descriptions = "\n".join(ablation_list)
    prompt = f"""
We are conducting an evolutionary optimization process for a MULTI-evolve-style multimutant extrapolation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
```python
{sota_algorithm}
```

# Knowledge base

{ablation_descriptions}

## Your Task

Brainstorm 5 concrete ways to improve extrapolation from singles+doubles to 3-10 mutants.
Promising directions include:
- better mutation features
- pairwise or higher-order interaction handling
- target transforms or calibration
- sample weighting across mutational load
- ensembling additive and epistatic models

Then pick one direction, explain it briefly, and write the code.
"""
    return prompt


def construct_idea_gen_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for a MULTI-evolve-style multimutant extrapolation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Generate 3 distinct extrapolation-improvement ideas.

Use this format:
** Idea 1 **
Hypothesis: <your idea>
Reasoning: <why it may improve Pearson R or top-5% precision>

** Idea 2 **
Hypothesis: <your idea>
Reasoning: <why it may improve Pearson R or top-5% precision>

** Idea 3 **
Hypothesis: <your idea>
Reasoning: <why it may improve Pearson R or top-5% precision>
"""
    return prompt


def construct_idea_select_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for a MULTI-evolve-style multimutant extrapolation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Select one idea to test next.

Use this format:

Idea ID: <Idea ID>
Experiment description: <concrete implementation idea>
"""
    return prompt


def construct_idea_select_no_code_prompt(sota_algorithm, idea_repo):
    return construct_idea_select_prompt(sota_algorithm, idea_repo)


def construct_code_impl_prompt(sota_algorithm, idea_id, exp_description, selected_idea_text=None):
    idea_section = f"""Idea ID: {idea_id}
Experiment description: {exp_description}"""
    if selected_idea_text:
        idea_section = f"""Selected idea:
{selected_idea_text}

Idea ID: {idea_id}
Experiment description: {exp_description}"""

    prompt = f"""
We are conducting an evolutionary optimization process for a MULTI-evolve-style multimutant extrapolation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
```python
{sota_algorithm}
```

## Your Task

Implement the selected idea below.

{idea_section}
"""
    return prompt


def construct_gen_hypothesis_prompt(sota_algorithm, idea_repo, idea):
    prompt = f"""
We are conducting an evolutionary optimization process for a MULTI-evolve-style multimutant extrapolation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Design one concrete experiment for idea {idea.id}.

Use this format:
Idea ID: {idea.id}
Experiment description: <concrete implementation idea>
"""
    return prompt


TOURNAMENT_PROMPT = "\n"

SUMMARIZE_EVAL_PROMPT = """
## Your Task

Provide a concise summary of this experiment. First write one short paragraph. Then provide exactly 1 bullet point beginning with `-`.

In the bullet, include:
- Results: combined_score, mean_pearson_r, mean_precision_top5
- The specific model or feature change that mattered
"""

EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The evaluator reports extrapolation quality across public DMS datasets.

### Understanding metrics
- `mean_pearson_r` measures global multimutant ranking/regression quality.
- `mean_precision_top5` measures recovery of the truly best high-order variants.
- `combined_score = 0.7 * mean_pearson_r + 0.3 * mean_precision_top5`.
"""

HPARAM_PROMPT = """
## Hyperparameter tuning
Would you like to tune any hyperparameters? If yes, explain your reasoning and respond with ONE candidate. If no, simply respond "No."
"""

HPARAM_IMPLEMENT_PROMPT = f"""
### Hyperparameter implementation
Please write the implementation of your hyperparameter candidate. Respond with a markdown-formatted code block.

{CODING_REQ}
"""

UPDATE_BASELINE_PROMPT = f"""
Should we update the baseline algorithm? Please answer yes or no then explain your reasoning. If yes, respond with a code block containing the candidate. If no, simply respond "No."

{CODING_REQ}
"""
