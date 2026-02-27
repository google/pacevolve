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

GPT = """
Please Paste the GPT training code here. Sample training code can be obtained from https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
"""

RJCH_DOCS = f"""
### Codebase documentation
Your code will be used to train a language model.

"""

TASK_INTRO = """
You are an expert researcher who specializes in LLM. We are studying how to improve the training efficiency of a ~124M scale LLM. Our goal is to improve the Pareto frontier of validation loss and training time. You will be given a fixed corpus and stepsize to train the model on.
"""


BACKGROUND = """
### Background on Large Language Models (LLMs)

A Large Language Model (LLM) is an advanced type of artificial intelligence model designed to understand, summarize, translate, predict, and generate coherent, human-like text. At their core, most LLMs are trained on a fundamental objective: **next-word prediction**. Given a sequence of words, the model learns to predict the most probable next word. This simple objective, when trained on a massive and diverse corpus of text from the internet, enables the model to learn grammar, facts, reasoning abilities, and different styles of writing.

***

### Benefits and Properties

* **Unsupervised Learning**: LLMs can be trained on vast amounts of raw text without needing explicit human-made labels, making it possible to leverage internet-scale datasets.
* **Contextual Understanding**: Based on the Transformer architecture, the self-attention mechanism is exceptionally effective at capturing long-range dependencies, allowing the model to maintain context over long passages of text.
* **Scalability**: The Transformer architecture scales remarkably well. Increasing the model size (more layers, larger embeddings), training data, and compute consistently leads to better performance and the emergence of new capabilities not explicitly trained for (e.g., translation, summarization).

***

### LLM Evaluation

How well can a Large Language Model learn a language? It can be evaluated based on its performance during training and its efficiency, using standard industry metrics.

* **Validation Loss**: This is the model's Cross-Entropy Loss calculated on a held-out "validation" dataset that it does not see during training. This metric is the primary indicator of how well the model is **generalizing** to new text, rather than just memorizing the training data. A lower validation loss is better.
* **Training Time**: The total wall-clock time required to train the model to reach our target loss. If you don't reach the target loss, then training time is MEANINGLESS. It would be considered a FAILURE.

Note your PRIMARY goal is to reach a validation loss target with as little time as possible. If you don't reach the target loss, then any time improvement is MEANINGLESS. A 0.01 validation loss increase is considered significant!
"""

KNOWLEDGE_BASE = """
# Knowledge base
- Muon optimizer improved validation loss.
- Architectural changes including untied embeddings and output heads, padded embeddings, ReLU² activations, zero-initialized projections, and QK-normalization improved performance. ReLU^2 is known to perform better than GELU and SwiGLU.
- Various skip connections, including U-net patterns for values and embeddings improved performance.
- Merged the QKV (Query, Key, Value) weights and implemented a long-short attention pattern to improve performance.
- Applying RoPE to half truncated head dimension is known to perform better than applying it to full head dimension.
- There may be a way to distribute the load of finding bos token indicies for all 8 files. If each GPU is given 1 file instead of 8 to locate the bos_tokens, this could save up to roughly 200ms*7 = 1.4 seconds assuming zero overhead.
- The total number of [4,768,768] attention variables to 10. There are 22 MLP variables of size [768x4,768]. In Muon attention is getting batched such that 6/16ths on the gradient calcs are on padding tokens. There may be a way to move 2 of the attention variables into the MLP batch, such that MLP is 24/24 and attn is 8/8, instead of MLP being 22/24 and attn being 10/16.
- Each optimization is there for a reason, reverting them to a more conventional set up will very likely degrade the performance, so DO NOT try an idea just because it is a "common" approach. Instead, BE CREATIVE!
"""

def construct_mutation_prompt(sota_algorithm, ablation_list):
  ablation_descriptions = "\n".join(ablation_list)
  prompt = f"""
We are conducting an evolutionary optimization process for training a language model.

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

You will make reasonable changes to improve the performance of language model training (We provide you with performance metrics targets, note that they are not necessarily the performance of the current SoTA algorithms). The experiment strategy is to improve the current state-of-the-art algorithm by evaluating dozens or hundreds of candidates with small perturbations. Try to strike a balance between safe, easy changes that are very likely to improve performance ("exploit-heavy" candidates) and more exploratory changes that help us understand the space of possible algorithms ("explore-heavy" candidates).

Anything except increasing model size is fair game. You understand certain changes cause change in model size (such as different attention mechanism), but keep model size under 150M. You may think about how to improve the optimizer, introducing/replacing different layer types or other architectural changes (just don't increase model size too much), different positional embeddings. Reason about how those changes affect both validation loss and training time.

You must consider the results of past experiments when designing your candidate. For example, if the notes show poor performance from a strategy, does it make sense to try the opposite? Is there a hyperparameter that needs to be tuned properly? Is there any additional information that you can compute or obtain? These are just suggestions - feel free to come up with additional directions to explore.

Your task is to analyze the current state-of-the-art algorithm, construct an algorithm candidate by editing the current state-of-the-art method, and write the final Python output code for the candidate.

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
We are conducting an evolutionary optimization process for training a language model.

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

When proposing a new design or hyperparameter configuration, you should start by conducting a research brainstorming exercise where you develop 3 different options to explore the design space. Go through each option, providing a comprehensive explanation for the proposed changes including
* The underlying rationale and expected impact.
* The specific reason why you expect this experiment to be worth running.

# Final Instructions

You should try to understand why a particular design performed well / poorly, so that you can make a more informed choice for the next set of designs. It is important to strike a good balance between exploration of the design space and tuning to find the best hyperparameter values. One way to tune is to develop some hypotheses about how the design might affect performance, and then conduct experiments that would test those hypotheses.

You are encouraged to consider configurations that we might not have thought of before, and it is likely that you will discover some patterns during your search process. You are STRONGLY encouraged to look through your experiment history and refer back to the designs that we have already tested, to make sure that the design you are proposing makes sense in that broader research context (i.e., are not too similar, and are informed by past results). If an idea has seen rich experiment history but the performance has plateaued, then perhaps it's time to switch to a new idea.

Feel free to think outside of the box - you are allowed to propose candidates that are high risk (or even expected to be quality-negative) if it helps you understand the problem better, furthers your long-term research agenda, and (most importantly) increases your chance to produce a final candidate. The goal is to produce a final candidate after roughly 500 attempts, not to find a locally-optimal solution right away. That said, please do be mindful of your compute usage - do not redo configurations that you have already tried (the results will almost certainly be the same), and please do your best to deliver a good candidate by the end of your 500 attempts. We will take the best candidate discovered in your entire tuning process as your final candidate - it does not matter WHEN you find a good candidate, as long as you eventually do so.

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
We are conducting an evolutionary optimization process for training a language model.

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

You job is to come up with an experiment to test one of the ideas in the idea repo. Think about an experiment to run that you think will stand the best shot of helping you to accomplish your overall goal (to discover a good candidate at some point during your search).

Please avoid propose experiments that you have already tested - the results will almost certainly be the same and this will waste computational resources.

Go through each idea's experiment history carefully to understand how well it has been tested.

You should balance the exploration vs exploitation tradeoff when you selct your ideas via developing an upper confidence bound (UCB(a) = bar x_a + \sqrt[(2 \ln n)/n_a]) based on the experiment results you see and the number of explorations that we will provide you.

# Final Instructions

You should try to understand why a particular design performed well / poorly, so that you can make a more informed choice for the next set of experiments. 

You are encouraged to consider configurations that we might not have experimented with before, and it is likely that you will discover some patterns during your search process. You are encouraged to look through your experiment history and refer back to the designs that we have already tested, to make sure that the design you select makes sense in that broader research context (i.e., are not too similar, and are informed by past results). If an idea has seen rich experiment history but the performance has plateaued, then perhaps it's time to switch to a new idea.

Feel free to think outside of the box - you are allowed to select candidates that are high risk (or even expected to be quality-negative) if it helps you understand the problem better, furthers your long-term research agenda, and (most importantly) increases your chance to produce a final candidate. The goal is to produce a final candidate after roughly 500 attempts, not to find a locally-optimal solution right away. 

Go through the idea history carefully, DO NOT propose an experiment that has been tried before in the idea repo (the results will almost certainly be the same), and please do your best to deliver a good candidate by the end of your 500 attempts. We will take the best candidate discovered in your entire tuning process as your final candidate - it does not matter WHEN you find a good candidate, as long as you eventually do so.


You should use the following format for the idea selection part:

Idea ID: <Idea ID>
Experiment description: <Provide a concrete but concise description on the experiment you want to try, DO NOT HALLUCINATE IDEA ID, YOU MUST SELECT ONE FROM the idea repo above, however, you can provide more details about the experiment detail you want to try based on that idea>

Once your brainstorming and idea generation process is finished, you are ready to write code. Please follow the guideline when completing the coding part:

{CODING_REQ}


"""
    return prompt


SUMMARIZE_EVAL_PROMPT = """
## Your Task

Your task is to provide a final concise summary of this entire experiment iteration. This summary will be added to our knowledge base and used to inform future experiments. First, summarize the key findings in a short paragraph. This paragraph will not be used in the knowledge base: it just serves to help you organize your thoughts. Then, provide **exactly 1 bullet point** summarizing the key findings and your final lesson. Each bullet MUST start on a new line and begin with a hyphen (-). Keep your bullets SHORT - they do not need to be complete sentences; they just need to be clear and detailed. DO NOT include obviously true or trivial statements, such as "per-dataset hyperparameter tuning is important." Instead, focus on the key findings that will inform future experiments. 
Also, be concrete about which improvements to which current techniques lead to which results. Do not use vague terms such as SoTA since SoTA is constantly evolving and future audience may not know what you are referring to. Your statements must be self-contained. 
In the bullet point, first include the best result from the current trial before introducing the exp and analyze the results (in the format of Results: Training time: xxx).

Here is an example of a good summary:
- Results: Training time: xxx. This experiment examined a tan term to the equation since prior results indicate a more complex relationship between the t and v. Results show improvements with the newly introduced tan term and reached target validation loss in xxx ms.
"""


EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The table presents the validation loss and training time.

### Understanding metrics
A lower validation loss is better. A lower training time is better.
"""

HPARAM_PROMPT = """
## Hyperparameter tuning
Would you like to tune any hyperparameters?

Note that you MUST NOT write dataset-specific code (e.g., choose hyperparameters based on the dataset name). We will consider the best hyperparameter configuration on a per-dataset basis when evaluating algorithms.

If yes, explain your reasoning and respond with ONE candidate that you would like to try. Do not write any code yet - we will write the implementation in the next step. If you do not want to tune any hyperparameters, simply respond "No."
"""

HPARAM_IMPLEMENT_PROMPT = f"""
### Hyperparameter implementation
Please write the implementation of your hyperparameter candidate. Respond with a markdown-formatted code block that implements your improved algorithm.

{CODING_REQ}
"""

UPDATE_BASELINE_PROMPT = f"""
Should we update the baseline algorithm? Please answer yes or no then explain your reasoning. If the answer is yes, respond with a code block containing the candidate that we should use as the new baseline algorithm - this will most likely be the candidate that performed the best overall in hyperparameter ablations. If no, simply respond "No."

{CODING_REQ}
"""


PRE_EVAL_ANALYSIS_PROMPT = """
Create a lightweight Python analysis function for the current candidate before evaluation.

Output a markdown code block that defines:
`def analyze_candidate(candidate_source: str) -> dict[str, float]:`

Requirements:
- Return numeric metrics only.
- Prefix metric names with `analysis_`.
- Keep compute cheap and deterministic.
- Focus on metrics that help explain failure modes and optimization potential.
"""
