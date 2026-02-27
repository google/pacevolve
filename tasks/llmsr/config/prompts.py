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

LLMSR = '''
    def equation(self, x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        """ Mathematical function for Acceleration in Non-linear Harmonic Oscillator

        Args:
            x: A numpy array representing observations of Position at time t.
            t: A numpy array representing observations of Time.
            v: A numpy array representing observations of Velocity at time t.
            params: Array of numeric constants or parameters to be optimized

        Return:
            A numpy array representing Acceleration in Non-linear Harmonic Oscillator as the result of applying the mathematical function to the inputs.
        """
        output = params[0] * x + params[1] * t + params[2] * v + params[3]
        return output
'''

CODING_REQ = """
While completing your task, you MUST:
- Enclose your code in triple backticks to properly format the code in Markdown.
- You MUST indent all functions one block like in the example since these methods are going to be embedded in a well-defined Python class. 
- You MUST NOT import any libraries at the beginning of the code due to indentation issue, you code block is embedded in a Python class. If you need to import Python native libraries, do it inside the function. We have already imported `import numpy as np` for you so you don't need to import yourself.
- You MUST NOT change the function signature (arguments, return value, and name). The system expects to call function `equation(self, x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray` with signature as shown above.
- Your solution should contain the function defined above and the function ONLY. DO NOT import packages globally, DO NOT wrap those functions in a class, DO NOT define a class, DO NOT define an __init__ method.
- You DO NOT need to escape special characters, such as newlines, with a double backslash when writing code. The system understands the traditional "backslash-n" character style.
- Each triple-backtick enclosed code block in your output must contain valid Python and be a valid implementation.
- DO NOT reimplement evaluate() nor compute_output_base_metrics() in your code, ONLY implement equation(). evaluate() and compute_output_base_metrics() will be provided during evaluation.
"""

RJCH_DOCS = f"""
### Codebase documentation
Your code will be run to find the mathematical function skeleton that represents Acceleration in Non-linear Harmonic Oscillator, given data on Position at time t, Time, and Velocity at time t, with our algorithm.

#### Writing code

{CODING_REQ}
"""

TASK_INTRO = """
You are an expert researcher who specializes in Physics. We are studying fitting the Acceleration part in a Non-linear Harmonic Oscillator, an important problem in physics. Our goal is to find the correct equation for a dataset.
"""


BACKGROUND = """
### Background on Acceleration in a Non-linear Harmonic Oscillator

The simple harmonic oscillator (SHO) is a fundamental model in physics, with a restoring force directly proportional to displacement. Many real-world systems, however, have a non-linear restoring force, leading to a **non-linear harmonic oscillator**. In these systems, acceleration is not a simple sinusoidal function of time.

def compute_output_base_metrics(self, y_pred, y):
    nonnan_idx = np.argwhere(~np.isnan(y_pred))
    y_pred = y_pred[nonnan_idx]
    y = y[nonnan_idx]

    var = np.var(y)
    nmse = np.mean((y - y_pred)**2) / var 
    if np.sum((y - y.mean())**2) == 0:
        print(y)
    r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))
    kdt = kendalltau(y, y_pred)[0]
    mape = mean_absolute_percentage_error(y, y_pred)
    log10_nmse = np.log10(nmse)

    return {
        "mse": np.mean((y - y_pred)**2),
        "nmse": nmse,
        "log10_nmse": log10_nmse,
        "r2": r2,
        "kdt": kdt,
        "mape": mape,
    }

def evaluate(self, train_data: dict, test_data: dict, ood_test_data: dict) -> float:
    ''' Evaluate the equation on data observations.'''
    
    # Load data observations
    MAX_NPARAMS = 10
    params = [1.0]*MAX_NPARAMS
    inputs, outputs = train_data['inputs'], train_data['outputs']
    X = inputs
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        x_inputs = X[:, 0]
        t_inputs = X[:, 1]
        v_inputs = X[:, 2]

        # Pass the individual columns to the equation
        y_pred = self.equation(x_inputs, t_inputs, v_inputs, params)
        # y_pred = self.equation(*X, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    inputs, outputs = test_data['inputs'], test_data['outputs']
    X = inputs
    x_inputs = X[:, 0]
    t_inputs = X[:, 1]
    v_inputs = X[:, 2]
    
    y_pred = self.equation(x_inputs, t_inputs, v_inputs, optimized_params)
    metrics = self.compute_output_base_metrics(y_pred, outputs)

    inputs, outputs = ood_test_data['inputs'], ood_test_data['outputs']
    X = inputs
    x_inputs = X[:, 0]
    t_inputs = X[:, 1]
    v_inputs = X[:, 2]
    
    y_pred = self.equation(x_inputs, t_inputs, v_inputs, optimized_params)
    ood_metrics = self.compute_output_base_metrics(y_pred, outputs)

    return {'log10_nmse': metrics['log10_nmse'], 'ood_log10_nmse': ood_metrics['log10_nmse']}

### Evaluation
You will be graded on how well your proposed equation fits the provided data, with a lower mean squared error yielding a better score.
"""

def construct_mutation_prompt(sota_algorithm, ablation_list):
  ablation_descriptions = "\n".join(ablation_list)
  prompt = f"""
We are conducting an evolutionary optimization process for Acceleration in a Non-linear Harmonic Oscillator.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

# Knowledge base

{ablation_descriptions}

## Your Task

When proposing a new design or hyperparameter configuration, you should start by conducting a research brainstorming exercise where you develop 5 different options to explore the design space. Go through each option, providing a comprehensive explanation for the proposed changes including
* The underlying rationale and expected impact.
* The specific reason why you expect this experiment to be worth running.

Once you have brainstormed enough, pick an option that you think will stand the best shot of helping you to accomplish your overall goal (to discover a good candidate at some point during your search). Please try to strike a good balance between exploring dramatically new directions and proposing "safer" incremental changes - both types of exploration are equally important to make good progress. Once you have selected your final idea - the one that we will actually implement and apply to the model - write down your final idea and provide a concise explanation. Please avoid re-training models / designs that you have already tested - the results will almost certainly be the same and this will waste computational resources.

Note that we will ask you to summarize the idea and results after we run the experiment, but you will not have access to your brainstorming notes at that time. For this reason, your code should contain a complete description of your final idea, motivation, hypotheses, etc. It is okay to have a large file-level comment with this information, since this is research code.

Once your brainstorming and idea generation process is finished, you are ready to write code.

{CODING_REQ}

# Final Instructions


You should try to understand why a particular design performed well / poorly, so that you can make a more informed choice for the next set of designs. It is important to strike a good balance between exploration of the design space and tuning to find the best hyperparameter values. One way to tune is to develop some hypotheses about how the design might affect performance, and then conduct experiments that would test those hypotheses.

You are encouraged to consider configurations that we might not have thought of before, and it is likely that you will discover some patterns during your search process. You are STRONGLY encouraged to look through your experiment history and refer back to the designs that we have already tested, to make sure that the design you are proposing makes sense in that broader research context (i.e., are not too similar, and are informed by past results).

Feel free to think outside of the box - you are allowed to propose candidates that are high risk (or even expected to be quality-negative) if it helps you understand the problem better, furthers your long-term research agenda, and (most importantly) increases your chance to produce a final candidate. The goal is to produce a final candidate after roughly 500 attempts, not to find a locally-optimal solution right away. That said, please do be mindful of your compute usage - do not redo configurations that you have already tried (the results will almost certainly be the same), and please do your best to deliver a good candidate by the end of your 500 attempts. We will take the best candidate discovered in your entire tuning process as your final candidate - it does not matter WHEN you find a good candidate, as long as you eventually do so.

The knowledge base contains summarization of the trials you have done so far.
  """
  return prompt


def construct_idea_gen_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for Acceleration in a Non-linear Harmonic Oscillator.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```


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

When there are less than or equal to 5 ideas, you should try to think out of the box and come up with at least one idea that does not fall under existing ideas.

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
We are conducting an evolutionary optimization process for Acceleration in a Non-linear Harmonic Oscillator.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```


### Idea Repo
Idea repos contain ideas that we have generated so far, and experiments we have run to test these hypotheses.

{idea_repo}

## Your Task

You job is to come up with an experiment to test one of the ideas in the idea repo. Think about an experiment to run that you think will stand the best shot of helping you to accomplish your overall goal (to discover a good candidate at some point during your search).

Please avoid propose experiments that you have already tested - the results will almost certainly be the same and this will waste computational resources.

Go through each idea's experiment history carefully to understand how well it has been tested.

# Final Instructions

You should try to understand why a particular design performed well / poorly, so that you can make a more informed choice for the next set of experiments. 

You are encouraged to consider configurations that we might not have experimented with before, and it is likely that you will discover some patterns during your search process. You are encouraged to look through your experiment history and refer back to the designs that we have already tested, to make sure that the design you select makes sense in that broader research context (i.e., are not too similar, and are informed by past results). If an idea has seen rich experiment history but the performance has plateaued, then perhaps it's time to switch to a new idea.

Feel free to think outside of the box - you are allowed to select candidates that are high risk (or even expected to be quality-negative) if it helps you understand the problem better, furthers your long-term research agenda, and (most importantly) increases your chance to produce a final candidate. The goal is to produce a final candidate after roughly 500 attempts, not to find a locally-optimal solution right away. 

Go through the idea history carefully, DO NOT propose an experiment that has been tried before in the idea repo (the results will almost certainly be the same), and please do your best to deliver a good candidate by the end of your 500 attempts. We will take the best candidate discovered in your entire tuning process as your final candidate - it does not matter WHEN you find a good candidate, as long as you eventually do so.

You should balance the exploration vs exploitation tradeoff when you selct your ideas via developing an upper confidence bound (UCB(a) = bar x_a + \sqrt[(2 \ln n)/n_a]) based on the experiment results you see and the number of times each idea has been tried (We will provide you with this information).

You should use the following format for the idea selection part:

Idea ID: <Idea ID>
Experiment description: <Provide a concrete but concise description on the experiment you want to try, DO NOT HALLUCINATE IDEA ID, YOU MUST SELECT ONE FROM the idea repo above, however, you can provide more details about the experiment detail you want to try based on that idea>

Once your brainstorming and idea generation process is finished, you are ready to write code. Please follow the guideline when completing the coding part:

{CODING_REQ}


"""
    return prompt


def construct_idea_select_no_code_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for Acceleration in a Non-linear Harmonic Oscillator.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```


### Idea Repo
Idea repos contain ideas that we have generated so far, and experiments we have run to test these hypotheses.

{idea_repo}

## Your Task

You job is to come up with an experiment to test one of the ideas in the idea repo. Think about an experiment to run that you think will stand the best shot of helping you to accomplish your overall goal (to discover a good candidate at some point during your search).

Please avoid propose experiments that you have already tested - the results will almost certainly be the same and this will waste computational resources.

Go through each idea's experiment history carefully to understand how well it has been tested.

# Final Instructions

You should try to understand why a particular design performed well / poorly, so that you can make a more informed choice for the next set of experiments. 

You are encouraged to consider configurations that we might not have experimented with before, and it is likely that you will discover some patterns during your search process. You are encouraged to look through your experiment history and refer back to the designs that we have already tested, to make sure that the design you select makes sense in that broader research context (i.e., are not too similar, and are informed by past results). If an idea has seen rich experiment history but the performance has plateaued, then perhaps it's time to switch to a new idea.

Feel free to think outside of the box - you are allowed to select candidates that are high risk (or even expected to be quality-negative) if it helps you understand the problem better, furthers your long-term research agenda, and (most importantly) increases your chance to produce a final candidate. The goal is to produce a final candidate after roughly 500 attempts, not to find a locally-optimal solution right away. 

Go through the idea history carefully, DO NOT propose an experiment that has been tried before in the idea repo (the results will almost certainly be the same), and please do your best to deliver a good candidate by the end of your 500 attempts. We will take the best candidate discovered in your entire tuning process as your final candidate - it does not matter WHEN you find a good candidate, as long as you eventually do so.

You should balance the exploration vs exploitation tradeoff when you selct your ideas.

You should use the following format for the idea selection part:

Idea ID: <Idea ID>
Experiment description: <Provide a concrete but concise description on the experiment you want to try, DO NOT HALLUCINATE IDEA ID, YOU MUST SELECT ONE FROM the idea repo above, however, you can provide more details about the experiment detail you want to try based on that idea>

"""
    return prompt




def construct_code_impl_prompt(sota_algorithm, idea_id, exp_description):
    prompt = f"""
We are conducting an evolutionary optimization process for Acceleration in a Non-linear Harmonic Oscillator.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```


## Your Task

Your job is to implement the following idea:

Idea ID: {idea_id}
Experiment description: {exp_description}
Once your brainstorming and idea generation process is finished, you are ready to write code. Please follow the guideline when completing the coding part:

{CODING_REQ}


"""
    return prompt


def construct_gen_hypothesis_prompt(sota_algorithm, idea_repo, idea):
    prompt = f"""
We are conducting an evolutionary optimization process for Acceleration in a Non-linear Harmonic Oscillator.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

### Idea Repo
Idea repos contain ideas that we have generated so far, and experiments we have run to test these hypotheses.

{idea_repo}

## Your Task

You job is to come up with an experiment to test idea {idea.id} in the repo. Think about an experiment to run that you think will stand the best shot of helping you to accomplish your overall goal (to discover a good candidate at some point during your search).

Please avoid propose experiments that you have already tested - the results will almost certainly be the same and this will waste computational resources.

Go through each idea's experiment history carefully to understand how well it has been tested.

# Final Instructions

You should try to understand why a particular design performed well / poorly, so that you can make a more informed choice for the next set of experiments. 

You are encouraged to consider configurations that we might not have experimented with before, and it is likely that you will discover some patterns during your search process. You are encouraged to look through your experiment history and refer back to the designs that we have already tested, to make sure that the design you select makes sense in that broader research context (i.e., are not too similar, and are informed by past results). If an idea has seen rich experiment history but the performance has plateaued, then perhaps it's time to switch to a new idea.

Feel free to think outside of the box - you are allowed to select candidates that are high risk (or even expected to be quality-negative) if it helps you understand the problem better, furthers your long-term research agenda, and (most importantly) increases your chance to produce a final candidate. The goal is to produce a final candidate after roughly 500 attempts, not to find a locally-optimal solution right away. 

Go through the idea history carefully, DO NOT propose an experiment that has been tried before in the idea repo (the results will almost certainly be the same), and please do your best to deliver a good candidate by the end of your 500 attempts. We will take the best candidate discovered in your entire tuning process as your final candidate - it does not matter WHEN you find a good candidate, as long as you eventually do so.

You should balance the exploration vs exploitation tradeoff when you selct your ideas.

You should use the following format for the idea selection part:

Idea ID: {idea.id}
Experiment description: <Provide a concrete but concise description on the experiment you want to try, DO NOT HALLUCINATE IDEA ID, YOU MUST USE the given idea corresponding to the idea ID above, however, you can provide more details about the experiment detail you want to try based on that idea>

Once your brainstorming and idea generation process is finished, you are ready to write code. Please follow the guideline when completing the coding part:

{CODING_REQ}


"""
    return prompt


TOURNAMENT_PROMPT = """

"""


SUMMARIZE_EVAL_PROMPT = """
## Your Task

Your task is to provide a final concise summary of this entire experiment iteration. This summary will be added to our knowledge base and used to inform future experiments. First, summarize the key findings in a short paragraph. This paragraph will not be used in the knowledge base: it just serves to help you organize your thoughts. Then, provide **exactly 1 bullet point** summarizing the key findings and your final lesson. Each bullet MUST start on a new line and begin with a hyphen (-). Keep your bullets SHORT - they do not need to be complete sentences; they just need to be clear and detailed. DO NOT include obviously true or trivial statements, such as "per-dataset hyperparameter tuning is important." Instead, focus on the key findings that will inform future experiments. 
Also, be concrete about which improvements to which current techniques lead to which results. Do not use vague terms such as SoTA since SoTA is constantly evolving and future audience may not know what you are referring to. Your statements must be self-contained. 
In the bullet point, first include the best result from the current trial before introducing the exp and analyze the results (in the format of Results: nmse: xxx, ood_nmse: xxx).

Here is an example of a good summary:
- Results: nmse: xxx, ood_nmse: xxx. This experiment examined a tan term to the equation since prior results indicate a more complex relationship between the t and v. Results show improvements with the newly introduced tan term.
"""


EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The table presents regression loss by your proposed equation.

### Understanding metrics
You want your regression loss to be as small as possible.
"""

HPARAM_PROMPT = """
## Hyperparameter tuning
Would you like to tune any hyperparameters?

Note that you MUST NOT write dataset-specific code (e.g., choose hyperparameters based on the dataset name). We will consider the best hyperparameter configuration on a per-dataset basis when evaluating algorithms.

Note that you CANNOT tune servers, duplicates, objects, and epsilon, those are set during the eval phase.

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
Create a lightweight Python analysis function for the current candidate before evaluation that can help with analyzing the results 
and provide better undestanding for the solution in addtion to the outcome metrics.

Output a markdown code block that defines:
`def analyze_candidate(candidate_source: str) -> dict[str, float]:`

Requirements:
- Return numeric metrics only.
- Prefix metric names with `analysis_`.
- Keep compute cheap and deterministic.
- Focus on metrics that help explain failure modes and optimization potential.
"""
