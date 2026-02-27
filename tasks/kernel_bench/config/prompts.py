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

import os
import sys

def read_file(file_path: str) -> str:
    """
    Reads the entire content of a file and returns it as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        raise

def load_kernel_prompt(
    kernel_name: str,
    base_src_path: str = "~/auto_evo/tasks/kernel_bench/src/kernels/",
    filename: str = "kernel_archive.py"
) -> str:
    """
    Loads the source code for a kernel given its name.
    
    This constructs the file path based on the convention:
    {base_src_path}/{kernel_name}/{filename}
    
    Args:
        kernel_name: The name of the kernel (e.g., "BatchNorm").
        base_src_path: The base directory containing all kernel source folders.
        filename: The name of the kernel source file to read.
        
    Returns:
        The content of the kernel file as a string.
    """
    # Construct the full path
    # e.g., /home/minghaoyan_google_com/.../kernels/BatchNorm/kernel_archive.py
    base_path = os.path.expanduser(base_src_path)
    file_path = os.path.join(base_path, kernel_name, filename)
    
    # print(f"Loading kernel prompt from: {file_path}")
    
    # Use the read_file utility to get the content
    return read_file(file_path)

BATCHNORM = load_kernel_prompt("BatchNorm")
CONV3D_DIVIDE = load_kernel_prompt("Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum")
CONV3D_MAX = load_kernel_prompt("Conv3d_Max_LogSumExp_ReLU")
CONV_TRANSPOSE_2D = load_kernel_prompt("ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide")
GELU = load_kernel_prompt("GELU")
MM_LARGE_K = load_kernel_prompt("Matmul_with_large_K_dimension")
MAX_POOL_2D = load_kernel_prompt("Max_Pooling_2D")
MLP = load_kernel_prompt("MLP")
RMSNORM = load_kernel_prompt("RMSNorm")
SOFTMAX = load_kernel_prompt("Softmax")
VGG16 = load_kernel_prompt("VGG16")
MEAN_RED = load_kernel_prompt("Mean_reduction_over_a_dimension")
CONV3D_SQ = load_kernel_prompt("Conv3d_Square")
BMM_NORM = load_kernel_prompt("BMM_InstanceNorm_Sum_ResidualAdd_Multiply")
ALEXNET = load_kernel_prompt("AlexNet")
MOBILENETV2 = load_kernel_prompt("MobileNetV2")
RNN = load_kernel_prompt("RNN")
SQ_MM = load_kernel_prompt("Square_matrix_multiplication")
BMM = load_kernel_prompt("Batched_matrix_multiplication")
MM_T = load_kernel_prompt("Matmul_with_transposed_both")
LAYERNORM = load_kernel_prompt("LayerNorm")

CODING_REQ = """
While completing your task, you MUST:
- Enclose your code in triple backticks to properly format the code in Markdown.
- You MUST name your class ModelNew(nn.Module) as shown in the example. You must not change the function signatures of __init__(self, num_features: int), and forward(self, x: torch.Tensor) -> torch.Tensor either. If you need to define a new parameter, hard code it in the constructor.
- You DO NOT need to escape special characters, such as newlines, with a double backslash when writing code. The system understands the traditional "backslash-n" character style.
- Each triple-backtick enclosed code block in your output must contain valid Python and be a valid implementation.
"""

RJCH_DOCS = f"""
### Codebase documentation
Your code will be run to evaluate the correctness and runtime of your kernel. You must ensure the correctness of your kernel while optimizing the runtime.

#### Writing code

{CODING_REQ}
"""

TASK_INTRO = """
You are an expert researcher who specializes in high performance machine learning kernels. We are studying optimizing the kernel of a given machine learning operator (could be an operation, a neural network layer, or a model).
"""

# The following prompt is modified from GEPA: https://arxiv.org/pdf/2507.19457
BACKGROUND = """
To optimize a given PyTorch model by replacing operators with custom CUDA kernels, follow these detailed instructions. Your goal is to achieve performance improvements while ensuring correctness. Name your optimized output architecture ‘ModelNew‘. Output the new model code in codeblocks. Please generate real code, NOT pseudocode, and ensure the code compiles and is fully functional. Do not include testing code.

### Steps to Create Custom CUDA Kernels

#### 1. Identify Operators to Replace

* Analyze the model to identify operators that can benefit from custom CUDA implementations.

* Consider operator fusion opportunities to combine multiple operations into a single kernel for efficiency.

#### 2. Setup and Compilation

* Use ‘torch.utils.cpp_extension.load_inline‘ to compile your CUDA code. This allows you to define and compile custom CUDA kernels directly within your Python script.

* Ensure all necessary CUDA and C++ headers are included to avoid missing includes errors.

#### 3. Implementing the CUDA Kernel

* Write the CUDA kernel code, ensuring it is optimized for parallel execution. Use shared memory to reduce global memory accesses and ensure coalesced memory access patterns.

* Example structure for a CUDA kernel:

```cpp
__global__ void my_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Perform computation
    }
}

```

#### 4. Kernel Launch Configuration

* Configure the kernel launch parameters (number of blocks and threads per block) to maximize GPU utilization.

* Example:

```
const int block_size = 256;
const int num_blocks = (size + block_size - 1) / block_size;
my_kernel<<<num_blocks, block_size>>>(input, output, size);

```

#### 5. Error Handling and Debugging

* Implement error checking after CUDA API calls and kernel launches using `cudaGetLastError()` to catch errors early.

* Use CUDA debugging tools like ‘cuda-memcheck‘ and NVIDIA Nsight for debugging and profiling.

* Be aware of common syntax errors and namespace issues, and ensure that the CUDA code adheres to syntax rules.

#### 6. Integrating with PyTorch

* Define a Python function that wraps the CUDA kernel call and integrates it into the PyTorch model.

* Example:

```python
def my_custom_op(input):
    output = torch.zeros_like(input)
    my_kernel(input.data_ptr<float>(), output.data_ptr<float>(), input.numel())
    return output

```

#### 7. Compatibility and Compute Capability

* Ensure that the CUDA code is compatible with the target GPU’s compute capability (A100-SXM4-40GB).

### Additional Best Practices

* **Optimize Memory Usage**: Minimize data transfers between host and device, and use shared memory to reduce global memory access.

* **Atomic Operations**: When using atomic operations like ‘atomicMax‘ with floating-point numbers, ensure correct usage by following best practices, such as using appropriate data types and minimizing contention.

* **Performance Optimization**: Maximize parallel execution, optimize memory access patterns, and use compiler flags to enhance performance.

* **Namespace Usage**: Avoid adding class declarations or function definitions directly to reserved namespaces like ‘cuda‘. Use nested namespaces within non-reserved namespaces to organize code.

* **Numerical Precision**: Be aware of floating-point arithmetic issues, such as non-associativity, and use appropriate precision levels for calculations.

By following these instructions, you can effectively replace PyTorch operators with custom CUDA kernels, ensuring both performance improvements and correctness.

### Instruction for Replacing PyTorch Operators with Custom CUDA Kernels

Your task is to optimize a given PyTorch model by replacing certain operators with custom CUDA kernels to achieve performance improvements. Follow the steps below to ensure a successful implementation:

#### Step 1: Identify Operators for Replacement

* **Criteria for Selection**: Choose operators that are computationally intensive and have potential for parallelization. Consider operators that are frequently used in the model’s forward pass.

* **Operator Fusion**: Look for opportunities to fuse multiple operators into a single CUDA kernel, such as combining matrix multiplication with activation functions (e.g., matmul + ReLU).

#### Step 2: Implement Custom CUDA Kernels

* **Kernel Structure**: Define your CUDA kernel using the ‘**global**‘ specifier. Ensure that each thread handles a specific part of the computation. Use correct index calculations to access data.

* **Memory Management**:

  * Allocate memory for input, output, and any intermediate data on the GPU using `cudaMalloc`. Use `cudaMemcpy` to transfer data between host and device.

  * Utilize shared memory to cache frequently accessed data and reduce global memory accesses.

  * Ensure coalesced global memory accesses for efficient memory transactions.

* **Numerical Stability and Boundary Conditions**:

  * Implement verification mechanisms to ensure numerical stability. Use ‘**host** **device**‘ functions for testing on both CPU and GPU.

  * Handle boundary conditions to prevent out-of-bounds memory access. Ensure that thread indices are within valid ranges.

* **Optimization Techniques**:

  * Use shared memory to reduce global memory accesses and improve performance.

  * Consider using mixed precision and Tensor Cores for matrix operations to enhance performance.

  * Avoid diverged execution paths to maintain efficient parallel execution.

#### Step 3: Integrate CUDA Kernels into PyTorch Model

* **Inline Compilation**: Use ‘torch.utils.cpp_extension.load_inline‘ to compile your CUDA code and integrate it into the PyTorch model.

* **Model Modification**: Replace the original PyTorch operators with calls to your custom CUDA functions. Ensure that the new model architecture (‘ModelNew‘) is fully functional and compiles without errors.

### Evaluation
You will be graded on kernel correctness and the runtime of the kernel. The lower the runtime, the better the kernel.
"""

def construct_mutation_prompt(sota_algorithm, ablation_list):
  ablation_descriptions = "\n".join(ablation_list)
  prompt = f"""
We are conducting an evolutionary optimization process for optimizing high performance machine learning kernels.

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
We are conducting an evolutionary optimization process for optimizing high performance machine learning kernels.

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
Idea 1
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>

Idea 2
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>

...

Idea N
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>

"""
    return prompt


def construct_idea_select_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for optimizing high performance machine learning kernels.

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

You should use the following format for the idea selection part:

Idea ID: <Idea ID>
Experiment description: <Provide a concrete but concise description on the experiment you want to try, DO NOT HALLUCINATE IDEA ID, YOU MUST SELECT ONE FROM the idea repo above, however, you can provide more details about the experiment detail you want to try based on that idea>

Once your brainstorming and idea generation process is finished, you are ready to write code. Please follow the guideline when completing the coding part:

{CODING_REQ}


"""
    return prompt


def construct_idea_select_no_code_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for optimizing high performance machine learning kernels.

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

"""
    return prompt




def construct_code_impl_prompt(sota_algorithm, idea_id, exp_description):
    prompt = f"""
We are conducting an evolutionary optimization process for optimizing high performance machine learning kernels.

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
We are conducting an evolutionary optimization process for optimizing high performance machine learning kernels.

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

You should balance the exploration vs exploitation tradeoff when you selct your ideas via developing an upper confidence bound (UCB(a) = bar x_a + \sqrt[(2 \ln n)/n_a]) based on the experiment results you see and the number of times each idea has been tried (We will provide you with this information).

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
In the bullet point, first include the best result from the current trial before introducing the exp and analyze the results (in the format of Results: Kernel speedup: <Baseline time / Kernel time>). If kernel speedup is greater than 1, then the proposed solution is faster than the baseline.

Here is an example of a good summary:
- Results: Kernel speedup: <Baseline time / Kernel time>. This experiment examined a tan term to the equation since prior results indicate a more complex relationship between the t and v. Results show improvements with the newly introduced tan term.
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
Create a lightweight Python analysis function for the current candidate before evaluation.

Output a markdown code block that defines:
`def analyze_candidate(candidate_source: str) -> dict[str, float]:`

Requirements:
- Return numeric metrics only.
- Prefix metric names with `analysis_`.
- Keep compute cheap and deterministic.
- Focus on metrics that help explain failure modes and optimization potential.
"""
