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

EPLB = '''
import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight,
                                 fill_value=-1,
                                 dtype=torch.int64,
                                 device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i
                 for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt
'''

CODING_REQ = """
While completing your task, you MUST:
- Enclose your code in triple backticks to properly format the code in Markdown.
- Your code will replace the content between RegexTagCustomPruningAlgorithmStart and RegexTagCustomPruningAlgorithmEnd in eplb_1.py.
- You MUST NOT change the function signature of `rebalance_experts`. The system expects:
  def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  returning (physical_to_logical_map, logical_to_physical_map, expert_count).
- You MAY add helper functions (e.g., balanced_packing, replicate_experts) inside the editable block.
- You MUST use `import torch` at the top of your block; other imports can be added as needed.
- Each triple-backtick enclosed code block must contain valid Python and a valid implementation.
"""

BACKGROUND = """
### Background on Expert Parallelism Load Balancer (EPLB)

The EPLB algorithm rearranges experts in Mixture-of-Expert (MoE) models to balance load across GPUs. It takes load metrics from the vLLM server and rearranges experts, optionally creating replicas, to achieve better load balancing. The goals are:
1. Improve load balancing (higher balancedness score is better)
2. Reduce algorithm execution time (faster rebalancing is better, since perfect load balancing is NP-hard)

Your implementation is evaluated on a workload trace. The combined_score = (balancedness_score + speed_score) / 2, where speed_score rewards faster execution.
"""

TASK_INTRO = """
You are an expert programmer specializing in optimization algorithms. We are optimizing the Mixture-of-Expert (MoE) Expert Parallelism Load Balancer (EPLB) expert rearrangement algorithm for vLLM. Your goal is to improve both load balancing quality and algorithm efficiency.
"""


def construct_mutation_prompt(sota_algorithm, ablation_list):
    ablation_descriptions = "\n".join(ablation_list)
    prompt = f"""
We are conducting an evolutionary optimization process for the Expert Parallelism Load Balancer (EPLB).

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

# Knowledge base

{ablation_descriptions}

## Your Task

When proposing a new design, you should start by conducting a research brainstorming exercise where you develop 5 different options to explore the design space. Go through each option, providing a comprehensive explanation for the proposed changes including
* The underlying rationale and expected impact.
* The specific reason why you expect this experiment to be worth running.

Once you have brainstormed enough, pick an option that you think will stand the best shot of helping you accomplish your overall goal (better load balancing and/or faster execution). Strike a balance between exploration and incremental changes. Once you have selected your final idea, write it down and provide a concise explanation.

Once your brainstorming and idea generation process is finished, you are ready to write code.

The knowledge base contains summarization of the trials you have done so far.
"""
    return prompt


def construct_idea_gen_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for the Expert Parallelism Load Balancer (EPLB).

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

### Idea Repo
Idea repos contain ideas that we have generated so far, and experiments we have run to test these hypotheses.

{idea_repo}

## Your Task

When proposing a new design, you should start by conducting a research brainstorming exercise where you develop 3 different options to explore the design space. Go through each option, providing a comprehensive explanation for the proposed changes.

Go through the idea and experiment history carefully, DO NOT re-propose an idea that has been well tested already.

You should follow the following format when generating ideas:
** Idea 1 **
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>

** Idea 2 **
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>

** Idea 3 **
Hypothesis: <Your idea here>
Reasoning: <Your reasoning here>
"""
    return prompt


def construct_idea_select_prompt(sota_algorithm, idea_repo):
    prompt = f"""
We are conducting an evolutionary optimization process for the Expert Parallelism Load Balancer (EPLB).

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

### Idea Repo
Idea repos contain ideas that we have generated so far, and experiments we have run to test these hypotheses.

{idea_repo}

## Your Task

Your job is to come up with an experiment to test one of the ideas in the idea repo. Think about an experiment to run that will stand the best shot of helping you accomplish your overall goal (better load balancing and/or faster execution).

You should use the following format for the idea selection part:

Idea ID: <Idea ID>
Experiment description: <Provide a concrete but concise description on the experiment you want to try>
"""
    return prompt


def construct_idea_select_no_code_prompt(sota_algorithm, idea_repo):
    return construct_idea_select_prompt(sota_algorithm, idea_repo)


def construct_code_impl_prompt(sota_algorithm, idea_id, exp_description, selected_idea_text=None):
    idea_section = f"""Idea ID: {idea_id}
Experiment description: {exp_description}"""
    if selected_idea_text:
        idea_section = f"""A research agent selected the following idea for you to implement:

Selected idea:
{selected_idea_text}

Idea ID: {idea_id}
Experiment description: {exp_description}"""

    prompt = f"""
We are conducting an evolutionary optimization process for the Expert Parallelism Load Balancer (EPLB).

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

## Your Task

Your job is to implement the selected idea above.

{idea_section}
"""
    return prompt


def construct_gen_hypothesis_prompt(sota_algorithm, idea_repo, idea):
    prompt = f"""
We are conducting an evolutionary optimization process for the Expert Parallelism Load Balancer (EPLB).

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Your job is to come up with an experiment to test idea {idea.id} in the repo.

You should use the following format:
Idea ID: {idea.id}
Experiment description: <Provide a concrete but concise description on the experiment you want to try>
"""
    return prompt


TOURNAMENT_PROMPT = "\n"

SUMMARIZE_EVAL_PROMPT = """
## Your Task

Your task is to provide a final concise summary of this entire experiment iteration. This summary will be added to our knowledge base. First, summarize the key findings in a short paragraph. Then, provide **exactly 1 bullet point** summarizing the key findings and your final lesson. Each bullet MUST start on a new line and begin with a hyphen (-). Keep your bullets SHORT. Be concrete about improvements. In the bullet point, first include the best result from the current trial (in the format of Results: combined_score: xxx, balancedness_score: xxx, speed_score: xxx).
"""

EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The table presents the EPLB evaluation metrics for your proposed algorithm.

### Understanding metrics
combined_score = (balancedness_score + speed_score) / 2. Higher combined_score is better.
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
