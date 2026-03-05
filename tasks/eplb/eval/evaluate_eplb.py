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

"""CLI evaluator for EPLB (Expert Parallelism Load Balancer)."""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from typing import TypedDict

import torch

REBALANCE_INTERVAL = 100
NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4


class EvaluationResult(TypedDict, total=False):
    balancedness_score: float
    speed_score: float
    combined_score: float
    error: str


def load_workloads(path: str) -> list[torch.Tensor]:
    with open(path, "r") as f:
        data = json.load(f)

    total_len = len(data["load_history"])
    workloads = []
    for i in range(0, total_len, REBALANCE_INTERVAL):
        start = i
        end = min(start + REBALANCE_INTERVAL, total_len)
        load = torch.tensor(
            [x["logical_expert_load"] for x in data["load_history"][start:end]]
        ).sum(dim=0)
        workloads.append(load)

    return workloads


def simulate_inference(
    log2phy: torch.Tensor, logcnt: torch.Tensor, workload: torch.Tensor
) -> float:
    """Simulate MoE inference with the given expert mapping; return balancedness factor."""
    num_layers, num_logical_experts = workload.shape
    num_physical_experts = NUM_REPLICAS
    total_physical_load = torch.zeros(
        num_layers, num_physical_experts, dtype=torch.float, device=workload.device
    )

    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            logical_load = workload[layer_id][logical_id].item()
            if logical_load <= 0:
                continue
            num_replicas = int(logcnt[layer_id][logical_id].item())
            if num_replicas <= 0:
                continue
            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
            replica_load = logical_load / num_replicas
            total_physical_load[layer_id, physical_ids] += replica_load

    total_load = total_physical_load.sum()
    if total_load == 0:
        return 0.0

    layer_avg = total_physical_load.mean(dim=1)
    layer_max = total_physical_load.max(dim=1).values
    avg_load = layer_avg.sum().item()
    max_load = layer_max.sum().item()
    balancedness = avg_load / max_load if max_load > 0 else 0.0

    return balancedness


def evaluate(program_path: str, workload_path: str) -> EvaluationResult:
    workloads = load_workloads(workload_path)

    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        assert spec is not None
        program = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(program)

        if not hasattr(program, "rebalance_experts"):
            return {
                "balancedness_score": 0.0,
                "speed_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing `rebalance_experts` function",
            }

        balancedness_scores = []
        times = []
        for i in range(len(workloads) - 1):
            start_time = time.perf_counter()
            _, log2phy, logcnt = program.rebalance_experts(
                workloads[i],
                NUM_REPLICAS,
                NUM_GROUPS,
                NUM_NODES,
                NUM_GPUS,
            )
            balancedness_score = simulate_inference(log2phy, logcnt, workloads[i + 1])
            end_time = time.perf_counter()
            balancedness_scores.append(balancedness_score)
            times.append(end_time - start_time)

        avg_balancedness_score = sum(balancedness_scores) / len(balancedness_scores)
        avg_time = sum(times) / len(times)
        speed_score = 0.02 / avg_time
        combined_score = (avg_balancedness_score + speed_score) / 2
        return {
            "balancedness_score": float(avg_balancedness_score),
            "speed_score": float(speed_score),
            "combined_score": float(combined_score),
        }
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {
            "balancedness_score": 0.0,
            "speed_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate EPLB candidate.")
    parser.add_argument("--candidate_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--workload_path",
        type=str,
        default=None,
        help="Path to workload file; defaults to data_path/expert-load.json",
    )
    args = parser.parse_args()

    workload_path = args.workload_path
    if workload_path is None:
        workload_path = os.path.join(args.data_path, "expert-load.json")

    if not os.path.exists(workload_path):
        print(
            f"Workload file not found: {workload_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    result = evaluate(args.candidate_path, workload_path)

    if "error" in result:
        print(f"Error during evaluation: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Parseable output for eval_utils.parse_eval_results (matches llmsr/kernel_bench style)
    print(f"Candidate: {result}")


if __name__ == "__main__":
    main()
