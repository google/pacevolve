#!/bin/bash
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

# --- Script to run parameter-golf GPT training ---
#
# Usage:
#   ./run.sh <conda_env> <data_path> <tokenizer_path> <nproc> <mode>
#
# Modes:
#   syntax  - Python syntax check only (no GPU required)
#   train   - Full training run via torchrun
#
# Example:
#   ./run.sh pgolf /workspace/data/fineweb10B_sp1024 /workspace/data/tokenizers/fineweb_1024_bpe.model 8 train

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <data_path> <tokenizer_path> <nproc> <mode>"
    echo "  mode: syntax | train"
    exit 1
fi

DATA_PATH=$2
TOKENIZER_PATH=$3
NPROC=$4
MODE=$5

export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TRAIN_SCRIPT="$SCRIPT_DIR/../src/train_gpt.py"

if [ "$MODE" = "syntax" ]; then
    python -m py_compile "$TRAIN_SCRIPT"
    exit $?
fi

nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

export DATA_PATH="$DATA_PATH"
export TOKENIZER_PATH="$TOKENIZER_PATH"

torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT"
