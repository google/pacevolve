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

import dataclasses
import subprocess
import time
import os
import signal # For os.killpg and signal constants
import logging 
from typing import Any, Tuple, Dict
import pickle
import yaml
import glob

logger = logging.getLogger("controller")

@dataclasses.dataclass
class CompilationConfig:
  target_file_path: str
  pip_path: str | None = None

@dataclasses.dataclass
class CompletedProcess:
  args: str
  returncode: int
  stdout: str
  stderr: str


def _call_shell_command(
  command: str,
  max_retries: int = 3,
  timeout: int = 600,
) -> CompletedProcess | None:
  logger.info(f"_call_shell_command: Preparing to run command: {command}")
  num_retries = 0
  while num_retries < max_retries:
    num_retries += 1
    logger.info(
      f"_call_shell_command: Attempt {num_retries} of {max_retries} "
      f"for command: {command}"
    )

    process_handle = None
    try:
      process_handle = subprocess.Popen(
          command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
          text=True, shell=True, start_new_session=True 
      )
      stdout, stderr = process_handle.communicate(timeout=timeout)
      returncode = process_handle.returncode
      logger.info(
        f"_call_shell_command: Command '{command}' done. RC: {returncode}"
      )
      if stdout: logger.debug(f"_call_shell_command: STDOUT:\n{stdout}")
      if stderr: logger.debug(f"_call_shell_command: STDERR:\n{stderr}")
      return CompletedProcess(args=command, returncode=returncode, stdout=stdout, stderr=stderr)

    except subprocess.TimeoutExpired as e:
      logger.warning(
        f"_call_shell_command: Command '{command}' timed out after {timeout} "
        f"seconds on attempt {num_retries}."
      )
      if e.stdout:
        logger.info(f"--- Partial STDOUT before timeout ---\n{e.stdout.strip()}")
      if e.stderr:
        logger.warning(f"--- Partial STDERR before timeout ---\n{e.stderr.strip()}")
        
      if process_handle and process_handle.poll() is None:
        logger.info(
          "_call_shell_command: Timeout: Terminating process group "
          f"{os.getpgid(process_handle.pid)} for command '{command}'."
        )
        try:
          os.killpg(os.getpgid(process_handle.pid), signal.SIGKILL)
        except ProcessLookupError:
          logger.info(
            "_call_shell_command: Process group for "
            f"{process_handle.pid} already terminated."
          )
        except Exception as e_kill:
          logger.error(
            "_call_shell_command: Exception during killpg for "
            f"{process_handle.pid}: {e_kill}"
          )
        finally:
          try: process_handle.wait(timeout=10)
          except subprocess.TimeoutExpired:
            logger.warning(
              f"_call_shell_command: Process {process_handle.pid} did "
              "not terminate gracefully after SIGKILL and wait."
            )
          except Exception as e_wait:
            logger.error(
              "_call_shell_command: Exception during final wait for "
              f"{process_handle.pid}: {e_wait}"
            )

      if num_retries >= max_retries:
        logger.error(
          f"_call_shell_command: Command '{command}' timed out after all "
          f"{max_retries} retries. Giving up."
        )
        return None 
      logger.info("_call_shell_command: Retrying command...")
      time.sleep(1)

    except Exception as e:
      logger.error(
        f"_call_shell_command: Command '{command}' failed with "
        f"non-timeout exception: {e}"
      )
      return CompletedProcess(
        args=command,
        returncode=process_handle.returncode if process_handle else -1,
        stdout="",
        stderr=str(e),
      )
  return None


def parse_island_cuda_visible_devices(raw_value: Any) -> list[str] | None:
  """Normalizes island-to-GPU mappings from CLI/config input."""
  if raw_value is None:
    return None
  if isinstance(raw_value, str):
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values or None
  if isinstance(raw_value, (list, tuple)):
    values = [str(item).strip() for item in raw_value if str(item).strip()]
    return values or None
  raise ValueError(
    "Island GPU mapping must be a comma-separated string or a list of device ids."
  )


def resolve_cuda_visible_devices_for_island(
  config: dict,
  island_id: int,
) -> str | None:
  """Returns the CUDA-visible device string assigned to one island."""
  evaluation_config = config.get("evaluation", {})
  island_gpu_map = parse_island_cuda_visible_devices(
    evaluation_config.get("island_cuda_visible_devices")
  )
  if island_gpu_map is None:
    return None
  if island_id < 0 or island_id >= len(island_gpu_map):
    raise ValueError(
      f"No GPU mapping configured for island {island_id}. "
      f"Configured mappings: {island_gpu_map}"
    )
  return island_gpu_map[island_id]


def _check_dominance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> int:
    """Compares two points for Pareto dominance, assuming maximization for all objectives."""
    p1_better, p2_better = False, False
    for i in range(len(p1)):
        if p1[i] > p2[i]:
            p1_better = True
        elif p2[i] > p1[i]:
            p2_better = True
    if p1_better and not p2_better: return 1
    elif p2_better and not p1_better: return -1
    elif not p1_better and not p2_better: return 0
    else: return 2

def update_pareto_frontier(
    new_point: Tuple[float, ...],
    sol: str,
    pareto_set: Dict[Tuple[float, ...], str],
) -> bool:
    """
    Updates a Pareto frontier set with a new point, assuming maximization.
    Uses solution length as a tie-breaker for equal points.
    """
    points_to_remove = []

    for existing_point, existing_sol in pareto_set.items():
        dominance_status = _check_dominance(new_point, existing_point)

        if dominance_status == -1:
            return False
            
        elif dominance_status == 1:
            points_to_remove.append(existing_point)
            
        elif dominance_status == 0:
            if len(sol) < len(existing_sol):
                points_to_remove.append(existing_point)
            else:
                return False

    for point in points_to_remove:
        del pareto_set[point]

    pareto_set[new_point] = sol
    return True


def build_global_pareto_from_config(config_path: str, sol: str, pareto_set: Dict[Tuple[float, ...], str]) -> Dict[Tuple[float, ...], str]:
    """
    Loads a YAML config and builds a single, combined Pareto frontier from all datasets.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A single dictionary representing the global Pareto set.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading or parsing config file {config_path}: {e}")
        return {}
    
    base_results_path = config.get('paths', {}).get('results_path', '.')
    eval_configs = config.get('evaluation', {}).get('eval_configs', [])

    print(f"Building a single Pareto frontier from all datasets in {config_path}...")
    
    global_sol = []
    for dataset_config in eval_configs:
        dataset_name = dataset_config.get('dataset')
        if not dataset_name:
            continue
        
        print(f"--- Processing results from dataset: {dataset_name} ---")
        pkl_file_path = glob.glob(f"{base_results_path}/{dataset_name}.pkl")

        try:
            with open(pkl_file_path, 'rb') as file:
                data = pickle.load(file)
                global_sol.extend(data)
                print(f"global sol is {global_sol}")
        except Exception as e:
            print(f"Error processing file {pkl_file_path}: {e}")

    update_pareto_frontier(tuple(global_sol), sol, pareto_set)

    return pareto_set
