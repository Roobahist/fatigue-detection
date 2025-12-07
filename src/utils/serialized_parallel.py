import logging
import math
import multiprocessing
import os
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm


def wrapped_worker_function(params_and_worker):
    """Wrapper to safely execute worker function with error context."""
    worker_function, params, task_index = params_and_worker
    try:
        return worker_function(**params)
    except Exception as e:
        logging.error(f"Error in task {task_index}: {e}")
        raise


def get_task_slice(total_items: int):
    """
    Determine the slice of work for this SLURM task.
    
    Returns:
        tuple: (start_index, end_index, task_id, total_tasks)
    """
    # Case 1: SLURM job array
    tid = os.getenv("SLURM_ARRAY_TASK_ID")
    tcount = os.getenv("SLURM_ARRAY_TASK_COUNT")
    if tid is not None and tcount is not None:
        task_id = int(tid)
        total_tasks = int(tcount)
    else:
        # Case 2: multi-node with ntasks
        procid = os.getenv("SLURM_PROCID")
        ntasks = os.getenv("SLURM_NTASKS")
        if procid is not None and ntasks is not None:
            task_id = int(procid)
            total_tasks = int(ntasks)
        else:
            # Local fallback: single task
            return 0, total_items, 0, 1

    # Slice the workload
    chunk_size = math.ceil(total_items / total_tasks)
    start = task_id * chunk_size
    end = min(start + chunk_size, total_items)
    return start, end, task_id, total_tasks


def process(
    worker_function: Callable,
    worker_params_list: List[Dict[str, Any]],
    handler_function: Optional[Callable[[Any], Any]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    maxtasksperchild: Optional[int] = None,
    use_tqdm: bool = False,
    auto_slice: bool = True,
) -> List[Any]:
    """
    Process a list of tasks with optional automatic SLURM task slicing.

    Args:
        worker_function: Function to execute for each task
        worker_params_list: List of parameter dictionaries for worker_function
        handler_function: Optional function to process results. Can accept either
                         a single positional argument or **kwargs if result is a dict
        parallel: Whether to use multiprocessing
        max_workers: Maximum number of worker processes (defaults to SLURM_CPUS_PER_TASK or 1)
        maxtasksperchild: Maximum tasks per child process before recycling
        use_tqdm: Whether to show progress bar
        auto_slice: Whether to automatically slice work for SLURM distributed execution.
                   When True and in a SLURM environment with multiple tasks, this function
                   will only process a subset of worker_params_list. You'll need to aggregate
                   results across all SLURM tasks externally.

    Returns:
        List of results from processing. If auto_slice=True in a distributed SLURM
        environment, returns only the results for this task's slice of work.
    """

    # Automatically slice the work if running in SLURM environment
    original_list_length = len(worker_params_list)
    if auto_slice:
        start, end, task_id, total_tasks = get_task_slice(len(worker_params_list))
        if total_tasks > 1:  # Only slice if we're actually in a distributed environment
            logging.info(
                f"[SLURM] Task {task_id}/{total_tasks} → processing items {start}:{end} "
                f"(of {original_list_length} total items)"
            )
            logging.warning(
                f"[SLURM] auto_slice=True: This task will return only {end - start} results. "
                f"You must aggregate results from all {total_tasks} tasks externally."
            )
            worker_params_list = worker_params_list[start:end]

    results = []
    if parallel:
        logging.info("Processing in Parallel Mode ...")
        
        # Safely set multiprocessing start method
        try:
            multiprocessing.set_start_method("spawn", force=False)
        except RuntimeError:
            # Start method already set, continue with existing method
            pass

        # Determine number of workers
        if max_workers is None:
            max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        
        with multiprocessing.Pool(
            processes=max_workers,
            maxtasksperchild=maxtasksperchild,
        ) as pool:
            tasks = [
                (worker_function, params, idx) 
                for idx, params in enumerate(worker_params_list)
            ]
            for result in tqdm(
                pool.imap_unordered(wrapped_worker_function, tasks),
                total=len(worker_params_list),
                disable=not use_tqdm,
            ):
                if handler_function:
                    processed_result = _apply_handler(handler_function, result)
                    if processed_result is not None:
                        results.append(processed_result)
                else:
                    results.append(result)
    else:
        logging.info("Processing in Serialized Mode ...")
        for idx, params in enumerate(tqdm(worker_params_list, disable=not use_tqdm)):
            try:
                result = worker_function(**params)
                if handler_function:
                    processed_result = _apply_handler(handler_function, result)
                    if processed_result is not None:
                        results.append(processed_result)
                else:
                    results.append(result)
            except Exception as e:
                logging.error(f"Error in task {idx}: {e}")
                raise

    return results


def _apply_handler(handler_function: Callable, result: Any) -> Any:
    """
    Apply handler function to result with backward compatibility.
    
    Tries to unpack result as **kwargs if it's a dict, otherwise passes as single argument.
    """
    if isinstance(result, dict):
        try:
            # Try unpacking as kwargs (backward compatible behavior)
            return handler_function(**result)
        except TypeError:
            # Handler doesn't accept **kwargs, try single argument
            return handler_function(result)
    else:
        # Non-dict result, pass as single argument
        return handler_function(result)