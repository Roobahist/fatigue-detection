import logging
import multiprocessing
import os, math

from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm


def wrapped_worker_function(params_and_worker):
    worker_function, params = params_and_worker
    return worker_function(**params)


def get_task_slice(total_items: int):
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
    worker_params_list: List[Dict],
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
        handler_function: Optional function to process results
        parallel: Whether to use multiprocessing
        max_workers: Maximum number of worker processes
        maxtasksperchild: Maximum tasks per child process
        use_tqdm: Whether to show progress bar
        auto_slice: Whether to automatically slice work for SLURM distributed execution
    
    Returns:
        List of results from processing
    """
    
    # Automatically slice the work if running in SLURM environment
    if auto_slice:
        start, end, task_id, total_tasks = get_task_slice(len(worker_params_list))
        if total_tasks > 1:  # Only slice if we're actually in a distributed environment
            print(f"[SLURM] Task {task_id}/{total_tasks} → processing items {start}:{end}")
            worker_params_list = worker_params_list[start:end]

    results = []
    if parallel:
        logging.info("Processing in Parallel Mode ...")
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool(
            processes=max_workers, maxtasksperchild=maxtasksperchild
        ) as pool:
            tasks = [(worker_function, params) for params in worker_params_list]
            for result in tqdm(
                pool.imap_unordered(wrapped_worker_function, tasks),
                total=len(worker_params_list),
                disable=not use_tqdm,
            ):  
                if handler_function:
                    processed_result = handler_function(**result)
                    if processed_result is not None:
                        results.append(processed_result)
                else:
                    results.append(result)
    else:
        logging.info("Processing in Serialized Mode ...")
        for params in tqdm(worker_params_list, disable=not use_tqdm):
            result = worker_function(**params)
            if handler_function:
                processed_result = handler_function(**result)
                if processed_result is not None:
                    results.append(processed_result)
            else:
                results.append(result)

    return results