import logging
import multiprocessing
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm


# Move the wrapped worker function outside
def wrapped_worker_function(params_and_worker):
    worker_function, params = params_and_worker
    return worker_function(**params)


def process(
    worker_function: Callable,
    handler_function: Callable[[Any], Any],
    worker_params_list: List[Dict],
    parallel: bool = True,
    max_workers: Optional[int] = None,
    maxtasksperchild: Optional[int] = 1,
    use_tqdm: bool = False,
) -> List[Any]:

    results = []
    if parallel and len(worker_params_list) > 1:
        logging.info("Processing in Parallel Mode ...")
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool(
            processes=max_workers, maxtasksperchild=maxtasksperchild
        ) as pool:
            # Pass both the worker function and parameters to the pool
            tasks = [(worker_function, params) for params in worker_params_list]
            for result in tqdm(
                pool.imap_unordered(wrapped_worker_function, tasks),
                total=len(worker_params_list),
                disable=not use_tqdm,
            ):
                processed_result = handler_function(**result)
                if processed_result is not None:
                    results.append(processed_result)
    else:
        logging.info("Processing in Serialized Mode ...")
        for params in tqdm(worker_params_list, disable=not use_tqdm):
            result = worker_function(**params)
            processed_result = handler_function(**result)
            if processed_result is not None:
                results.append(processed_result)
    return results
