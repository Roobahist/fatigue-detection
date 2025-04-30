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



# import logging
# import multiprocessing
# from typing import Any, Callable, Dict, List, Optional, Union
# from functools import partial
# from contextlib import contextmanager

# from tqdm import tqdm


# def _worker_wrapper(params_with_func):
#     """Worker wrapper that handles exceptions and captures worker results."""
#     worker_function, params = params_with_func
#     try:
#         return {"status": "success", "result": worker_function(**params), "error": None}
#     except Exception as e:
#         return {"status": "error", "result": None, "error": str(e)}


# @contextmanager
# def get_pool(parallel: bool, max_workers: Optional[int] = None, 
#              maxtasksperchild: Optional[int] = None, 
#              start_method: Optional[str] = None):
#     """Context manager for handling pool creation and cleanup."""
#     if parallel and start_method:
#         # Only set start method if it hasn't been set already
#         try:
#             multiprocessing.get_context(start_method)
#         except ValueError:
#             multiprocessing.set_start_method(start_method, force=True)
    
#     if parallel:
#         pool = multiprocessing.Pool(processes=max_workers, maxtasksperchild=maxtasksperchild)
#         try:
#             yield pool
#         finally:
#             pool.close()
#             pool.join()
#     else:
#         yield None


# def process(
#     worker_function: Callable,
#     handler_function: Callable[[Any], Any],
#     worker_params_list: List[Dict],
#     parallel: bool = True,
#     max_workers: Optional[int] = None,
#     maxtasksperchild: Optional[int] = 100,  # Better default
#     use_tqdm: bool = False,
#     start_method: Optional[str] = 'spawn',
#     error_handler: Optional[Callable] = None,
#     chunk_size: Optional[int] = 1,
# ) -> Dict[str, Union[List[Any], List[Exception]]]:
#     """
#     Process a list of parameters with worker and handler functions.
    
#     Args:
#         worker_function: Function to execute for each parameter set
#         handler_function: Function to process worker results
#         worker_params_list: List of parameter dictionaries for worker
#         parallel: Whether to process in parallel
#         max_workers: Maximum number of worker processes
#         maxtasksperchild: Number of tasks per child process
#         use_tqdm: Whether to show progress bar
#         start_method: Multiprocessing start method ('spawn', 'fork', 'forkserver')
#         error_handler: Optional function to handle errors
#         chunk_size: Chunk size for imap_unordered
        
#     Returns:
#         Dictionary with processed results and any errors
#     """
#     results = []
#     errors = []
    
#     # Skip empty parameter lists
#     if not worker_params_list:
#         return {"results": results, "errors": errors}
    
#     # Default error handler if none provided
#     if error_handler is None:
#         error_handler = lambda e, params: logging.error(f"Error processing {params}: {e}")
    
#     if parallel and len(worker_params_list) > 1:
#         logging.info(f"Processing {len(worker_params_list)} tasks in Parallel Mode...")
        
#         with get_pool(parallel, max_workers, maxtasksperchild, start_method) as pool:
#             if pool is None:  # Fallback to serial if pool creation failed
#                 parallel = False
#             else:
#                 tasks = [(worker_function, params) for params in worker_params_list]
#                 iterator = pool.imap_unordered(_worker_wrapper, tasks, chunksize=chunk_size)
                
#                 for response in tqdm(iterator, total=len(worker_params_list), disable=not use_tqdm):
#                     if response["status"] == "success":
#                         try:
#                             processed_result = handler_function(**response["result"])
#                             if processed_result is not None:
#                                 results.append(processed_result)
#                         except Exception as e:
#                             error_handler(e, "handler_function")
#                             errors.append(e)
#                     else:
#                         error_handler(response["error"], "worker_function")
#                         errors.append(response["error"])
    
#     # Fallback to serial processing
#     if not parallel or len(worker_params_list) <= 1:
#         logging.info(f"Processing {len(worker_params_list)} tasks in Serial Mode...")
#         for params in tqdm(worker_params_list, disable=not use_tqdm):
#             try:
#                 result = worker_function(**params)
#                 try:
#                     processed_result = handler_function(**result)
#                     if processed_result is not None:
#                         results.append(processed_result)
#                 except Exception as e:
#                     error_handler(e, {"params": params, "stage": "handler"})
#                     errors.append(e)
#             except Exception as e:
#                 error_handler(e, {"params": params, "stage": "worker"})
#                 errors.append(e)
    
#     return {"results": results, "errors": errors}