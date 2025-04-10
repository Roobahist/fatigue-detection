import logging
import multiprocessing
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm


def process(
    handler_function: Callable[[Any], Any],
    worker_function: Callable,
    worker_params_list: List[Dict],
    parallel: bool = True,
    max_workers: Optional[int] = None,
    maxtasksperchild: Optional[int] = 1,
    use_tqdm: bool = False,
) -> List[Any]:

    results = []
    if parallel:
        logging.info("Processing in Parallel Mode ...")
        with multiprocessing.Pool(
            processes=max_workers, maxtasksperchild=maxtasksperchild
        ) as pool:
            for result in tqdm(
                pool.imap(worker_function, worker_params_list),
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


# def process(
#     worker_function: Callable,
#     worker_params_list: List[Dict],
#     parallel: bool = True,
#     max_workers: Optional[int] = None,
#     maxtasksperchild: Optional[int] = 1,
#     use_tqdm: bool = False,
# ) -> List[Any]:

#     if parallel:
#         logging.info("Processing in Parallel Mode ...")
#         with multiprocessing.Pool(
#             processes=max_workers, maxtasksperchild=maxtasksperchild
#         ) as pool:
#             if use_tqdm:
#                 results = list(tqdm(pool.imap(worker_function, worker_params_list), total=len(worker_params_list)))
#             else:
#                 results = pool.map(worker_function, worker_params_list)
#     else:
#         logging.info("Processing in Serialized Mode ...")
#         results = []
#         if use_tqdm:
#             for params in tqdm(worker_params_list):
#                 results.append(worker_function(**params))
#         else:
#             for params in worker_params_list:
#                 results.append(worker_function(**params))
#     return results
