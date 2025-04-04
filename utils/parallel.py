# ct2fea/utils/parallel.py
import multiprocessing as mp
from functools import partial
import numpy as np
from typing import Callable, Any, List, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

logger = logging.getLogger(__name__)


def get_optimal_chunk_size(total_size: int, n_jobs: int) -> int:
    """Calculate optimal chunk size for parallel processing"""
    chunk_size = max(1, total_size // (n_jobs * 4))
    return min(chunk_size, 1000)  # Cap at 1000 to avoid memory issues


def parallel_map_blocks(
    func: Callable,
    data: np.ndarray,
    n_jobs: Optional[int] = None,
    chunk_size: Optional[int] = None,
    use_threads: bool = False,
) -> np.ndarray:
    """Apply function to 3D volume in parallel across blocks

    Args:
        func: Function to apply to each block
        data: Input 3D numpy array
        n_jobs: Number of parallel jobs (None = use all cores)
        chunk_size: Size of chunks to process (None = auto-determine)
        use_threads: Use threads instead of processes (better for I/O bound tasks)

    Returns:
        Processed array
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(data.shape[0], n_jobs)

    # Split array into chunks
    chunks = []
    for i in range(0, data.shape[0], chunk_size):
        end = min(i + chunk_size, data.shape[0])
        chunks.append(data[i:end])

    # Process chunks in parallel
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor_class(max_workers=n_jobs) as executor:
        results = list(executor.map(func, chunks))

    # Combine results
    return np.concatenate(results)


def parallel_map_slices(
    func: Callable,
    data: np.ndarray,
    n_jobs: Optional[int] = None,
    use_threads: bool = False,
) -> np.ndarray:
    """Apply function to 3D volume in parallel across slices

    Args:
        func: Function to apply to each slice
        data: Input 3D numpy array
        n_jobs: Number of parallel jobs (None = use all cores)
        use_threads: Use threads instead of processes (better for I/O bound tasks)

    Returns:
        Processed array
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    # Process slices in parallel
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor_class(max_workers=n_jobs) as executor:
        results = list(executor.map(func, [data[i] for i in range(data.shape[0])]))

    # Combine results
    return np.stack(results)


def distribute_task(
    items: List[Any],
    task_func: Callable,
    n_jobs: Optional[int] = None,
    use_threads: bool = False,
) -> List[Any]:
    """Distribute a general task across multiple processes/threads

    Args:
        items: List of items to process
        task_func: Function to apply to each item
        n_jobs: Number of parallel jobs (None = use all cores)
        use_threads: Use threads instead of processes

    Returns:
        List of processed items
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor_class(max_workers=n_jobs) as executor:
        return list(executor.map(task_func, items))
