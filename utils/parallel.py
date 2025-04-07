# ct2fea/utils/parallel.py
import multiprocessing as mp
import numpy as np
from typing import Callable, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def parallel_map_blocks(
    func: Callable,
    data: np.ndarray,
    n_jobs: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """Apply function to 3D volume in parallel across blocks

    Args:
        func: Function to apply to each block
        data: Input 3D numpy array
        n_jobs: Number of parallel jobs (None = use all cores)
        chunk_size: Size of chunks to process (None = auto-determine)

    Returns:
        Processed array
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    if chunk_size is None:
        chunk_size = max(1, data.shape[0] // (n_jobs * 2))

    # Split array into chunks
    chunks = []
    for i in range(0, data.shape[0], chunk_size):
        end = min(i + chunk_size, data.shape[0])
        chunks.append(data[i:end])

    # Process chunks in parallel
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(func, chunks)

    # Combine results
    return np.concatenate(results)


def parallel_map_slices(
    func: Callable,
    data: np.ndarray,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Apply function to 3D volume in parallel across slices

    Args:
        func: Function to apply to each slice
        data: Input 3D numpy array
        n_jobs: Number of parallel jobs (None = use all cores)

    Returns:
        Processed array
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    # Process slices in parallel
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(func, [data[i] for i in range(data.shape[0])])

    # Combine results
    return np.stack(results)
