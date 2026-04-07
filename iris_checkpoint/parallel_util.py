"""CPU parallelism helpers (ProcessPool for stats; thread count helpers)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def worker_cap(default_max: int = 8) -> int:
    n = os.cpu_count() or 4
    return max(1, min(default_max, n))


def parallel_map_process(
    fn: Callable[[T], R],
    items: list[T],
    max_workers: int | None = None,
    chunksize: int = 8,
) -> list[R]:
    """ProcessPoolExecutor map with sane chunksize for many small tasks."""
    if not items:
        return []
    w = max_workers or worker_cap()
    if w == 1 or len(items) < 16:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=w) as ex:
        return list(ex.map(fn, items, chunksize=max(1, len(items) // (w * 4))))


def thread_map(fn: Callable[[T], R], items: list[T], max_workers: int | None = None) -> list[R]:
    """ThreadPoolExecutor for I/O-bound work (e.g. PIL loads)."""
    if not items:
        return []
    w = max_workers or worker_cap()
    if w == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(fn, items))
