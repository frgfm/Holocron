# Copyright (C) 2019-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import multiprocessing as mp
from math import sqrt
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

Inp = TypeVar("Inp")
Out = TypeVar("Out")


__all__ = ["parallel", "find_image_size"]


def parallel(
    func: Callable[[Inp], Out],
    arr: Sequence[Inp],
    num_threads: Optional[int] = None,
    progress: bool = False,
    **kwargs: Any,
) -> Sequence[Out]:
    """Performs parallel tasks by leveraging multi-threading.

    >>> from holocron.utils.misc import parallel
    >>> parallel(lambda x: x ** 2, list(range(10)))

    Args:
        func: function to be executed on multiple workers
        arr: function argument's values
        num_threads: number of workers to be used for multiprocessing
        progress: whether the progress bar should be displayed
        kwargs: keyword arguments of tqdm

    Returns:
        list: list of function's results
    """
    num_threads = num_threads if isinstance(num_threads, int) else min(16, mp.cpu_count())
    if num_threads < 2:
        if progress:
            results = list(map(func, tqdm(arr, total=len(arr), **kwargs)))
        else:
            results = map(func, arr)  # type: ignore[assignment]
    else:
        with ThreadPool(num_threads) as tp:
            if progress:
                results = list(tqdm(tp.imap(func, arr), total=len(arr), **kwargs))
            else:
                results = tp.map(func, arr)

    return results


def find_image_size(dataset: Sequence[Tuple[Image.Image, Any]], **kwargs: Any) -> None:
    """Computes the best image size target for a given set of images

    Args:
        dataset: an iterator yielding a PIL Image and a target object
        kwargs: keyword args of matplotlib.pyplot.show

    Returns:
        the suggested height and width to be used
    """
    # Record height & width
    _shapes = parallel(lambda x: x[0].size, dataset, progress=True)

    shapes = np.asarray(_shapes)[:, ::-1]
    ratios = shapes[:, 0] / shapes[:, 1]
    sides = np.sqrt(shapes[:, 0] * shapes[:, 1])

    # Compute median aspect ratio & side
    median_ratio = np.median(ratios)
    median_side = np.median(sides)

    height = int(round(median_side * sqrt(median_ratio)))
    width = int(round(median_side / sqrt(median_ratio)))

    # Double histogram
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(ratios, bins=30, alpha=0.7)
    axes[0].title.set_text(f"Aspect ratio (median: {median_ratio:.2})")
    axes[0].grid(True, linestyle="--", axis="x")
    axes[0].axvline(median_ratio, color="r")
    axes[1].hist(sides, bins=30, alpha=0.7)
    axes[1].title.set_text(f"Side (median: {int(median_side)})")
    axes[1].grid(True, linestyle="--", axis="x")
    axes[1].axvline(median_side, color="r")
    fig.suptitle(f"Median image size: ({height}, {width})")
    plt.show(**kwargs)
