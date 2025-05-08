import numpy as np
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
from functools import partial

from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

from typing import Callable, Generator, Optional


class LossIsGoodEnough(Exception):
    pass


def reduced_chi_squared(predicted: np.ndarray[float, ...], observed: np.ndarray[float], p: int) -> float:
    df = len(observed) - p
    return (1 / df) * np.sum(((observed - predicted) ** 2) / np.var(observed))


def residual(
        params: [float, ...],
        voxel: np.ndarray[float],
        *,
        model: Callable[[float, ...], float] = None,
        score_function: Callable[[np.ndarray[float, ...], np.ndarray[float, ...], int], float] = None,
        loss_thresh: float = 1e-3) -> np.ndarray[float]:
    """Thin wrapper around the model to calculate residual and throw stop exception at loss threshold."""
    pred = model(*params)
    res = pred - voxel
    if np.dot(res, res) <= loss_thresh:
        if score_function is not None:
            score = score_function(pred, voxel, p=len(params))
        else:
            score = 0.0
        raise LossIsGoodEnough(params, score)
    return res


def make_residual(**kwargs) -> Callable[[float, ...], float]:
    return partial(residual, **kwargs)


def jacobian(
        params: np.ndarray[float, ...],
        voxel: np.ndarray[float, ...],
        *,
        model: Callable[[float, ...], float],
        eps: float = 1e-4) -> np.ndarray[float]:
    """Calculate a simple Jacobian approximation by modelling eps increments on each parameter."""
    residual_function = make_residual(voxel=voxel, model=model, loss_thresh=-1)
    r0 = residual_function(params)
    rn = np.zeros((len(params), *r0.shape), dtype=r0.dtype)
    for i, p in enumerate(params):
        e = np.zeros_like(params)
        e[i] = eps
        rn[i] = residual_function(params + e)
    jac = (rn - r0[np.newaxis, ...]) / eps
    return jac.T


def make_jacobian(**kwargs) -> Callable[[np.ndarray[float, ...]], np.ndarray[float]]:
    return partial(jacobian, **kwargs)


def fit_voxel(
    voxel,
    model: Callable[[float, ...], float],
    *,
    x0: list[float, ...],
    loss_thresh: float = 1e-3,
    score_function: Callable[[np.ndarray[float, ...], np.ndarray[float, ...], int], float] = reduced_chi_squared,
    eps = 1e-4,
    **kwargs) -> tuple[np.ndarray[float, ...], np.ndarray[float, ...]]:
    """Perform least squares fit on a single voxel. loss threshold is used to calculate a ftol that crosses that loss."""

    residual_function = make_residual(voxel=voxel, model=model, loss_thresh=loss_thresh, score_function=score_function)
    jacobian_function = make_jacobian(voxel=voxel, model=model, eps=eps)
    if loss_thresh is not None and loss_thresh > 0:
        kwargs['gtol'] = 1e-15
        kwargs['xtol'] = 1e-15
        kwargs['ftol'] = 1e-15
    try:
        _ = least_squares(residual_function, x0, jac=jacobian_function, **kwargs)
    except LossIsGoodEnough as exc:
        params = exc.args[0]
        score = exc.args[1]
    else:
        params = np.array([np.nan] * len(x0))
        score = 1e6
    return params, score


def volume_iter(
    volume: np.ndarray[float],
) -> Generator[tuple[int, int, np.ndarray[float]]]:
    """Yields a single voxel (H x W) from an image colume shape Ch x H x W, ignoring nan pixels"""
    for y, x in product(range(volume.shape[1]), range(volume.shape[2])):
        if np.any(np.isnan(volume[:, y, x]), axis=0):
            continue
        yield y, x, volume[:, y, x]


def make_voxel_mapper(
        t: list[int, int, np.ndarray[float, ...]],
        **kwargs) -> tuple[int, int, np.ndarray[float]]:
    return t[0], t[1], fit_voxel(t[2], **kwargs)


def fit_volume(
    volume: np.ndarray[float],
    n_workers: Optional[int] = None,
    x0: np.ndarray[float, ...] = None,
    **kwargs) -> tuple[np.ndarray[float, ...], np.ndarray[float,...]]:

    # Get n_workers and calculate chunksize (Enough for ~2 chunks per worker)
    n_workers = n_workers or mp.cpu_count() // 2
    chunksize = (np.count_nonzero(~np.any(np.isnan(volume), axis=0)) / n_workers) // 2

    param_image = np.zeros((len(x0), *volume.shape[1:]), dtype=np.float32)
    score_image = np.zeros(volume.shape[1:], dtype=np.float32)
    voxel_mapper = partial(make_voxel_mapper, x0=x0, **kwargs)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for y, x, (params, score) in tqdm(executor.map(
            voxel_mapper,  # -> y, x, (params, score)
            volume_iter(volume),
            chunksize=int(chunksize),
        )):
            param_image[:, y, x] = params
            score_image[x, y] = score

    return param_image, score
