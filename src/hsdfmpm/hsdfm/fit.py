import numpy as np
from itertools import product
from functools import partial

from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

from typing import Callable, Generator, Optional


class LossIsGoodEnough(Exception):
    pass


def residual(
        params: [float, ...],
        voxel: np.ndarray[float],
        *,
        model: Callable[[float, float], float] = None,
        loss_thresh: float = 1e-3) -> np.ndarray[float]:
    """Thin wrapper around the model to calculate residual and throw stop exception at loss threshold."""
    r = model(*params) - voxel
    if np.dot(r, r) <= loss_thresh:
        raise LossIsGoodEnough(params)
    return r


def make_residual(
        voxel: np.ndarray[float],
        model: Callable[[float, float], float] = None,
        loss_thresh: float = -1) -> Callable[[float, ...], float]:
    return partial(residual, voxel=voxel, model=model, loss_thresh=loss_thresh)


def jacobian(
        params: np.ndarray[float, ...],
        voxel: np.ndarray[float, ...],
        *,
        model: Callable[[float, ...], float],
        eps: float = 1e-4) -> np.ndarray[float]:
    """Calculate a simple Jacobian approximation by modelling eps increments on each parameter."""
    residual_funciton = make_residual(voxel=voxel, model=model)
    r0 = residual_funciton(params)
    rn = np.zeros((len(params), *r0.shape), dtype=r0.dtype)
    for i, p in enumerate(params):
        e = np.zeros_like(params)
        e[i] = eps
        rn[i] = residual_funciton(params + e)
    jac = (rn - r0[np.newaxis, ...]) / eps
    return jac.T


def make_jacobian(
        voxel: np.ndarray[float, ...],
        model: Callable[[float, ...], float],
        eps: float = 1e-4) -> Callable[[np.ndarray[float, ...]], np.ndarray[float]]:
    return partial(jacobian, voxel=voxel, model=model, eps=eps)


def fit_voxel(
    voxel,
    model: Callable[[float, ...], float],
    *,
    x0: list[float, ...],
    loss_thresh: float = 1e-3,
    eps = 1e-4,
    **kwargs,
):
    """Perform least squares fit on a single voxel. loss threshold is used to calculate a ftol that crosses that loss."""

    residual_function = make_residual(voxel=voxel, model=model, loss_thresh=loss_thresh)
    jacobian_function = make_jacobian(voxel=voxel, model=residual_function, eps=eps)
    try:
        fit_result = least_squares(residual_function, x0, **kwargs)
    except LossIsGoodEnough as exc:
        params = exc.args[0]
        return params
    else:
        raise RuntimeError("Solver failed to reach loss threshold")


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
    **kwargs) -> np.ndarray[float]:

    # Get n_workers and calculate chunksize (Enough for ~2 chunks per worker)
    n_workers = n_workers or mp.cpu_count() // 2
    chunksize = (np.count_nonzero(~np.any(np.isnan(volume), axis=0)) / n_workers) // 2

    param_image = np.zeros((len(x0), *volume.shape[1:]), dtype=np.float32)
    voxel_mapper = partial(make_voxel_mapper, x0=x0, **kwargs)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for y, x, params in executor.map(
            voxel_mapper,  # -> y, x, params
            volume_iter(volume),
            chunksize=int(chunksize),
        ):
            param_image[:, y, x] = params

    return param_image
