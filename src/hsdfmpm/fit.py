from functools import partial
from typing import Callable, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from hsdfmpm.utils import ImageData
from skimage.restoration import estimate_sigma
from scipy.optimize import curve_fit
from scipy.ndimage import distance_transform_edt


def image_reduced_chi_sqaured(
    image: Union[ImageData, NDArray[float]],
    y_pred: NDArray[float],
    p: int,
    bin_factor: int = 1,
) -> float:
    r"""
    Calculate a reduced chi-squared for an image, using a wavelet estimation sigma (:py:func:`skimage.restoration.estimate_sigma`).

    :param image: The image that was fit (`y_data`)
    :type image: ImageData | NDArray[float]
    :param y_pred: The predicted y from the fitted model.
    :type y_pred: NDArray[float]
    :param p: The number of parameters for the fitted model.
    :type p: int
    :param bin_factor: The factor of bins the image was spatially binned by, assumes bin averageing. This scales the sigma estimate for the model.
    :return: The reduced chi-squared.
    :rtype: float
    """

    sigma = estimate_sigma(image) / np.sqrt(bin_factor)
    mask = ~(np.isnan(y_pred) | np.isnan(image))
    df = np.sum(mask) - p

    residual = (image - y_pred) / sigma
    chi_sq = np.sum(residual[mask] ** 2)

    return chi_sq / df


def fit_images_to_model(
    model: Callable[[float, ...], float],
    x_image: Union[ImageData, NDArray[float]],
    y_image: Union[ImageData, NDArray[float]],
    **kwargs,
) -> tuple(NDArray[float], NDArray[float], float):
    r"""
    A thin wrapper around the fitting routine, to fit images to a model.

    :param model: The model to fit of the signature `model_fn(x, *args)` where `x` is an element of `x_data`.
    :type model: Callable[[float, ...], float]
    :param x_image: The image to fit with.
    :type x_image: ImageData | NDArray[float]
    :param y_image: The image to fit to.
    :type y_image: ImageData | NDArray[float]
    :param kwargs: Additional keyword arguments to pass to the fitting routine.
    :return: A tuple of the parameters, the covariance matrix.
    :rtype: tuple[NDArray[float], NDArray[float]]
    """
    params, cov = curve_fit(model, x_image, y_image, **kwargs)

    return params, cov


def pO2_of_sO2(sO2: float, p50: float, n: float = 2.7) -> float:
    r"""
    Calculate the partial pressure of oxygen (pO₂) from oxygen saturation (sO₂) using the Hill equation.

    The relationship between oxygen saturation and partial pressure of oxygen is nonlinear and described
    by the Hill equation, which is a sigmoid function.

    :param sO2: Oxygen saturation as a fraction (0 to 1).
    :type sO2: float
    :param p50: Partial pressure of oxygen at which hemoglobin is 50% saturated (P₅₀).
    :type p50: float
    :param n: Hill coefficient, describing the cooperativity of oxygen binding. Default is 2.7.
    :type n: float, optional
    :returns: The partial pressure of oxygen (pO₂) corresponding to the given oxygen saturation.
    :rtype: float
    :raises ValueError: If sO₂ equals 1, to avoid division by zero.

    """
    return ((sO2 * p50 ** n) / (1 - sO2 ** n)) ** (1 / n)


def pO2_of_vascular_distance(d: float, diff_limit: float, pO2_0: float) -> float:
    r"""
    Calculate the partial pressure of oxygen (pO₂) at a given distance from the vasculature based on oxygen diffusion.

    This function assumes an exponential decay of oxygen partial pressure with distance from the blood vessel
    (i.e., the vessel surface), governed by the oxygen diffusion limit.

    :param d: The distance from the blood vessel (in micrometers, millimeters, etc.).
    :type d: float
    :param diff_limit: The diffusion limit, which is a measure of how far oxygen can diffuse from the vessel.
    :type diff_limit: float
    :param sO2_0: The oxygen saturation at the vessel surface (saturation at 0 distance).
    :type sO2_0: float
    :returns: The oxygen partial pressure (pO₂) at the given distance from the vessel.
    :rtype: float
    """
    return pO2_0 * np.exp(-d / diff_limit)


def nadh_of_pO2(pO2: float, A1: float, A2: float, p50: float, dx: float) -> float:
    r"""
    Calculate the NADH concentration based on partial pressure of oxygen (pO₂) using a sigmoid function.

    The function uses the Hill equation to model the relationship between oxygen partial pressure and NADH
    concentration, which is often used to represent cellular metabolic activity.

    :param pO2: The partial pressure of oxygen (pO₂).
    :type pO2: float
    :param A1: The scaling factor for the upper asymptote of the sigmoid.
    :type A1: float
    :param A2: The scaling factor for the lower asymptote of the sigmoid.
    :type A2: float
    :param p50: The partial pressure of oxygen at which NADH concentration is halfway between A1 and A2.
    :type p50: float
    :param dx: The steepness of the sigmoid curve.
    :type dx: float
    :returns: The NADH concentration based on the given oxygen partial pressure.
    :rtype: float
    """
    return A2 + (A1 + A2) / (1 + np.exp((pO2 - p50) / dx))


def orr_of_nadh(nadh: float, fad: float) -> float:
    r"""
    Calculate the optical redox ratio (ORR) based on NADH concentration.

    The ORR is calculated as the ratio of the mean FAD concentration to the sum of the mean FAD concentration
    and the NADH concentration.

    :param nadh: The NADH concentration.
    :type nadh: float
    :param fad: The FAD concentration.
    :type fad: float
    :returns: The optical redox ratio (ORR).
    :rtype: float
    """
    return fad / (fad + nadh)


def nadh_of_d(d: float, diff_limit: float, A1: float, A2: float, p50: float, dx: float, sO2_0: float) -> float:
    """
    Calculate the NADH concentration at a given vascular distance using oxygen saturation, distance,
    and other parameters.

    .. citation: Kasischke, K., "Two-photon NADH imaging exposes boundaries of oxygen diffusion in cortical vascular
     supply regions", *Journal of Cerebral Blood Flow and Metabolism: Official Journal of the International Society of
     Cerebral Blood Flow and Metabolism*, 2011

    This function chains together the calculation of oxygen partial pressure at a specific distance
    from the vasculature (based on oxygen saturation and diffusion distance), followed by the calculation
    of NADH concentration using the oxygen partial pressure.

    :param d: The distance from the blood vessel (in micrometers, millimeters, etc.).
    :type d: float
    :param diff_limit: The diffusion limit, a measure of how far oxygen can diffuse from the vessel.
    :type diff_limit: float
    :param A1: The scaling factor for the upper asymptote of the NADH concentration sigmoid.
    :type A1: float
    :param A2: The scaling factor for the lower asymptote of the NADH concentration sigmoid.
    :type A2: float
    :param p50: The partial pressure of oxygen at which NADH concentration is halfway between A1 and A2.
    :type p50: float
    :param dx: The steepness of the NADH concentration sigmoid curve.
    :type dx: float
    :returns: The NADH concentration at the given distance from the vasculature.
    :rtype: float
    :raises ValueError: If the calculated oxygen partial pressure results in an invalid value for the
                        subsequent calculations.
    """
    return nadh_of_pO2(
        pO2_of_vascular_distance(d, diff_limit, pO2_of_sO2(sO2_0, p50)),
        A1,
        A2,
        p50,
        dx,
    )


def orr_of_d(d: float, diff_limit: float, A1: float, A2: float, p50: float, dx: float, sO2_0, fad:float) -> float:
    r"""
    Calculate the optical redox ratio (ORR) based on vascular distance, oxygen saturation, and other parameters.

    This function chains together the calculations of oxygen partial pressure, NADH concentration, and ORR,
    incorporating the exponential decay of oxygen with distance from the blood vessel.

    :param d: The distance from the vasculature.
    :type d: float
    :param diff_limit: The diffusion limit, a measure of how far oxygen can diffuse from the vessel.
    :type diff_limit: float
    :param A1: The scaling factor for the upper asymptote of the NADH concentration sigmoid.
    :type A1: float
    :param A2: The scaling factor for the lower asymptote of the NADH concentration sigmoid.
    :type A2: float
    :param p50: The partial pressure of oxygen at which NADH concentration is halfway between A1 and A2.
    :type p50: float
    :param dx: The steepness of the NADH concentration sigmoid curve.
    :type dx: float
    :param so2: The oxygen saturation at the vessel surface (saturation at 0 distance).
    :type so2: np.ndarray or float
    :param fad: The FAD concentration.
    :type fad: float
    :returns: The optical redox ratio (ORR) at the given vascular distance.
    :rtype: float
    """
    return orr_of_nadh(
        nadh_of_pO2(
            pO2_of_vascular_distance(d, diff_limit, pO2_of_sO2(sO2_0, p50)),
            A1,
            A2,
            p50,
            dx,
        ),
        fad
    )

def make_model_for_vascular_distance(
        target: Literal["orr", "nadh"],
        sO2_0: float,
        fad: Optional[float] = None,
) -> Callable[[float, ...], float]:
    """A factory to create a model of either `ORR` or `NADH` intensity as a function of distance from vascualture, with :math:`sO_2(0)` fixed.

    :param target: The target variable to model.
    :type target: Literal["orr", "nadh"]
    :param sO2_0: The saturation concentration at distance of 0.
    :type sO2_0: float.
    :param fad: The FAD concentration, required for target of `ORR`.
    :type fad: float
    :returns: The model of either `ORR` or `NADH` intensity.
    :rtype: Callable[[float, ...], float]
    """
    if target == "orr":
        return partial(orr_of_d(sO2_0=sO2_0, fad=fad))
    elif target == "nadh":
        return partial(nadh_of_pO2(sO2_0=sO2_0))
    else:
        raise ValueError(f"No model for requested target, {target}.")
    

def distance_to_vasculature(
        vasc_mask: NDArray[bool], um_per_pixel: float = 1.0
) -> NDArray[float]:
    """Thin wrapper to crete a distance map from a boolean mask

    :param vasc_mask: The map of vascualture where vascualture is `True`.
    :type vasc_mask: NDArray[bool]
    :param um_per_pixel: The scale of the image in :math:`\frac{\mathrm{\mu m}}{\mathrm{pixel}}`.
    :type um_per_pixel: float
    :returns: The distance map from a boolean mask.
    :rtype: NDArray[float]
    """
    not_vascualture = ~vasc_mask
    return distance_transform_edt(not_vascualture.astype(np.float32)) * um_per_pixel
