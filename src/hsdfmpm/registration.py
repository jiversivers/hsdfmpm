import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from skimage.transform import warp, AffineTransform
from skimage.metrics import normalized_mutual_information
from scipy.optimize import minimize
from numpy.typing import NDArray

def register_images_crosscorr(
    fixed_img: NDArray[float], moving_img: NDArray[float]
) -> tuple[NDArray[float], tuple[float, float]]:
    """
    Register two images using phase cross-correlation.

    :param fixed_img: Reference image
    :type fixed_img: NDArray[float]
    :param moving_img: Image to be registered
    :type moving_img: NDArray[float]

    :returns:The registered moving image and the shift tuple (y, x) to register the images (with :py:func:`scipy.ndimage.shift`).
    :rtype: tuple[NDArray[float], tuple[int, int]]
    """
    shift_px, error, diffphase = phase_cross_correlation(fixed_img, moving_img)
    registered_img = shift(moving_img, shift_px)

    return registered_img, shift_px


def register_images_mutual_info(
    fixed_img: NDArray[float], moving_img: NDArray[float]
) -> tuple[NDArray[float], NDArray[float]]:
    """
    Register two images using mutual information optimization.

    :param fixed_img: Reference image
    :type fixed_img: NDArray[float]
    :param moving_img: Image to be registered
    :type moving_img: NDArray[float]

    :returns:The registered moving image and the warp tuple to register the images with :py:func:`scipy.ndimage.warp`
    :rtype: tuple[NDArray[float], NDArray[float]]
    """

    def mutual_info_shift(shift):
        transform = AffineTransform(translation=shift)
        shifted_img = warp(moving_img, transform.inverse)
        return -normalized_mutual_information(fixed_img, shifted_img)

    initial_shift = [0, 0]
    result = minimize(mutual_info_shift, initial_shift, method="Powell")

    optimal_shift_px = result.x
    transform_optimal = AffineTransform(translation=optimal_shift_px)
    registered_img = warp(moving_img, transform_optimal.inverse)

    return registered_img, transform_optimal.inverse


def display_overlay(
    fixed_img: NDArray[float], moving_img: NDArray[float], alpha: float = 0.5
) -> None:
    """
    Display the overlay of two registered images.

    Parameters:
    - orr_img: numpy.ndarray, reference image
    - registered_img: numpy.ndarray, registered image
    - alpha: float, transparency level for overlay
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(fixed_img, cmap="gray", interpolation="none")
    plt.imshow(moving_img, cmap="jet", interpolation="none", alpha=alpha)
    plt.title("Overlay of Registered Images")
    plt.axis("off")
    plt.show()
