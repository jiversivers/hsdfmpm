import itertools
import json
from pathlib import Path

import numpy as np

from numpy.lib._stride_tricks_impl import as_strided
from photon_canon.lut import LUT
from skimage.filters import gabor_kernel
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from enum import Enum
from typing import Callable, Union

from ..utils import vectorize_img, ensure_path, iterable_array


def read_metadata_json(file_path: Union[str, Path]) -> dict[str, list[float]]:
    grouped_metadata = {
        "AbsTime": [],
        "ExpTime": [],
        "Filter": [],
        "AvgInt": [],
        "Wavelength": [],
    }

    # Open and read the file contents
    try:
        with open(file_path, "r") as file:
            json_data = json.load(file)  # Directly load the JSON data from the file

        # Iterate through each entry and group values by field name
        for entry in json_data:
            grouped_metadata["AbsTime"].append(entry.get("AbsTime", None))
            grouped_metadata["ExpTime"].append(entry.get("ExpTime", None))
            grouped_metadata["Filter"].append(entry.get("Filter", None))
            grouped_metadata["AvgInt"].append(entry.get("AvgInt", None))
            grouped_metadata["Wavelength"].append(entry.get("Wavelength", None))

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return grouped_metadata


def normalize_integration_time(
    hyperstack: np.ndarray[float], integration_time: list[float]
) -> np.ndarray[float]:
    hyperstack /= np.array(integration_time)[:, np.newaxis, np.newaxis]
    return hyperstack


def normalize_to_standard(
    hyperstack: np.ndarray[float], standard: np.ndarray[float], bg: np.ndarray[float]
):
    return (hyperstack - bg) / (standard - bg)


def get_local_stdev(
    image: np.ndarray[float], shape: tuple[int, int]
) -> np.ndarray[float]:
    C, H, W = image.shape
    factor = np.asarray((H, W)) // shape
    new_shape = (C, shape[0], factor[0], shape[1], factor[1])
    new_strides = (
        image.strides[0],
        image.strides[1] * factor[0],
        image.strides[1],
        image.strides[2] * factor[1],
        image.strides[2],
    )

    blocks = as_strided(image, shape=new_shape, strides=new_strides)
    return np.nanstd(blocks, axis=(2, 4))


def mask_by_k_clustering(
    src: np.ndarray[float], ks: list[int], **kwargs
) -> np.ndarray[bool]:
    # Cluster on all ks and score
    ks = iterable_array(ks)
    clusters, scores = try_n_clusters(src, ks, **kwargs)

    # Take slice from best scorer
    clusters, elbow = find_elbow_clusters(clusters, scores)

    # Collapse to mask
    mask = slice_clusters(
        src,
        clusters,
        slice_to_take=kwargs["slice_to_take"] if "slice_to_take" in kwargs else None,
    )

    return mask


def k_cluster(
    src: np.ndarray[float], k: int = 3, include_location: bool = False
) -> np.ndarray[int]:
    X = prepare_clustering_variables(src, include_location=include_location)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", init="random").fit(X)
    return kmeans.labels_.reshape(src.shape[-2:])


def intra_vs_inter_cluster_variance(
    src: np.ndarray[float], labels: np.ndarray[int]
) -> np.float64:
    if src.ndim > 2:
        src = src.reshape(src.shape[0], -1).T
        labels = labels.flatten()
    clusters = np.unique(labels)
    centroids = [np.nanmean(src[labels == lab]) for lab in clusters]
    global_mean = np.nanmean(src)
    intra = np.nansum(
        [
            np.nansum((src[labels == lab] - centr) ** 2)
            for lab, centr in zip(clusters, centroids)
        ]
    )
    inter = np.nansum(
        [
            np.count_nonzero(labels == lab) * (centr - global_mean) ** 2
            for lab, centr in zip(clusters, centroids)
        ]
    )
    return inter / (inter + intra)


def try_n_clusters(
    src: np.ndarray[float], ks: list[int], **kwargs
) -> tuple[np.ndarray[bool], np.ndarray[float]]:
    # KMeans Clustering
    clusters = np.zeros((len(ks),) + src.shape[-2:])
    scores = np.zeros(len(ks))
    for i, k in enumerate(ks):
        clusters[i] = k_cluster(
            src,
            k,
            include_location=(
                kwargs["include_location"] if "include_location" in kwargs else False
            ),
        )
        scores[i] = intra_vs_inter_cluster_variance(src, clusters[i])
    return clusters, scores


def find_elbow_clusters(
    clusters: np.ndarray[int], scores: np.ndarray[float]
) -> tuple[np.ndarray[int], int]:
    # Find where more clusters stops improving inter/intragroup variance
    elbow = np.argmax(np.gradient(scores)) + 1

    # Select that configuration of  clusters
    return clusters[elbow], elbow


def slice_clusters(
    src: np.ndarray[float], clusters: np.ndarray[bool], slice_to_take: slice = None
) -> np.ndarray[bool]:
    if slice_to_take is None:
        slice_to_take = slice(2, None)

    # Select the clusters (ordered by intensity and selected from slice)
    selected = np.argsort(
        [np.average(src[..., clusters == i]) for i in np.unique(clusters)]
    )[slice_to_take]

    # Select src where it is in the selected clusters
    in_cluster_mask = np.any([clusters == sel for sel in selected], axis=0)
    return in_cluster_mask


def mask_by_gmm(src: np.ndarray[float], ks: list[int], **kwargs) -> np.ndarray[bool]:
    # Cluster on all ks and score
    ks = iterable_array(ks)
    clusters, scores = try_n_gaussians(src, ks, **kwargs)

    # Take slice from best scorer
    clusters, elbow = find_elbow_clusters(clusters, scores)

    # Collapse to mask
    mask = slice_clusters(
        src,
        clusters,
        slice_to_take=kwargs["slice_to_take"] if "slice_to_take" in kwargs else None,
    )

    return mask


def try_n_gaussians(
    src: np.ndarray[float], ks: list[int], **kwargs
) -> tuple[np.ndarray[bool], np.ndarray[float]]:
    # Gaussian Mixture Modelling
    clusters = np.zeros((len(ks),) + src.shape[-2:])
    scores = np.zeros(len(ks))
    for i, k in enumerate(ks):
        clusters[i] = gm_model(
            src,
            k,
            include_location=(
                kwargs["include_location"] if "include_location" in kwargs else False
            ),
        )
        scores[i] = intra_vs_inter_cluster_variance(src, clusters[i])
    return clusters, scores


def gm_model(
    src: np.ndarray[float], k: int = 3, include_location: bool = False
) -> np.ndarray[int]:
    X = prepare_clustering_variables(src, include_location=include_location)
    labels = GaussianMixture(n_components=k, random_state=42).fit_predict(X)
    return labels.reshape(src.shape[-2:])


def prepare_clustering_variables(
    src: np.ndarray[float], include_location: bool = False
) -> tuple[np.ndarray[float], np.ndarray[int]]:
    shape = src.shape[-2:]
    X = vectorize_img(src, include_location=include_location)
    X = StandardScaler().fit_transform(X)
    return X


def find_cycles(root: Union[str, Path], search_term="metadata.json") -> list[Path]:
    found_paths = []
    root = ensure_path(root)
    for path, _, files in root.walk():
        for f in files:
            if search_term in f:
                found_paths.append(Path(path))
                break
    return found_paths


def gabor_filter_bank(
    frequency=1, theta_step=np.radians(15), sigma_x=None, sigma_y=None, offset=None
) -> np.ndarray[float]:
    theta = np.arange(0, np.pi, theta_step)
    freqs = iterable_array(frequency)
    sigma_x = iterable_array(sigma_x) if sigma_x is not None else 1.5 / freqs
    sigma_y = iterable_array(sigma_y) if sigma_y is not None else 0.5 / freqs
    offset = iterable_array(offset) if offset is not None else np.zeros_like(freqs)
    kernels = [
        np.abs(gabor_kernel(frequency=f, theta=t, sigma_x=sx, sigma_y=sy, offset=o))
        for t, (f, sx, sy, o) in itertools.product(
            theta, zip(freqs, sigma_x, sigma_y, offset)
        )
    ]
    return kernels


def leastsq_reflectance(
    hyperstack: np.ndarray[float], eps: np.ndarray[float]
) -> tuple[np.ndarray[float], np.ndarray[float]]:

    # Create design matrix
    A = np.column_stack([eps.T, np.ones_like(eps.T)])

    # Create output array
    c = np.zeros((A.shape[1], *hyperstack.shape[1:]))

    # Fit each pixel
    for i, j in itertools.product(
        range(hyperstack.shape[1]), range(hyperstack.shape[2])
    ):
        log_r = -np.log((hyperstack[:, i, j]))
        c[:, i, j], _, _, _ = np.linalg.lstsq(A, log_r, rcond=None)
    return c
