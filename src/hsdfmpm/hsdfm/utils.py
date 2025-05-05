import itertools
import json
from pathlib import Path

import numpy as np

from numpy.lib._stride_tricks_impl import as_strided
from photon_canon.lut import LUT
from skimage.filters import gabor_kernel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from enum import Enum
from typing import Callable, Union

from ..utils import vectorize_img, ensure_path, iterable_array

def read_metadata_json(file_path):
    grouped_metadata = {
        'AbsTime': [],
        'ExpTime': [],
        'Filter': [],
        'AvgInt': [],
        'Wavelength': [],
    }

    # Open and read the file contents
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)  # Directly load the JSON data from the file

        # Iterate through each entry and group values by field name
        for entry in json_data:
            grouped_metadata['AbsTime'].append(entry.get('AbsTime', None))
            grouped_metadata['ExpTime'].append(entry.get('ExpTime', None))
            grouped_metadata['Filter'].append(entry.get('Filter', None))
            grouped_metadata['AvgInt'].append(entry.get('AvgInt', None))
            grouped_metadata['Wavelength'].append(entry.get('Wavelength', None))

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return grouped_metadata

def normalize_integration_time(hyperstack, integration_time):
    hyperstack /= np.array(integration_time)[:, np.newaxis, np.newaxis]
    return hyperstack

def normalize_to_standard(hyperstack, standard, bg):
    return (hyperstack - bg) / (standard - bg)

def get_local_stdev(image, shape):
    C, H, W = image.shape
    factor = np.asarray((H, W)) // shape
    new_shape = (C, shape[0], factor[0], shape[1], factor[1])
    new_strides = (
        image.strides[0],
        image.strides[1] * factor[0],
        image.strides[1],
        image.strides[2] * factor[1],
        image.strides[2]
    )

    blocks = as_strided(image, shape=new_shape, strides=new_strides)
    return np.nanstd(blocks, axis=(2, 4))

def k_cluster_macro(src, ks, slice_to_take=None):
    pass

def k_cluster(src, k=3, include_location=False):
    shape = src.shape[-2:]
    X = vectorize_img(src, include_location=include_location)
    if include_location:
        X = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', init='random').fit(X)
    return kmeans.labels_.reshape(shape)

def intra_vs_inter_cluster_variance(src, labels):
    if src.ndim > 2:
        src = src.reshape(src.shape[0], -1).T
        labels = labels.flatten()
    clusters = np.unique(labels)
    centroids = [np.nanmean(src[labels == lab]) for lab in clusters]
    global_mean = np.nanmean(src)
    intra = np.nansum(
        [np.nansum(
            (src[labels == lab] - centr) ** 2
        ) for lab, centr in zip(clusters, centroids)
        ]
    )
    inter = np.nansum(
        [np.count_nonzero(labels == lab) * (centr -global_mean) ** 2 for lab, centr in zip(clusters, centroids)]
    )
    return inter / (inter + intra)

def try_n_clusters(src, ks):
    # KMeans Clustering
    clusters = np.zeros((len(ks),) + src.shape[-2:])
    scores = np.zeros(len(ks))
    for i, k in enumerate(ks):
        clusters[i] = k_cluster(src, k)
        scores[i] = intra_vs_inter_cluster_variance(src, clusters[i])
    return clusters, scores

def find_elbow_clusters(clusters, scores):
    # Find where more clusters stops improving inter/intragroup variance
    elbow = np.argmax(np.gradient(scores)) + 1

    # Select that configuration of  clusters
    return clusters[elbow], elbow

def slice_clusters(src, clusters, slice_to_take=None):
    if slice_to_take is None:
        slice_to_take = slice(2, None)

    # Select the clusters (ordered by intensity and selected from slice)
    selected = np.argsort([np.average(src[..., clusters == i]) for i in np.unique(clusters)])[slice_to_take]

    # Select src where it is in the selected clusters
    in_cluster_mask = np.any([clusters == sel for sel in selected], axis=0)
    return in_cluster_mask

def find_cycles(root: Union[str, Path], search_term='metadata.json') -> list[Path]:
    found_paths = []
    root = ensure_path(root)
    for path, _, files in root.walk():
        for f in files:
            if search_term in f:
                found_paths.append(Path(path))
                break
    return found_paths

def gabor_filter_bank(frequency=1, theta_step=np.radians(15), sigma_x=None, sigma_y=None, offset=None):
    theta = np.arange(0, np.pi, theta_step)
    freqs = iterable_array(frequency)
    sigma_x = iterable_array(sigma_x) if sigma_x is not None else 1.5 / freqs
    sigma_y = iterable_array(sigma_y) if sigma_y is not None else 0.5 / freqs
    offset = iterable_array(offset) if offset is not None else np.zeros_like(freqs)
    kernels = [
        np.abs(gabor_kernel(frequency=f, theta=t, sigma_x=sx, sigma_y=sy, offset=o))
        for t, (f, sx, sy, o) in itertools.product(theta, zip(freqs, sigma_x, sigma_y, offset))
    ]
    return kernels