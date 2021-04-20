#!/usr/bin/env python3
# -*- coding: utf-8
"""Cluster algorithms implemented as one-off functions."""
from sklearn.cluster import (
    SpectralClustering, KMeans, MiniBatchKMeans, AgglomerativeClustering,
)
import numpy as np
import cv2


def add_position_feature(image_data: np.ndarray) -> np.ndarray:
    """
    Append the coodinates of each pixel to the color array in image.


    Takes in an image with shape (m, n, 3) and returns an
    image with shape (m, n, 5). The last two are the x and
    y coordinates of the pixel.

    Parameters
    ----------
    image_data: numpy array
        Image to transform

    Returns
    -------
    numpy array
        Transformed image
    """

    if len(image_data.shape) == 3:
        result = np.zeros((
            image_data.shape[0],
            image_data.shape[1],
            image_data.shape[2] + 2
        ))
    else:
        result = np.zeros((
            image_data.shape[0],
            image_data.shape[1],
            3
        ))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i][j] = np.append(
                image_data[i][j], np.array([i, j])
            )

    return result


def get_accuracy(predicted: np.ndarray, true: np.ndarray) -> float:
    """Get accuracy of predicted foreground in comparison to true."""

    predicted_binary = (predicted > 0).astype("int8")
    true_binary = (true > 0).astype("int8")

    total_size = predicted_binary.size
    diff = float(np.sum(np.abs(true_binary - predicted_binary)))
    return 1 - (diff / total_size)


def get_foreground_label(labels: np.array) -> np.array:
    """Return labels array which sets foreground cluster id to 1."""

    # Use average of 5x5 box
    foreground_label = np.sum(labels[0:5, 0:5])
    if foreground_label > 25 / 2:
        return np.ones(labels.shape) - labels
    else:
        return labels


def apply_prediction(mask: np.array, image_data: np.ndarray) -> np.ndarray:
    """Return label mask applied to given image"""

    return cv2.bitwise_and(
        image_data.astype("uint8"), image_data.astype("uint8"),
        mask=mask.astype("uint8")
    )


def flatten_image(image_data: np.ndarray) -> np.ndarray:
    """Flatten given image for KMeans."""

    return image_data.reshape(
        image_data.shape[0] * image_data.shape[1], -1
    )


def kmeans(image_data: np.ndarray, fpos: bool=False) -> np.ndarray:
    """Run KMeans on given image"""

    if fpos:
        to_predict = flatten_image(add_position_feature(image_data))
    else:
        to_predict = flatten_image(image_data)
    labels = KMeans(
        n_clusters=2, n_jobs=-1, init="k-means++"
    ).fit_predict(to_predict).astype("uint8")
    labels.resize(
        image_data.shape[0], image_data.shape[1], 1
    )
    labels = get_foreground_label(labels).astype("uint8")
    prediction = apply_prediction(labels, image_data)

    return prediction


def minibatch_kmeans(image_data: np.ndarray, fpos: bool=False) -> np.ndarray:
    """Run mini-batch kmeans"""

    if fpos:
        to_predict = flatten_image(add_position_feature(image_data))
    else:
        to_predict = flatten_image(image_data)
    labels = MiniBatchKMeans(
        n_clusters=2,init="k-means++").fit_predict(to_predict).astype("uint8")
    labels.resize(
        image_data.shape[0], image_data.shape[1], 1
    )
    labels = get_foreground_label(labels)
    prediction = apply_prediction(labels, image_data)

    return prediction


def spectral(image_data: np.ndarray) -> np.ndarray:
    """
    Run Spectral Clustering on given image.

    Resources
    * https://sklearn.org/auto_examples/cluster/plot_face_segmentation.html#sphx-glr-auto-examples-cluster-plot-face-segmentation-py
    * https://sklearn.org/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py
    """

    scale = 0.3
    scaled_size = (
        int(image_data.shape[1] * scale), int(image_data.shape[0] * scale)
    )
    # image_data_scaled is (m * scale, n * scale, 3)
    image_data_scaled = cv2.resize(image_data, scaled_size)
    # reduced is (m * scale * n * scale, 1)
    reduced = flatten_image(image_data_scaled)
    # labels is (m * scale * n * scale, 1)
    labels = SpectralClustering(
        n_clusters=2, eigen_solver="amg", assign_labels="discretize",
        n_jobs=-1
    ).fit_predict(reduced)
    labels.resize(scaled_size[1], scaled_size[0])
    # label_mask is (m * scale, n * scale, 1)
    label_mask = (get_foreground_label(labels) * 255).astype("uint8")
    prediction = apply_prediction(label_mask, image_data_scaled)
    prediction_upscale = cv2.resize(
        prediction,
        (image_data.shape[1], image_data.shape[0])
    )

    return prediction_upscale


def hac(image_data: np.ndarray, linkage: str, fpos: bool=False) -> np.ndarray:
    """Run HAC Clustering on given image."""

    scale = 0.2
    scaled_size = (
        int(image_data.shape[1] * scale), int(image_data.shape[0] * scale)
    )
    image_data_scaled = cv2.resize(image_data, scaled_size)
    if fpos:
        reduced = flatten_image(add_position_feature(image_data_scaled))
    else:
        # reduced is (m * scale * n * scale, 3)
        reduced = flatten_image(image_data_scaled)
    labels = AgglomerativeClustering(
        n_clusters=2, linkage=linkage).fit_predict(reduced)
    # labels is (m * scale * n * scale, 1)
    labels.resize(
        scaled_size[1], scaled_size[0]
    )
    # label_mask is (m * scale, n * scale, 1)
    label_mask = (get_foreground_label(labels) * 255).astype("uint8")
    prediction = apply_prediction(label_mask, image_data_scaled)
    prediction_upscale = cv2.resize(
        prediction,
        (image_data.shape[1], image_data.shape[0])
    )

    return prediction_upscale
