#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main module of csci4831final."""
import logging
import os
from typing import Any, List
from dataclasses import dataclass
import argparse
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import cv2


logging.basicConfig(
    level=logging.DEBUG, 
    encoding="utf-8",
    format="(%(levelname)s): %(asctime)s --- %(message)s"
)


@dataclass
class Test:
    """Class for keeping track of testing a model."""

    model: Any  # sklearn model the test was performed on
    image_transform: Any  # transform to perform on each image
    desc: str  # Printable description of the test
    acc: float = 0  # Accuracy of the model
    avg_time: float = 0  # Average running time of the model 


@dataclass
class PeopleImage:
    """Class for keeping track of a people image."""

    filename: str  # Filename of image on disk
    path: str  # Path to image on disk
    data: np.ndarray  # output of cv2.imread
    true_mask: np.ndarray = None  # True foreground mask of the image



TESTS = [
    Test(
        model=KMeans(n_clusters=2),
        image_transform=lambda x: x,
        desc="KMeans with 2 clusters and no image transformation."
    )
]


def get_true_mask(in_image: PeopleImage) -> None:
    """
    Sets the true_mask attribute of a PeopleImage.

    This function takes in a PeopleImage class and then sets
    the true_mask attribute to what the image's true foreground
    mask should be.
    """

    raise RuntimeError("Not Implemented")
    

def load_images(image_dir: str) -> List[PeopleImage]:
    """
    Goes through the given image directory and loads images.

    Resources:
    * https://careerkarma.com/blog/python-list-files-in-directory/
    * https://www.tutorialkart.com/opencv/python/opencv-python-read-display-image/
    
    Arguments
    ---------
    image_dir: str
        Location of images of people.


    Returns
    -------
    list of PeopleImage
    """

    images = list()
    for file in os.listdir(image_dir):
        path = os.path.join(image_dir, file)
        images.append(
            PeopleImage(
                filename=file,
                path=path,
                data=cv2.imread(path)
            )
        )

    return images


def do_test(test: Test, data: List[PeopleImage]) -> None:
    """
    Given a Test instance, run the test and set metric attributes.

    Resources:
    * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    * https://www.geeksforgeeks.org/time-perf_counter-function-in-python/

    Arguments
    ---------
    test: Test
        Test dataclass to use.
    data: list of PeopleImage
        Input data to test the model against

    Returns
    -------
    Test
        A fully populated Test instance.
    """

    accuracies = list()
    times = list() 
 

    for image in data:
        start = time.perf_counter()
        to_predict = test.image_transform(image.data)
        prediction = test.model.fit_predict(to_predict)
        acc = accuracy_score(image.true_mask, prediction)
        accuracies.append(acc)
        end = time.perf_counter()
        times.append(end - start)


    test.acc = np.mean(accuracies)
    test.avg_time = np.mean(times)


def save_results(tests: List[Test]) -> None:
    """
    Save results of test functions to disk.
    """

    raise RuntimeError("Not Implemented")
    

def main(image_dir: str, tests: List[Test], save: bool) -> None:
    """
    Main entrypoint function.


    Arguments
    ---------
    image_dir: str
        Location of images of people.
    tests: list of Test
        List of Tests to run.
    save: bool
        If True, will call ``save_results`` to save test
        results to disk.
    """

    logging.info(f"Loading images from {image_dir}...")
    images = load_images(image_dir)
    logging.info("Getting true masks...")
    for image in images:
        image.true_mask = get_true_mask(image)

    logging.info("Running tests...")
    for test in tests:
        logging.info(f"Running test: {test.desc}")
        do_test(test)

    logging.info("Finished running tests!")
    if save:
        logging.info("Saving results...")
        save_results(tests)

    logging.info("Done!")
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="csci4821 final project")
    parser.add_argument(
        "--image-dir", dest="image_dir", default="../People_Images"
    )
    parser.add_argument(
        "--save", dest="save", default=True
    )
    args = parser.parse_args()
    
    main(
        os.path.abspath(args.image_dir),
        args.save, 
        TESTS
    )
