#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main module of csci4831final."""
import logging
import os
from typing import Any, List, Tuple
from dataclasses import dataclass
import argparse
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.DEBUG, 
    encoding="utf-8",
    format="(%(levelname)s): %(asctime)s --- %(message)s"
)


@dataclass
class Test:
    """Class for keeping track of testing a model."""

    model: Any  # sklearn model the test was performed on
    model_name: str  # str name of the model
    transform: Any  # transform to perform on each image
    transform_name: str  # str name of the image transform
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
        to_predict = test.transform(image.data)
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


def make_tests(
    models: List[Tuple[Any, str]], 
    transforms: List[Tuple[Any, str]]
) -> List[Test]:
    """
    Construct tests of combinations of given transforms and models.

    Parameters
    ----------
    models: list of tuple
        Each tuple should be (model, model name)
    transforms: list of tuple
        Each tuple should be (transform, transform name)

    Returns
    -------
    list of Test
    """

    tests = list()

    for model, model_name in models:
        for transform, transform_name in transforms:
            tests.append(Test(
                model=model,
                model_name=model_name,
                transform=transform,
                transform_name=transform_name,
                desc=f"Model '{model_name}' with transform '{transform_name}'."
            ))

    return tests


def make_graph(
    results: List[Test], 
    save: bool = False,
    show: bool = True
) -> None:
    """
    Make prettry graphs representing test results.

    Graphs are displayed

    Parameters
    ----------
    results: list of Test
        Fully filled out Test instances
    save: bool, optional
        If True, will save to disk. Defaults to False.
    show: bool, optional
        If True, will show graphs. Defaults to True.
    """


    def make_bar(x, x_label, y, y_label, title):
        fig, ax = plt.subplots()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.bar(x, y)

        return fig, ax

    transform_by_model = dict()  # model name: (transform name, acc, time)
    model_by_transform = dict()  # transform name: (model name, acc, time)
    for result_top in results:
        if transform_by_model.get(result_top.model_name, None) is None:
            transform_by_model[result_top.model_name] = list()
        if model_by_transform.get(result_top.transform_name, None) is None:
            transform_by_model[result_top.transform_name] = list()
            
        for result in results:
            transform_by_model[result_top.model_name].append(
                (result.model_name, result.acc, result.time)
            )
            model_by_transform[result_top.transform_name].append(
                (result.transform_name, result.acc, result.time)
            )

    
    for model_name, transforms in transform_by_model.items():
        y_acc = [transform[1] for transform in transforms]
        y_time = [transform[2] for transform in transforms]
        x = [transform[0] for transform in transforms]
        title_acc = f"Accuracy of {model_name}"
        title_time = f"Time of {model_name}"
        
        fig_acc, ax_acc = make_bar(
            x, "Transforms", y_acc, "Accuracy", title_acc
        )
        fig_time, ax_time = make_bar(
            x, "Transforms", y_time, "Time", title_time
        )

        if save:
            fig_acc.savefig(f"accuracy_of_{model_name}.png")
            fig_time.savefig(f"time_of_{model_name}.png")
        if show:
            fig_acc.show()
            fig_time.show()

    for transform_name, model in model_by_transform.items():
        y_acc = [model[1] for model in models]
        y_time = [model[2] for model in models]
        x = [model[0] for model in models]
        title_acc = f"Accuracy of {transform_name}"
        title_time = f"Time of {transform_name}"
        
        fig_acc, ax_acc = make_bar(
            x, "Models", y_acc, "Accuracy", title_acc
        )
        fig_time, ax_time = make_bar(
            x, "Models", y_time, "Time", title_time
        )
        
        if save:
            fig_acc.savefig(f"accuracy_of_{transform_name}.png")
            fig_time.savefig(f"time_of_{transform_name}.png")
        if show:
            fig_acc.show()
            fig_time.show()
        

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


    tests = make_tests(
        [  # models
            (KMeans(n_clusters=2), "KMeans with 2 Clusters")
        ],
        [  # transforms
            (lambda x: x, "Identity Transform")
        ]
    )
    
    main(
        os.path.abspath(args.image_dir),
        args.save, 
        tests
    )
