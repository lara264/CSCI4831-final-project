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
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    SpectralClustering
)
import cv2
import matplotlib.pyplot as plt
import get_points


logging.basicConfig(
    level=logging.INFO,
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
    mask_path: str  # Path to image forground mask on disk
    data: np.ndarray  # output of cv2.imread
    mask_data: np.ndarray  # True foreground mask of the image


def compute_mask(image: str, mask_dir: str) -> None:
    """
    Compute mask for given image

    Parameters
    ----------
    image : str
        image to compute mask for
    mask_dir : str
        directory to save result to
    """

    file = os.path.basename(image)
    mask_path = os.path.join(mask_dir, file)
    logging.info(f"Computing mask for {file}...")
    image = cv2.imread(image)
    cv2.imwrite(
        mask_path,
        get_points.get_true_mask(image)
    )


def load_images(
    image_dir: str,
    mask_dir: str,
    compute_masks: bool = False
) -> List[PeopleImage]:
    """
    Goes through the given image directory and loads images.

    Resources:
    * https://careerkarma.com/blog/python-list-files-in-directory/
    * https://www.tutorialkart.com/opencv/python/opencv-python-read-display-image/
    
    Arguments
    ---------
    image_dir: str
        Location of images of people.
    mask_dir: str
        Location of the foreground masks for the images of people.
    compute_masks: bool
        If True, will compute the mask of the image and save. If
        False, will try to load from disk. Defaults to False


    Returns
    -------
    list of PeopleImage
    """

    images = list()
    for file in os.listdir(image_dir):
        logging.info(f"Loading {file}...")
        path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file)
        image = cv2.imread(path)

        if compute_masks:
            compute_mask(image, mask_path)

        images.append(
            PeopleImage(
                filename=file,
                path=path,
                mask_path=mask_path,
                data=image,
                mask_data=cv2.imread(mask_path)
            )
        )

    return images


def do_test(test: Test, data: List[PeopleImage], save_dir: str) -> None:
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
        Input data to test the model against.
    save_dir: str
        Directory to save results to.

    Returns
    -------
    Test
        A fully populated Test instance.
    """

    accuracies = list()
    times = list() 
    results = list()

    for image in data:
        start = time.perf_counter()

        to_predict = test.transform(image.data).reshape(
            image.data.shape[0] * image.data.shape[1],
            -1
        )
        prediction = test.model.fit_predict(to_predict).reshape(
            image.data.shape[0], image.data.shape[1], 1
        )

        end = time.perf_counter()

        foreground_label = prediction[0][0]
        if foreground_label > 0:
            prediction = np.ones(prediction.shape) - prediction
            
        prediction = (prediction * 255).astype("uint8")
        predicted_mask = cv2.bitwise_and(
            image.data, image.data, mask=prediction
        )
        acc = float(np.sum(np.abs(
            cv2.cvtColor(image.mask_data, cv2.COLOR_BGR2GRAY) - 
            cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2GRAY)
        ))) / (image.mask_data.shape[0] * image.mask_data.shape[1])
        
        accuracies.append(acc)
        run_time = end - start
        times.append(run_time)
        result = [
            test.model_name, test.transform_name, acc, run_time, image.filename
        ]
        results.append(result)

        cv2.imwrite(
            os.path.join(save_dir, "_".join(map(str, result))), predicted_mask
        )


    test.acc = np.mean(accuracies)
    test.avg_time = np.mean(times)
    save_results(test, results)


def save_results(t: Test, results: List[List[str]]) -> None:
    """
    Save results of test functions to disk.
    """

    # save the results and store in a .txt file

    # storage format: ModelName_TransformName_Accuracy_RunTime

    avg_results = open("avg_results.csv", "a")
    all_results = open("all_results.csv", "a")

    data = f"{t.model_name},{t.transform_name},{t.acc},{t.avg_time}\n"
    avg_results.write(data)

    for one_run in results:
        data = f"{one_run[4]},{one_run[0]},{one_run[1]},{one_run[2]},{one_run[3]}\n"
        all_results.write(data)

    avg_results.close()
    all_results.close()


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
    save_dir: str,
    show: bool = True
) -> None:
    """
    Make pretty graphs representing test results.

    Parameters
    ----------
    results: list of Test
        Fully filled out Test instances
    save_dir: str
        Directory to save results to.
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
            model_by_transform[result_top.transform_name] = list()
            
        for result in results:
            transform_by_model[result_top.model_name].append(
                (result.transform_name, result.acc, result.avg_time)
            )
            model_by_transform[result_top.transform_name].append(
                (result.model_name, result.acc, result.avg_time)
            )

    
    for model_name, transforms in transform_by_model.items():
        y_acc = [transform[1] for transform in transforms]
        y_time = [transform[2] for transform in transforms]
        x = [transform[0] for transform in transforms]
        title_acc = f"Accuracy of {model_name}"
        title_time = f"Time of {model_name}"
        
        fig_acc, ax_acc = make_bar(
            x, "Transforms", y_acc, "Accuracy (%)", title_acc
        )
        fig_time, ax_time = make_bar(
            x, "Transforms", y_time, "Time (Seconds)", title_time
        )

        fig_acc.savefig(
            os.path.join(save_dir, f"accuracy_of_{model_name}.png")
        )
        fig_time.savefig(
            os.path.join(save_dir, f"time_of_{model_name}.png")
        )

        if show:
            fig_acc.show()
            fig_time.show()

    for transform_name, models in model_by_transform.items():
        y_acc = [model[1] for model in models]
        y_time = [model[2] for model in models]
        x = [model[0] for model in models]
        title_acc = f"Accuracy of {transform_name}"
        title_time = f"Time of {transform_name}"
        
        fig_acc, ax_acc = make_bar(
            x, "Models", y_acc, "Accuracy (%)", title_acc
        )
        fig_time, ax_time = make_bar(
            x, "Models", y_time, "Time (Seconds)", title_time
        )

        fig_acc.savefig(
            os.path.join(save_dir, f"accuracy_of_{transform_name}.png")
        )
        fig_time.savefig(
            os.path.join(save_dir, f"time_of_{transform_name}.png")
        )

        if show:
            fig_acc.show()
            fig_time.show()
        

def main(
    image_dir: str,
    mask_dir: str,
    tests: List[Test],
    compute_masks: bool,
    save_dir: str,
    show: bool
) -> None:
    """
    Main entrypoint function.


    Arguments
    ---------
    image_dir: str
        Location of images of people.
    mask_dir: str
        Location of foreground masks of people.
    tests: list of Test
        List of Tests to run.
    comput_masks: bool
        If True, will (re)compute masks and save results. If False,
        will attempt to load from disk.
    save_dir: str
        Directory to save results to.
    show: bool
        If True, will call ``make_graph`` to show graphs of results.
    """

    logging.info(
        f"Loading images from {image_dir} and masks from {mask_dir}..."
    )
    images = load_images(image_dir, mask_dir, compute_masks=compute_masks)

    logging.info("Running tests...")
    for test in tests:
        logging.info(f"Running test: {test.desc}")
        do_test(test, images, save_dir)

    logging.info("Finished running tests!")

    logging.info("Making graphs...")
    make_graph(tests, save_dir, show)

    logging.info("Done!")
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="csci4821 final project")
    parser.add_argument(
        "--run-tests", dest="run_tests", default=False,
        action="store_true", help="If given, will run model tests."
    )
    parser.add_argument(
        "--compute-mask", dest="compute_mask",
        help="If given, will compute true mask for given image. If 'ALL' is"
             "given, then will compute mask for all images as they are loaded"
    )
    parser.add_argument(
        "--mask-dir", dest="mask_dir", default="../People_Masks",
        help="Directory containing true foreground masks of images in"
             "image-dir."
    )
    parser.add_argument(
        "--image-dir", dest="image_dir", default="../People_Images",
        help="Directory containing pictures of people on a Zoom call."
    )
    parser.add_argument(
        "--save-dir", dest="save_dir", default="../results",
        help="Save results to the given folder."
    )
    parser.add_argument(
        "--show-graphs", dest="show", default=False, action="store_true",
        help="If given, will show graphs of results."
    )
    args = parser.parse_args()

    if not args.run_tests and not args.compute_mask:
        print("Expected one of '--run-tests' or '--compute-mask'.")
        exit(1)

    image_dir = os.path.abspath(args.image_dir)
    mask_dir = os.path.abspath(args.mask_dir)

    if args.compute_mask:
        compute_mask(args.compute_mask, mask_dir=mask_dir)
        exit(0)


    tests = make_tests(
        [  # models
            (AgglomerativeClustering(n_clusters=2), "HAC"),
            (KMeans(n_clusters=2), "KMeans"),
            (SpectralClustering(n_clusters=2), "Spectral"),
        ],
        [  # transforms
            (lambda i: i, "Identity"),
            (
                lambda i: cv2.GaussianBlur(i, (5, 5), 1),
                "5x5 Gaussian Blur with Sigma=1"
            ),
            (
                lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY),
                "Grayscale"
            )
        ]
    )

    if args.run_tests:
        main(
            image_dir=os.path.abspath(args.image_dir),
            mask_dir=os.path.abspath(args.mask_dir),
            tests=tests,
            compute_masks=args.compute_mask,
            save_dir=args.save_dir,
            show=args.show
        )
