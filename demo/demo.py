#!/usr/env/python3
# -*- coding: utf-8 -*-
"""Demo csci4831final code."""
import sys
import os
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="(%(levelname)s): %(asctime)s --- %(message)s",
)

sys.path.append(os.path.abspath("../"))
import csci4831final as csf


def demo(image: str, background: str, result_dir: str) -> None:
    """
    Perform demo of csci4831final code.

    Does the following in order:
    1. Craft a true foreground mask of image.
    2. Use mini-batch k-means with identity, grayscale and gaussian
       transforms to get predicted foreground mask.
    3. Use matplotlib to create result graph.
    4. Save results to result_dir.

    Parameters
    ----------
    image: str
        Path to image to process.
    background: str
        Path to virtual background image.
    result_dir: str
        Path to save results to.
    """

    if os.path.exists("all_results.csv"):
        os.remove("all_results.csv")
    if os.path.exists("avg_results.csv"):
        os.remove("avg_results.csv")
    for file in os.listdir(result_dir):
        if not file.startswith("."):
            os.remove(os.path.join(result_dir, file))

    logging.info("Starting...")
    logging.info("Reading images...")
    image_data = cv2.imread(image)
    background_data = cv2.imread(background)
    logging.info("Showing test image...")
    cv2.imshow("Test Image", image_data)
    cv2.imshow("Background Image", background_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    logging.info("Getting true mask using grabcut...")
    true_mask = csf.get_points.get_true_mask(image_data)
    logging.info("Got true mask...")

    logging.info("Testing mini-batch KMeans with ID, GS, and 51GB...")
    tests = csf.main.make_tests(
        [
            (csf.clusters.minibatch_kmeans, "MBK")
        ],
        [
            (lambda i: i, "ID"),
            (lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), "GS"),
            (lambda i: cv2.GaussianBlur(i, (5, 5), 1), "51GB"),
        ]
    )
    pi = csf.main.PeopleImage(
        filename=os.path.basename(image),
        path=image,
        mask_path="",
        data=image_data,
        mask_data=true_mask
    )
    for test in tests:
        logging.info(
            f"Running {test.model_name} with transform {test.transform_name}"
        )
        csf.main.do_test(test, [pi], result_dir)
        logging.info(f"Time taken: {test.avg_time}")
        logging.info(f"Accuracy: {test.acc}")

    logging.info("Showing Results...")
    fig = plt.figure()
    axes = list()
    rows = 3
    cols = 2
    i = 1
    for prediction in os.listdir(result_dir):
        logging.info(f"Loading {prediction}...")
        filename, model_name, trans_name, acc, run_time = prediction.split("_")
        pred_data = np.float32(cv2.imread(
            os.path.join(result_dir, prediction), cv2.IMREAD_COLOR
        ))
        virt_data = np.float32(
            csf.vbg.replace_background(pred_data, background_data)
        )

        pred_data = np.int32(cv2.cvtColor(pred_data, cv2.COLOR_BGR2RGB))
        virt_data = np.int32(cv2.cvtColor(virt_data, cv2.COLOR_BGR2RGB))

        axes.append(fig.add_subplot(rows, cols, i))
        axes[-1].set_title(f"Prediction ({trans_name})")
        plt.imshow(np.int32(pred_data))

        axes.append(fig.add_subplot(rows, cols, i + 1))
        plt.imshow(np.int32(virt_data))

        i += 2

    fig.tight_layout()
    plt.show()

    return None


if __name__ == "__main__":
    demo(
        os.path.abspath("./image.jpg"),
        os.path.abspath("./background.jpg"),
        os.path.abspath("./results")
    )
