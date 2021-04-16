#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main module of csci4831final."""
import os
from typing import List
from dataclasses import dataclass
import numpy as np
import argparse
import cv2


@dataclass
class PeopleImage:
    """Class for keeping track of a people image."""

    filename: str      # Filename of image on disk
    path: str          # Path to image on disk
    image: np.ndarray  # output of cv2.imread
    


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
                image=cv2.imread(path)
            )
        )

    return images


def main(image_dir: str) -> None:
    """
    Main entrypoint function.


    Arguments
    ---------
    image_dir: str
        Location of images of people.
    """

    print(load_images(image_dir))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="csci4821 final project")
    parser.add_argument(
        "--image-dir", dest="image_dir", default="../People_Images"
    )
    args = parser.parse_args()
    
    main(args.image_dir)
