#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply virtual background to given foreground mask."""
import numpy as np
import os
import argparse
import cv2


def fill_holes(image_data: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Fill holes in given image that may be empty from clustering.

    Parameters
    ----------
    image_data: np.ndarray
        image to fill
    original: np.ndarray
        original image

    Returns
    -------
    np.ndarray: resulting image with filled holes
    """

    result = image_data.copy()
    for i in range(1, result.shape[0] - 1):
        first = None
        last = None
        for j in range(1, result.shape[1] - 1):
            if not np.all(image_data[i][j] == 0):
                if first is None:
                    first = j
                last = j
        if last is not None and first is not None:
            for j in range(first, last + 1):
                result[i][j] = original[i][j]
                if np.all(result[i][j] == 0):
                    result[i][j] = [1, 1, 1]

    return result


def replace_background(
    fg_data: np.ndarray,
    bg_data: np.ndarray
)-> np.ndarray:
    """
    Perform virtual background replacement on given image.

    Parameters
    ----------
    fg_data: numpy array
        Image to add virtual background to

    bg_data: numpy array
        Virtual background image

    Returns
    -------
    numpy array
        Resulting image
    """

    background = cv2.resize(bg_data, (fg_data.shape[1], fg_data.shape[0]))
    result = np.zeros(fg_data.shape)
    for i in range(fg_data.shape[0]):
        for j in range(fg_data.shape[1]):
            if list(fg_data[i][j]) == [0, 0, 0]:
                result[i][j] = background[i][j]
            else:
                result[i][j] = fg_data[i][j]

    return result


def main(fg: str, bg: str, out: str, original: str) -> None:
    """
    Apply virtual background to given foreground mask.

    Resources:
    * https://debuggercafe.com/image-foreground-extraction-using-opencv-contour-detection/

    Parameters
    ----------
    fg: str
        Foreground mask image.
    bg: str
        Background mask image.
    out: str
        Output file path
    original: str
        Path to original foreground image.

    Returns
    -------
    None
    """

    fg = os.path.abspath(fg)
    bg = os.path.abspath(bg)
    out = os.path.abspath(out)

    fg_data = cv2.imread(fg)
    bg_data = cv2.imread(bg)

    if original is not None:
        original = os.path.abspath(original)
        org_data = cv2.imread(original)
        fg_data = fill_holes(fg_data, org_data)

    result = replace_background(fg_data, bg_data)

    cv2.imwrite(out, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply virtual background to saved foreground mask."
    )
    parser.add_argument("--fg", "-f", dest="fg", help="Foreground image.")
    parser.add_argument("--bg", "-b", dest="bg", help="Background image.")
    parser.add_argument(
        "--original", "-r", dest="original", help="Original foreground image.",
        default=None
    )
    parser.add_argument("--out", "-o", dest="out", help="Result output.")
    args = parser.parse_args()

    main(args.fg, args.bg, args.out, args.original)
