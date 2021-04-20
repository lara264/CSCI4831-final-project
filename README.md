# CSCI4831 Final Project

Lara Chunko and Ryan Drew, 2021

## Installation

Installation ain't too bad. We use poetry for managing dependencies and 
installation. Just make sure you have Python 3.8 and ``poetry`` 
installed (i.e. ``pip install poetry``) and run the following:

```bash
$ git clone https://github.com/lara264/CSCI4831-final-project
$ cd CSCI-final-project
$ poetry install
```

Poetry will install dependencies and create a virtual environment.
The virtual environment can be accessed in two ways:

Run a command within the virtual environment:
```bash
$ poetry run <command>
```

or activate the virtual environment and run a command:

```bash
$ poetry shell
(csci4831final) $ <command>
```

## Running

Project is implemented as a script which can be run from the shell:

```bash
$ cd csci4831final
$ poetry run main.py --help

usage: main.py [-h] [--run-tests] [--compute-mask COMPUTE_MASK] [--mask-dir MASK_DIR] [--image-dir IMAGE_DIR] [--save-dir SAVE_DIR]
               [--show-graphs]

csci4821 final project

optional arguments:
  -h, --help            show this help message and exit
  --run-tests           If given, will run model tests.
  --compute-mask COMPUTE_MASK
                        If given, will compute true mask for given image.
  --mask-dir MASK_DIR   Directory containing true foreground masks of images inimage-dir.
  --image-dir IMAGE_DIR
                        Directory containing pictures of people on a Zoom call.
  --save-dir SAVE_DIR   Save results to the given folder.
  --show-graphs         If given, will show graphs of results.
```

Runtime is about 30-60 minutes. Logs are outputted to a file called `out.log` within the current working directory. Testing can be done on a small subset by utilizing the `--image-dir` and `--mask-dir` options. For instance:

```bash
$ mkdir results
$ mkdir -p small/{masks,people}
$ cp People_Images/Person_37.png small/people
$ cp People_Masks/Person_37.png small/masks
$ cd csci4831final
$ poetry run python main.py --run-tests --mask-dir ../small/masks --image-dir ../small/people
```

When using `--run-tests`, the following models and transforms are used:

* Models
    * KMeans
    * Mini-Batch KMeans
    * HAC with 'Ward' Linkage
    * HAC with 'Complete' Linkage
    * HAC with 'Average' Linkage
    * All of the above, but with a feature vector that includes pixel position
* Transforms
    * Identity (i.e. no transform)
    * Grayscale
    * 5x5 Gaussian Blur with sigma 1, 3, and 5

Each of these models and transforms are abbreviated in outputted results. Note that for HAC models, each image is scaled down 80% in order to save on runtime and memory restraints.

## Results

When running main.py, there are a couple of files that are created:

* `all_results.csv`: CSV file containing results from running each image against each model against each transform. Columns are: run time, filename, model name, transform name, accuracy. Will be placed in the current working directory.
* `avg_results.csv`: CSV file containing average results for each model against each transform. Columns are model name, transform name, accuracy, run time. Will be placed
  in the current working directory.
* `results/`: Directory containing output foreground masks for test, as well as graphs displaying average results. Will be placed in a folder up from the current working directory by default, but this can be changed with `--save-dir`.

## Modules

Here is a brief description of each module included:

* `main.py`: Main entrypoint module for running tests. Contains a skeleton for loading test images, creating test combinations, initiating models and saving results.
* `clusters`: Module holding clustering models that are tested, as well as some helper functions that each model needs. Each model is implemented as a function that takes in an input image and outputs the predicted foreground.
* `vbg`: Module for doing virtual background replacement. Takes in a foreground mask from either `People_Masks` or `results`, an image to be used as the background, and the original image of the person from `People_Images`. See usage:

```
usage: vbg.py [-h] [--fg FG] [--bg BG] [--original ORIGINAL] [--out OUT]

Apply virtual background to saved foreground mask.

optional arguments:
  -h, --help            show this help message and exit
  --fg FG, -f FG        Foreground image.
  --bg BG, -b BG        Background image.
  --original ORIGINAL, -r ORIGINAL
                        Original foreground image.
  --out OUT, -o OUT     Result output.
```
