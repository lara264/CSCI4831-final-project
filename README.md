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

usage: main.py [-h] [--run-tests] [--compute-mask COMPUTE_MASK] [--mask-dir MASK_DIR]
               [--image-dir IMAGE_DIR] [--save-dir SAVE_DIR] [--show-graphs]

csci4821 final project

optional arguments:
  -h, --help            show this help message and exit
  --run-tests           If given, will run model tests.
  --compute-mask COMPUTE_MASK
                        If given, will compute true mask for given image. If 'ALL' isgiven, then will
                        compute mask for all images as they are loaded
  --mask-dir MASK_DIR   Directory containing true foreground masks of images inimage-dir.
  --image-dir IMAGE_DIR
                        Directory containing pictures of people on a Zoom call.
  --save-dir SAVE_DIR   Save results to the given folder.
  --show-graphs         If given, will show graphs of results.
```

Since this program uses algorithms that take up a ton of memory, it
may be useful to limit the amount of memory that this program has
access to. For instance, on a Linux system with systemctl:

```
$ poetry shell
$ sudo systemd-run --scope -p MemoryMax=8G python main.py
```

For a good cookbook on other methods please see this [stackexchange](https://unix.stackexchange.com/questions/44985/limit-memory-usage-for-a-single-linux-process) post
