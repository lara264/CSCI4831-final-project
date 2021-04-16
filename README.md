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
$ poetry run csci4831final/main.py --help

usage: main.py [-h] [--image-dir IMAGE_DIR] [--save] [--show-graphs]

csci4821 final project

optional arguments:
  -h, --help            show this help message and exit
  --image-dir IMAGE_DIR
                        Directory containing pictures of people on a Zoom call.
  --save                If given, will save results to disk.
  --show-graphs         If given, will show graphs of results.
```
