<h1 style="text-align: center;">BRAILS++: Building Regional Asset Inventories at Large Scale</h1>

[![Tests](https://github.com/NHERI-SimCenter/BrailsPlusPlus/actions/workflows/tests.yml/badge.svg)](https://github.com/NHERI-SimCenter/BrailsPlusPlus/actions/workflows/tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/184673734.svg)](https://zenodo.org/badge/latestdoi/184673734)
[![PyPi version](https://badgen.net/pypi/v/BRAILS/)](https://pypi.org/project/BRAILS/)
[![PyPI download month](https://img.shields.io/pypi/dm/BRAILS.svg)](https://pypi.python.org/pypi/BRAILS/)

## What is it?

```BRAILS++``` is an object-oriented framework for building applications that focus on generating asset inventories for large geographic regions.

## How is the repo laid out?

+ ```brails```: A directory containing the classes
  - ```brails/types```: directory containing useful datatypes, e.g., ```ImageSet``` and ```AssetInventory```
  - ```brails/processors```: directory containing classes that do ```image_processing``` to make predictions, e.g. RoofShape
  - ```brails/segmenters```: directory containing classes that do image segmentation.
  - ```brails/scrapers```: directory containing classes that do internet downloads, e.g., footprint scrapers, image scrapers.
  - ```brails/filters```: directory containing image filters, e.g., classes that take images and revise or filter out thiings not needed.
  - ```brails/imputaters```: directory containing classes that fill in missing ```AssetInventory``` datasets, i.e. filling in features that are missing in certain Assets of the AssetInventory.
  - ```brails/inferers```: directory containing classes that infer new asset features based on existing features in the Assets of ```AssetInventory```.
  - ```brails/utils```: directory containing misc classes that do useful things, e.g. geometric conversions
+ ```examples```: A directory containing examples
+ ```tests```: A directory containing unit tests. The directory structure follows that of ```brails```

## Documentation

You can find the documentation for ```BRAILS++``` [here]().

## Installation instructions

```BRAILS++``` is NOT YET available on PyPI. For now, please install ```BRAILS++``` following the syntax:

```shell
pip install git+https://github.com/NHERI-SimCenter/BrailsPlusPlus
```
Developers and contributors, please read the [Contributing to BRAILS++]() page of the documentation before you commit your code.

## Acknowledgements

This work is based on material supported by the National Science Foundation under grants CMMI 1612843 and CMMI 2131111.


## Contact

NHERI-SimCenter nheri-simcenter@berkeley.edu

<!-- todo: instructions on how to lint the code, and specific subfolder or file. -->
<!-- todo: example with the test suite. -->
<!-- todo: instructions on how to run the tests -->
<!-- todo: instructions on how to check coverage -->
<!-- python -m pytest tests --cov=brails --cov-report html -->
