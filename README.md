<h1 style="text-align: center;">BRAILS++: Building Regional Asset Inventories at Large Scale</h1>

## What is it?

```BRAILS++``` is an object-oriented framework for building applications that focus on generating asset inventories for large geographic regions.

## How is the repo laid out?

#### :building_construction:UNDER CONSTRUCTION!! :building_construction: 

+ ```brails```: A directory containing the classes
  - ```brails/types```: directory containing useful datatypes, e.g., ```ImageSet``` and ```AssetInventory```
  - ```brails/processors```: directory containing classes that do ```image_processing``` to make predictions, e.g. RoofShape</li>
  - ```brails/segmenters```: directory containing classes that do image segmentation
  - ```brails/scrapers```: directory containing classes that do internet downloads, e.g., footprint scrapers, image scrapers
  - ```brails/filters```: directory containing image filters, e.g., classes that take images and revise or filter out
  - ```brails/imputaters```: directory containing classes that fill in missing ```AssetInventory``` datasets
  - ```brails/utils```: directory containing misc classes that do useful things, e.g. geometric conversions
+ ```examples```: A directory containing examples
+ ```tests```: A directory containing unit tests. The directory structure follows that of ```brails```

## Documentation

You can find the documentation for ```BRAILS++``` [here]().

## Installation instructions

```BRAILS++``` is available on PyPI under the name ```BRAILS```.

```shell
pip install BRAILS
```
Developers and contributors should read the [Contributing Code to Brails]() page of the documentation.
