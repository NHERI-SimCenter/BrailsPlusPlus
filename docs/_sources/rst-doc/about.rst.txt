.. _lblAbout:

*****
About
*****

The Building Regional Asset Inventories at Large Scale (BRAILS++) is an object-oriented framework for creating applications that focus on generating asset inventories for large geographic regions.

Like any object-oriented framework, BRAILS provides a number of high level classes whose interfaces prescribe a means by
which the classes in the workflow can be used. The following are these high level classes:

1. Scrapers: A scraper is code that will go out and get data from the www, e.g a scraper might go to the NSI database and obtain information on the buildings in a region.
   
2. Processor: A processor is code that wil take an image set and determine information for the images in the set.

3. Segmenters: A segmenter is code that will again take an image set and for each image will segment it, e.g. a segmenter
might identify the roof, windows, and doors, in an image.

4. Filters: A filter is code that will again take an image set and for each image will create a filtered image, e.g. a filter might remove everything but the building in the center of an image.

5. Imputers: An imputer is code that given an AssetInventory that contains missing information for certain fields of certain assets, provide a new inventory that contains a number of possible entries for those missing fields. hey generate what is termed **possible worlds**.
   
6. Inferer: An inferer is code that given the fields for each asset in an asset inventory, will infer new fields.

The classes work together, through their interface and the passing of common data types. The data types for BRAILS++, being:

1. AssetInventory

2. ImageSet

For the Documentaion of BRAILS++ we are providing examples and the api documentation. The api documentation being provided by sphinx-apidoc.

.. note::

   Some of the code in BRAILS++ is taken from the original `BRAILS work <https://nheri-simcenter.github.io/BRAILS-Documentation/common/about/cite.html>_`. The original BRAILS code has been re-factored to be more object-oriented and extended to include new features.
