.. _lblAbout:

*****
About
*****

The Building Regional Asset Inventories at Large Scale (BRAILS++) is an object-oriented framework for creating applications that focus on generating asset inventories for large geographic regions. The framework was developed by `NHERI SimCenter <https://simcenter.designsafe-ci.org/>`_ for the creation of asset inventories needed to understand the effects of natural hazards on the built environment.

Like any object-oriented framework, BRAILS provides a number of high level classes whose interfaces prescribe a means
by which the classes in the workflow can be used. The following are these high level classes:

#. Scrapers: A scraper is code that will go out and get data from the www, e.g a scraper might go to the NSI database and obtain information on the buildings in a region.
   
#. Processor: A processor is code that wil take an image set and determine information for the images in the set.

#. Segmenters: A segmenter is code that will again take an image set and for each image will segment it, e.g. a segmenter might identify the roof, windows, and doors, in an image.

#. Filters: A filter is code that will again take an image set and for each image will create a filtered image, e.g. a filter might remove everything but the building in the center of an image.

#. Imputers: An imputer is code that given an AssetInventory that contains missing information for certain fields of certain assets, provide a new inventory that contains a number of possible entries for those missing fields. hey generate what is termed **possible worlds**.
   
#. Inferer: An inferer is code that given the fields for each asset in an asset inventory, will infer new fields.

   
The classes work together, through their interface and the passing of common data types. The data types for BRAILS++, being:

1. AssetInventory

2. ImageSet

3. RegionBoundary

For the Documentaion of BRAILS++ we are providing examples and the api documentation. The api documentation being provided by sphinx-apidoc.

.. note::

   Some of the code in BRAILS++ is taken from the original `BRAILS work <https://nheri-simcenter.github.io/BRAILS-Documentation/common/about/cite.html>`_. The original BRAILS code has been re-factored to be more object-oriented and extended to include new features.
