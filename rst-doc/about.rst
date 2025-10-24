.. _lblAbout:

*****
About
*****

The Building Regional Asset Inventories for Large Scale Simulation (BRAILS++) is an object-oriented framework for creating applications that focus on generating asset inventories for large geographic regions. The framework was developed by `NHERI SimCenter <https://simcenter.designsafe-ci.org/>`_ for the creation of asset inventories needed to understand the effects of natural hazards on the built environment.

Overview of High-Level Classes
------------------------------

As an object-oriented framework, BRAILS++ provides a set of high-level classes, each with well-defined interfaces that allow them to work together seamlessly in a workflow. The primary classes are:

#. **Scrapers**: Code that retrieves data from external sources. For example, a scraper might access the NSI database to gather information on buildings in a given region.

#. **Processors**: Code that analyzes image sets to extract relevant information from each image.

#. **Filters**: Code that modifies image sets by filtering out irrelevant imagery. For example, a filter might isolate only the building of interest in an image.

#. **Imputers**: Code that fills in missing information in an ``AssetInventory``. Given an inventory with incomplete fields, imputers generate multiple possible values, creating what are called **possible worlds**.

#. **Inferers**: Code that infers new asset fields from existing information in an ``AssetInventory`` using empirically derived rulesets.

Shared Data Types
-----------------

These classes interact through their interfaces and by passing shared data types, which include:

1. **AssetInventory**: Represents the collection of assets and their attributes.

2. **ImageSet**: Represents a collection of images used in processing or filtering.

3. **RegionBoundary**: Represents the spatial boundaries of the area being analyzed.

Documentation
-------------

The BRAILS++ documentation includes **examples demonstrating practical workflows** as well as **API documentation** generated with `sphinx-apidoc <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`_.

.. note::

   Some of the code in BRAILS++ originates from the original `BRAILS work <https://nheri-simcenter.github.io/BRAILS-Documentation/common/about/cite.html>`_. The original code has been refactored to adopt a more object-oriented design and extended with additional features.

