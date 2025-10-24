.. _footprint_example:

Building Footprints
===================

Building footprints serve as the foundational data for generating any building inventory with |app|. Users may supply their own footprint datasets or leverage |app|’s capabilities to create an inventory from scratch. |app| includes four footprint scraper classes that enable automatic retrieval of footprint data from online sources:

#. ``MS_FootprintScraper`` using `Microsoft Footprint Database <https://github.com/microsoft/USBuildingFootprints>`_
#. ``OSM_FootprintScraper`` using `Open Street Maps (OSM) <https://www.openstreetmap.org/about>`_
#. ``USA_FootprintScraper`` using `USA Structures <https://gis-fema.hub.arcgis.com/pages/usa-structures>`_
#. ``OvertureMapsFootprintScraper`` using `Overture Maps <https://overturemaps.org/>`_

Each of these scraper classes implements a method ``get_footprints``, which returns the building footprints for a given ``RegionBoundary`` instance. The example below demonstrates how three different scraper classes can be used interchangeably to generate building inventories for a specified location when running the script.


.. literalinclude:: brails_footprint.py
   :language: python
   :linenos:

To run the example in **brails_footprint.py** for the location Berkeley, CA, use the following command
      
.. code-block::
      
   python3 brails_footprint.py "Berkeley, CA"
   
The example will print out the number of buildings obtained for each scraper.

.. code-block::
   
   ---------------------------
   Scraper          # building
   ---------------------------
   OSM              35547     
   Microsoft        29469     
   USA              28404     
   ---------------------------

When run, the example also prints a two-inventory subset of the data retrieved by each footprint scraper. As shown below, the features obtained for each asset vary between scrapers.

.. literalinclude:: output.txt
   :linenos:

Footprint Notebook
------------------

Here is a link to a Jupyter Notebook that runs the basic code, accompanied by graphics to better illustrate the output. 

.. raw:: html
	 
	 <a href=https://colab.research.google.com/github/NHERI-SimCenter/BrailsPlusPlus/blob/master/examples/footprint/brails_footprint_name_input.ipynb target=_blank> <img src=https://colab.research.google.com/assets/colab-badge.svg/></a>

         <a href=https://lightning.ai/new?repo_url=https%3A//github/NHERI-SimCenter/BrailsPlusPlus/blob/master/examples/footprint/brails_footprint_name_input.ipynb target=_blank> <img src=https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg></a>

      
.. note::

   #. The number of buildings varies across datasets due to differences in data sources, processing methods, geographic coverage, and update frequency. Since no dataset is perfect, users are encouraged to compare building inventories for their area of interest by overlaying them with satellite imagery to verify accuracy.
   #. OSM is a **community-driven** platform where volunteers manually contribute building footprints using ground surveys, GPS data, and licensed aerial imagery. Data quality varies by region and contributor activity, with particularly active communities in non-urban areas. Some buildings may have **NA** values where community data is not yet available.
   #. OSM, Microsoft Footprints, and Overture Maps provide global coverage, whereas USA Structures is limited to the United States.
   #. OSM data is updated in real time by contributors; Microsoft and Overture Maps datasets are updated periodically; USA Structures updates occur infrequently.

      
   
