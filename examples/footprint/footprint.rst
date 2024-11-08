.. _imputation_exammple:

Building Footprints
===================

Building footprints are required as the starting points for generating any building inventory using brails. The user can provide their own footprint data or they can use |app| to create an inventory. There are 3 scraper classes in |app| that can be used to obtain this from the web:

#. MS_FootprintScraper using `Microsoft Footprint Database <https://github.com/microsoft/USBuildingFootprints>`_
#. OSM_FootprintScraper using `Open Street Maps (OSM) <https://www.openstreetmap.org/about>`_
#. USA_FootprintScraper using `USA Structures <https://gis-fema.hub.arcgis.com/pages/usa-structures>`_

Each of these classes has a method **get_footprints** which will return the footprints for the **RegionBoundary** provided. In the example shown below, the 3 different classes are all used to generate an inventory for a location provided when the script is run.

.. literalinclude:: brails_footprint.py
   :language: python
   :linenos:

To run the example provided contained in a file example.py for for a "Berkeley, CA" location:
      
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

The example when run will also prints out a two inventory subset of the data obtained for each footprint scraper. As shown below, the features obtained for each of the assets is different between scrapers.

.. literalinclude:: output.txt
   :linenos:
      
.. note::

   #. The number of buildings differs across datasets due to variations in data sources, processing techniques, geographic coverage, and update frequency. Since no dataset is perfect, users are encouraged to compare building inventories from different datasets for their area of interest by overlaying them with satellite imagery for accuracy.
   #. OSM is a **community-driven** platform where volunteers manually contribute building footprints using ground surveys, GPS data, and licensed aerial imagery. This results in variable data quality depending on the region and contributor activity, with a particularly active community in non-urban areas. The results may contain **NA** values for buildings for which community has yet to provide data.
   #. OSM offers global coverage, Microsoft’s data is expanding internationally, and USA Structures is limited to the United States.
   #. OSM is updated in real time by contributors, Microsoft’s data is updated periodically, and USA Structures updates are infrequent.
      
   
