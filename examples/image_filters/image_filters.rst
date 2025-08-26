.. _filter_example:

Image Filters
=============

Image filters are modules designed to clean images before they are processed for information extraction. Removing unwanted elements from an image can improve the accuracy and efficiency of the processing algorithms.

The following example demonstrates this process: it uses the user’s chosen footprint scraper to create an :class:`~brails.types.asset_inventory.AssetInventory`. Then, it retrieves Google Street View images for a subset of the inventory and applies a :class:`~brails.filters.house_view.house_view.HouseView` filter, which attempts to isolate the house located at the center of each downloaded image. The filter returns a new, cleaned image.

A typical Python example is shown below:

.. literalinclude:: brails_filters.py
   :language: python
   :linenos:

To run, for example, the **brails_filters.py** script for Berkeley, CA, execute the following command in a terminal window:

.. code-block::
      
   python3 brails_filter.py USA_FootprintScraper "Berkeley, CA"


and the application would produce two :class:`~brails.types.image_set.ImageSet` objects:  

Raw Images

.. list-table::
   :widths: 33 33 33
   :header-rows: 0

   * - .. image:: images/street/images/gstrt_37.84910645_-122.27443686.jpg
     - .. image:: images/street/images/gstrt_37.85885402_-122.25080419.jpg
     - .. image:: images/street/images/gstrt_37.86096388_-122.27096762.jpg

   * - .. image:: images/street/images/gstrt_37.86240981_-122.29162218.jpg
     - .. image:: images/street/images/gstrt_37.86325404_-122.27462417.jpg
     - .. image:: images/street/images/gstrt_37.86369654_-122.26024302.jpg

   * - .. image:: images/street/images/gstrt_37.87800845_-122.27387780.jpg
     - .. image:: images/street/images/gstrt_37.87979406_-122.27365989.jpg
     - .. image:: images/street/images/gstrt_37.88182298_-122.26462540.jpg

Filtered Images:   

.. list-table::
   :widths: 33 33 33
   :header-rows: 0
   
   * - .. image:: images/filtered_images/gstrt_37.84910645_-122.27443686.jpg
     - .. image:: images/filtered_images/gstrt_37.85885402_-122.25080419.jpg
     - .. image:: images/filtered_images/gstrt_37.86096388_-122.27096762.jpg

   * - .. image:: images/filtered_images/gstrt_37.86240981_-122.29162218.jpg
     - .. image:: images/filtered_images/gstrt_37.86325404_-122.27462417.jpg
     - .. image:: images/filtered_images/gstrt_37.86369654_-122.26024302.jpg

   * - .. image:: images/filtered_images/gstrt_37.87800845_-122.27387780.jpg
     - .. image:: images/filtered_images/gstrt_37.87979406_-122.27365989.jpg
     - .. image:: images/filtered_images/gstrt_37.88182298_-122.26462540.jpg


	 
   
      

      


