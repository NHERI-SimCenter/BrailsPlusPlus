.. _filter_example:

Image Filters
=============

Image filters are modules for cleaning up images prior to processing them for information by a processor. Removing unwanted crap from an image can help the processors work better. The following example demonstrates this. It uses the users chosen footprin scraper to obtain start an assetInventory. It then obtains Google Streev view images for a subset of the inventory and then applies a House flter, which attempts to isolate the house in the center of the downloadded image. It returns a new image.


The following example shows a more typical python example. 

.. literalinclude:: brails_filters.py
   :language: python
   :linenos:

To run for example the **importer.py** script for Berkeley, CA the following would be issued frm a terminal window:

.. code-block::
      
   python3 brails_filter.py USA_FootprintScraper "Berkeley, CA"


and the application would produce two ImageSets:

.. literalinclude:: output.txt
   :linenos:   

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
   * - .. image:: images/filtered_images/gstrt_37.84910645_-122.27443686.jpg
     - .. image:: images/filtered_images/gstrt_37.85885402_-122.25080419.jpg
     - .. image:: images/filtered_images/gstrt_37.86096388_-122.27096762.jpg

   * - .. image:: images/filtered_images/gstrt_37.86240981_-122.29162218.jpg
     - .. image:: images/filtered_images/gstrt_37.86325404_-122.27462417.jpg
     - .. image:: images/filtered_images/gstrt_37.86369654_-122.26024302.jpg

   * - .. image:: images/filtered_images/gstrt_37.87800845_-122.27387780.jpg
     - .. image:: images/filtered_images/gstrt_37.87979406_-122.27365989.jpg
     - .. image:: images/filtered_images/gstrt_37.88182298_-122.26462540.jpg
   
.. note::


	 
   
      

      


