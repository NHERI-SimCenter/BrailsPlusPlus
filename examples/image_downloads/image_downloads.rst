.. _image_downloads:

Downloading Images
==================

Many of the modules in |app| make use of images and for that reason modules are available to go get images. Currently two modules exist:

#. GoogleSatellite
#. GoogleStreetview

This following is an example of how to use both building. Building upon the examples for generating footprints, the examples take as input the footprint scraper and the location of interest. An **AssetInventory** is then generated. A random subset of this inventory is created and both satallite and street view images are obtained for this inventory.

.. literalinclude:: brails_download_images.py
   :language: python
   :linenos:

.. note::

   #. To run the script you will need to obtain a Google API key.
   #. Downloding of images can cost you money.
   #. The downloading of images and processing of such takes time and resources. This is why the **get_random_sample()** method. For assets for which processed data is subsequently missing, **data imputation** can be employed to fill in missng data.

