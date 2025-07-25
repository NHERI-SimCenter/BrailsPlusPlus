.. _image_downloads:

Downloading Images
==================

Many modules in |app| rely on images, and to facilitate this, dedicated modules are available for image retrieval. Currently, the following two modules are supported:

#. GoogleSatellite
#. GoogleStreetview

The example below demonstrates how to use both modules. Building on the footprint generation examples, this workflow takes a footprint scraper and a specified location as input. An **AssetInventory** is generated, from which a random subset is selected. Satellite and street-view images are then retrieved for the selected inventory.

.. literalinclude:: brails_download_images.py
   :language: python
   :linenos:

.. note::

   #. To run the script you will need to obtain a Google API key.
   #. There are no costs associated with downloading images using these modules.
   #. Downloading and processing images requires time and computational resources. To mitigate this for test runs, the **get_random_sample()** method can be used to select a subset of assets. Subsequently, for any assets with missing processed data, **data imputation** techniques can be applied to address the gaps.

