.. _nsi_integration_exammple:

NSI Integration
===============

This is an example to demonstrate how |app| integrates with the `National Structures Inventory <https://www.hec.usace.army.mil/confluence/nsi>`_. There are two ways to integrate:

#. Create a inventory, that lacks footprint information.
#. Integrate with an existing inventory. For this integration, for each footprint for which an asset in the NSI exists, the features in the NSI are added to the existing features of the asset.


.. literalinclude:: brails_nsi.py
   :language: python
   :linenos:


The script is run by issuing the following would be issued from a terminal window:

.. code-block::
      
   python3 brails_nsi.py OSM_FootprintScraper "Berkeley, CA"

As shown in the print_output() of the smaller nsi inventory, the coordinates for such an inventory only contain a point as opposed to the building footprint.

.. literalinclude:: outputNSI.txt
   :linenos:

The ouput also demonstrates that the number of buildings in the two inventories, nsi_inventory and scraper_inventory isdifferent:

.. code-block::

   Total number of assets detected using NSI is 27705
   Total number of assets detected using OSM_FootprintScraper is 35547

As there are different numbers of buildings, when integrating NSI dataset into the footprint inventory, there will be assets for which no NSI data exists. In the integration perfmormed with the example as run, 2 of the 5 Assets in the subset inventories have no data available. This is shown in the output lines:

.. code-block::

The original and integrated inventories are as shown:   

.. literalinclude:: outputINTEGRATION.txt
   :linenos:   
      
.. note::

      #. The NSI is new and under development and as a consequence not perfect.
      #. When the number of buildings is different and integration is used, **imputation** may be needed.
