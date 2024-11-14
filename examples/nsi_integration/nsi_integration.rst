.. _nsi_integration_exammple:

NSI Integration
===============

This is an example to demonstrate how |app| integrates with the `National Structures Inventory <https://www.hec.usace.army.mil/confluence/nsi>`_. There are two ways to integrate:

#. Create an inventory that does not include footprint information.
#. Integrate with an existing inventory. For each footprint associated with an asset in the existing inventory, the features from NSI are obtained and merged with the asset's existing features.


.. literalinclude:: brails_nsi.py
   :language: python
   :linenos:


The script is executed by entering the following command in a terminal window:

.. code-block::
      
   python3 brails_nsi.py OSM_FootprintScraper "Berkeley, CA"

As shown in the print_output() of the smaller NSI inventory, the coordinates for such an inventory only contain points instead of the building footprint.

.. literalinclude:: outputNSI.txt
   :linenos:

The output also demonstrates that the number of buildings in the two inventories, nsi_inventory and scraper_inventory, are different:

.. code-block::

   Total number of assets detected using NSI is 27705
   Total number of assets detected using OSM_FootprintScraper is 35547

As there are different numbers of buildings, when integrating NSI dataset into the footprint inventory, there will be assets for which no NSI data exists. In the merge performed with the example run, 2 out of the 5 assets in the subset inventories do not have data available in NSI.  This is shown in the output lines:

.. code-block::

The original and integrated inventories are as shown:   

.. literalinclude:: outputINTEGRATION.txt
   :linenos:   


NSI Integration Notebook
------------------------

Here is a link to a Jupyter Notebook that runs the basic code, accompanied by graphics to better illustrate the output.

.. raw:: html
	 
	 <a href=https://colab.research.google.com/github/NHERI-SimCenter/BrailsPlusPlus/blob/master/examples/nsi_integration/brails_nsi_integration.ipynb target=_blank> <img src=https://colab.research.google.com/assets/colab-badge.svg/></a>

         <a href=https://lightning.ai/new?repo_url=https%3A//github.com/NHERI-SimCenter/BrailsPlusPlus/blob/master/examples/nsi_integration/brails_nsi_integration.ipynb target=_blank> <img src=https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg></a>

      
.. note::

      #. Information on the fields output for NSI can be found `here <https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/latest/technical-documentation#id-.TechnicalDocumentationv2022-NSIPublicFields>`_
      #. When the number of buildings in the NSI differs from the inventory it is being merged with, **imputation** may be required during the integration process.
      #. The NSI is new and under development and as a consequence is not perfect.	 
