.. _imputation_exammple:

Imputation Algorithms
=====================

This is an example of how to use the BRAILS imputation algorithms to fill in missing data
in an inventory. In this example, the user has a csv file that they will use to create the asset inventory. That csv file
contains rows, some of which are complete and some of which have missing column values. For example, for some rows the roof shape attribute may be missing. An inventory is first created with this csv file, a knn imputer is created, and this imputer returns a second inventory, which for the missing fields will contain a number of possible values.

.. literalinclude:: imputation.py
   :language: python
   :linenos:


Imputation Notebook
-------------------

Below is a link to a Jupyter notebook that runs this basic code, with graphics to better understand the
output.

.. raw:: html
	 
	 <a href=https://colab.research.google.com/github/NHERI-SimCenter/BrailsPlusPlus/blob/master/examples/imputation/imputation_example.ipynb target=_blank> <img src=https://colab.research.google.com/assets/colab-badge.svg/></a>"

         <a href=https://lightning.ai/new?repo_url=https://github.com/NHERI-SimCenter/BrailsPlusPlus/blob/master/examples/imputation/imputation_example.ipynb target=_blank> <img src=https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg/></a>
	 
.. nbgallery:: ./imputation_example.ipynb

.. raw:: html

   <hr>

.. note::

      #. Imputation is a statistical technique used to handle missing data by replacing it with substituted values. In |app| the goal is to fill in the gaps in an inventory dataset to ensure that analyses can proceed without having to throw away assets from the inventory due to missing values. Imputation is used in many other fields like data science, machine learning, and statistics.

       #. There are a number of algorithms available that are used. These algorithms can produce single values for each missing data-point or they can produce a number of possible values for each missing data-point, e.g. **K-NearestNeighbour**. If multiple possible values are generated, the inventory has a field for the **#samples**. Each sample representing, what is termed, one **possible world**.

       #. When multiple options in the workflow generate a number of samples field in the inventory, any option that generates samples is expected to generate the same number of samples as existing overriding users request to the contrary.
