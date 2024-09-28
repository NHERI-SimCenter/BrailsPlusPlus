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

.. nbgallery:: ./imputation_example.ipynb

.. raw:: html

   <hr>
