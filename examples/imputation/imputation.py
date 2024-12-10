# Written: sy Aug 2024
# License: BSD-2

"""
imputation.py
================

This is a simple BRAILS++ example to demonstrate imputating (estimating the
missing pieces) of an inventory dataset.

"""

import os
import sys
import json

from brails.utils.importer import Importer
from brails.types.image_set import ImageSet
from brails.types.asset_inventory import Asset, AssetInventory


# create the importer
importer = Importer()

#
# create an asset invenntory from the contents of a csv file
#

file_path = "./example_Tiburon.csv"
    
inventory = AssetInventory()
inventory.read_from_csv(file_path,keep_existing=True, id_column='index')

#
# its not perfect, in sense it contains missing data as shown for 4th asset
#


print(f'INCOMPLETE ASSET: {inventory.get_asset_features(4)[1]}')

knn_imputer_class = importer.get_class("KnnImputer")
imputer=knn_imputer_class(inventory,n_possible_worlds=10)
new_inventory = imputer.impute()

#
# Saving the imputed database into a geojson file 
#

filepath = 'tmp/imputed_inventory.geojson'
directory = os.path.dirname(filepath)
if not os.path.exists(directory):
    os.makedirs(directory)
    
new_inventory.write_to_geojson(filepath)

print(f'COMPLETE ASSET: {new_inventory.get_asset_features(4)[1]}')

