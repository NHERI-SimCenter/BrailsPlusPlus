"""
imputation.py
================

This is a simple BRAILS example to demonstrate imputating (estimating the
missing pieces) of an inventory dataset.

"""

import os
import sys
import json

# the following line is not neeeded if brails is imported from pypi
#   .. it is included here as it allows us to test the code on a nightly basis
sys.path.insert(1, "../../")

from brails.utils.utils import Importer
from brails.types.image_set import ImageSet
from brails.types.asset_inventory import Asset, AssetInventory
importer = Importer()

file_path = "./example_Tiburon.csv"
    
inventory = AssetInventory()
inventory.read_from_csv(file_path,keep_existing=True, id_column='index')


knn_imputer_class = importer.get_class("KnnImputer")
imputer=knn_imputer_class()
new_inventory = imputer.impute(inventory,n_possible_worlds=10)

new_inventory.print()


#
# Saving the imputed database into a geojson file 
#
geojson = new_inventory.get_geojson()
filepath = 'tmp/imputed_inventory.geojson'
directory = os.path.dirname(filepath)
if not os.path.exists(directory):
    os.makedirs(directory)
with open(filepath, "w") as f:
    json.dump(geojson, f, indent=2)
