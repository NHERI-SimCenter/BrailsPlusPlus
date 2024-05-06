# Written: fmk 4/23
# License: BSD-2

"""
 Purpose: Testing Importer and get_footprints methods
"""

import sys
import copy

sys.path.insert(1, "../../")

from brails.utils.utils import Importer

# create an Import to get the classes

importer = Importer()

#
# specify the BoundaryRegion & then use an NSI_Parser to get geojson
#


region_data = {"type": "locationName", "data": "Tiburon, CA"}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)

nsi_class = importer.get_class("NSI_Parser")
nsi = nsi_class()
inventory = nsi.get_raw_data_given_boundary(region_boundary_object, 'ft')
inventory.print()


#
# Get Footprints using OSM
#

print("Trying OSM_FootprintsScraper ...")

osm_class = importer.get_class("OSM_FootprintScraper")
osm_data = {"length": "ft"}
osm = osm_class(osm_data)
osm_inventory = osm.get_footprints(region_boundary_object)

small_inventory = osm_inventory.get_random_sample(3, 200)
copy_inventory = copy.deepcopy(small_inventory)

print('\n******** SMALL INVENTORY *********')
small_inventory.print()

print('\n******** NSI RAW DATA FOR SMALL INVENTORY *********')
small_inventory = nsi.get_raw_data_given_inventory(small_inventory, 'ft')
small_inventory.print()

print('\n******** NSI FILTERED DATA FOR SMALL INVENTORY *********')
#another_small_inventory = osm_inventory.get_random_sample(4, 200)
copy_inventory = nsi.get_filtered_data_given_inventory(copy_inventory, 'ft')
copy_inventory.print()

