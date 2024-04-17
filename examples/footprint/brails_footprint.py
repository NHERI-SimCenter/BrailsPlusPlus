# Written: fmk 4/23
# License: BSD-2

"""
 Purpose: Testing Importer and get_footprints methods
"""

import sys

sys.path.insert(1, "../../")

from brails.utils.utils import Importer



importer = Importer()

region_data = {"type": "locationName", "data": "Tiburon, CA"}

region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)

print("Trying OSM_FootprintsScraper")
osm_class = importer.get_class("OSM_FootprintScraper")
osm_data = {"length": "ft"}
instance1 = osm_class(osm_data)
osm_inventory = instance1.get_footprints(region_boundary_object)

print("num assets OSM", len(osm_inventory.inventory))

print("Trying USA_FootprintsScraper")
usa_class = importer.get_class("USA_FootprintScraper")
usa_data = {"length": "ft"}
instance2 = usa_class(usa_data)
usa_inventory = instance2.get_footprints(region_boundary_object)

print("num assets USA", len(usa_inventory.inventory))

# just get images for a small subset of the assets
small_inventory = usa_inventory.get_random_sample(4, 200)
print("num assets USA seubset", len(small_inventory.inventory))
