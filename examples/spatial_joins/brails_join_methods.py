# Written: bacetiner 06/25
# License: BSD-2

"""
brails_join_methods.py
===================

This is a simple BRAILS example to demonstrate how spatial joins are performed
on AssetInventory objects
"""

from brails.utils import Importer
from brails.utils import SpatialJoinMethods

# Define inventory location:
LOCATION = "Tiburon, CA"

# Create an Importer instance:
importer = Importer()

# Create a region boundary:
region_boundary_class = importer.get_class('RegionBoundary')
region_boundary_object = region_boundary_class({'type': 'locationName',
                                                'data': LOCATION})

# Create an NSI point inventory:
nsi_class = importer.get_class('NSI_Parser')
nsi = nsi_class()
nsi_inventory = nsi.get_raw_data(region_boundary_object)

# Create a FEMA USA Structures footprint inventory:
fp_scraper_class = importer.get_class('OSM_FootprintScraper')
fp_scraper = fp_scraper_class({'length': 'ft'})
fp_inventory = fp_scraper.get_footprints(region_boundary_object)

# Merge NSI data with FEMA USA structures data:
merged_inventory = SpatialJoinMethods.execute('GetPointsInPolygons',
                                              fp_inventory,
                                              nsi_inventory)
