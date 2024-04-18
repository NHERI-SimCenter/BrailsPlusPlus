# Written: fmk 4/23
# License: BSD-2

"""
 Purpose: Testing Importer and get_footprints methods
"""

import os
import sys

sys.path.insert(1, "../../")

from brails.utils.utils import Importer


#
# This script needs an Google API Key to run
#   -- suggest placing in file, here apiKey.txt, if you plan to commit as you don't want to make a mistake
#   -- apiKey.txt is in .gitignore so you have work to do to get it uploaded

apiKey = ""
if os.path.exists("apiKey.txt"):
    with open("apiKey.txt", "r") as file:
        apiKey = file.readline().strip()  # Read the first line and strip whitespace


#
# create the importer
#

importer = Importer()

#
# select a region and create a RegionBoundary
#

region_data = {"type": "locationName", "data": "Tiburon, CA"}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)

#
# Get AsetInventory for buildings in region using USA_FootprintScraper
#

print("Trying USA_FootprintsScraper ...")

usa_class = importer.get_class("USA_FootprintScraper")
usa_data = {"length": "ft"}
instance2 = usa_class(usa_data)
usa_inventory = instance2.get_footprints(region_boundary_object)

print("num assets USA", len(usa_inventory.inventory))

#
# Make the Inventory smaller as we are pulling images
#    - (4 buildings with a seed of 200)

small_inventory = usa_inventory.get_random_sample(10, 40)
print("num assets USA subset", len(small_inventory.inventory))

#
# Get satellite images using GoogleSatellite
#

google_satellite_class = importer.get_class("GoogleSatellite")
google_input = {"apiKey": apiKey}
google_satellite = google_satellite_class(google_input)
images_satellite = google_satellite.get_images(small_inventory, "tmp/satellite/")

images_satellite.print()

#
# Get street view images using GoogleStreetview
#

google_street_class = importer.get_class("GoogleStreetview")
google_street = google_street_class(google_input)
images_street = google_street.get_images(small_inventory, "tmp/street/")

images_street.print()
