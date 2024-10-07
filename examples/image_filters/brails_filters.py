# Written: fmk 4/23
# License: BSD-2

"""
 Purpose: Testing Importer and get_footprints methods
"""

from brails.types.image_set import ImageSet
from brails.utils.utils import Importer
import os
import sys

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

# select a region and create a RegionBoundary
#

region_data = {"type": "locationName", "data": "Tiburon, CA"}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)


#
# get the inventory
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

small_inventory = usa_inventory.get_random_sample(20, 100)
print("num assets USA subset", len(small_inventory.inventory))


#
# Get street view images using GoogleStreetview
#

google_input = {"apiKey": apiKey}
google_street_class = importer.get_class("GoogleStreetview")
google_street = google_street_class(google_input)
images_street = google_street.get_images(small_inventory, "tmp/street/")

images_street.print()

#
# now filter
#

filter_house = importer.get_class("HouseView")
filter_data = {}
filter1 = filter_house(filter_data)
filter1.filter(images_street, "filtered_images")


# input_images = ImageSet();
# input_images.set_directory("./images/", True)
# filter_house = importer.get_class("HouseView")
# filter_data={}
# filter1 = filter_house(filter_data)
# filter1.filter(input_images, "filtered_images")
