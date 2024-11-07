# Written: fmk 4/23
# License: BSD-2

"""
Example showcasing BRAILS' image downloading capabilities.

 Purpose: Testing 1) get_class method of Importer
                  2) get_footprints method of USA_FootprintScraper module
                  3) get_images and print_info methods of GoogleSatellite and
                     GoogleStreetview

"""

import os
import argparse
from brails import Importer

# This script needs a Google API Key to run.
# We suggest placing your API key in file apiKey.txt in the same directory as
# this script if you plan to commit changes to this example. This way, you do
# not risk accidentally uploading your API key (apiKey.txt is in .gitignore,
# so you have work to do to get it uploaded)

API_KEY_DIR = '../api_key.txt'
if os.path.exists(API_KEY_DIR):
    with open(API_KEY_DIR, 'r', encoding='utf-8') as file:
        api_key = file.readline().strip()  # Read first line & strip whitespace


from brails.utils.importer import Importer

def download_images():
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Demonstrate Importer.")
    parser.add_argument('scraper', type=str, help="Footprint Scraper")
    parser.add_argument('location', type=str, help="Location")    

    # Parse the arguments
    args = parser.parse_args()

    # Create the importer:
    importer = Importer()

    
    # Select a region and create its RegionBoundary:
    region_boundary_class = importer.get_class('RegionBoundary')
    region_boundary_object = region_boundary_class({'type': 'locationName', 'data': args.location})

    scraper_class = importer.get_class(args.scraper)    
    scraper = scraper_class({"length": "ft"})
    inventory = scraper.get_footprints(region_boundary_object)
    print(f"num assets found: {len(inventory.inventory)} for {args.location} using {args.scraper}")
    
    # Subsample from the extracted assets to keep the image downloading step quick.
    # Here, we are randomly sampling 20 buildings using a random seed value of 40:
    small_inventory = inventory.get_random_sample(20, 40)

    # Get aerial imagery for the selected subset using GoogleSatellite:
    google_satellite_class = importer.get_class('GoogleSatellite')
    google_satellite = google_satellite_class()
    images_satellite = google_satellite.get_images(small_inventory,
                                                   'tmp/satellite/')

    # Get street level imagery for the selected subset using GoogleStreetview:
    google_street_class = importer.get_class('GoogleStreetview')
    google_input = {'apiKey': api_key}
    google_street = google_street_class(google_input)
    images_street = google_street.get_images(small_inventory, 'tmp/street/')

    inventory.print_info()
#    small_inventory.print_info()
#    images_satellite.print_info()
#    images_street.print_info()

# Run the main function if this script is executed directly
if __name__ == "__main__":
    download_images()    
