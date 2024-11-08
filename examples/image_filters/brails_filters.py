"""
brails_filters.py
================
Example showing the use of BRAILS module that crops building images from panos.

 Purpose: Testing 1) get_class method of Importer
                  2) get_footprints using scraper provided
                  3) get_images using StreetView for subset of footprints
                  4) filter method of HouseView
"""

import os
import argparse
from pathlib import Path
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
else:
    raise FileNotFoundError('API key file not found. Please ensure the file'
                            f' exists at: {API_KEY_DIR}')


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Demonstrate Importer.")
    parser.add_argument('scraper', type=str, help="Footprint Scraper")
    parser.add_argument('location', type=str, help="Location")    

    # Parse the arguments
    args = parser.parse_args()

    importer = Importer()

    region_boundary_class = importer.get_class("RegionBoundary")
    region_boundary_object = region_boundary_class({"type": "locationName", "data": args.location})
    scraper_class = importer.get_class(args.scraper)
    scraper = scraper_class({"length": "ft"})
    inventory = scraper.get_footprints(region_boundary_object)
    print(f"num assets found: {len(inventory.inventory)} for {args.location} using {args.scraper}")    

    # Subsample from the extracted assets to keep the image downloading step quick.
    # Here, we are randomly sampling 10 buildings using a random seed value of 100:
    small_inventory = inventory.get_random_sample(10, 100)


    # Get street level imagery for the selected subset using GoogleStreetview:
    google_street_class = importer.get_class('GoogleStreetview')
    google_street = google_street_class({'apiKey': api_key})
    images_street = google_street.get_images(small_inventory, 'tmp/street/')

    images_street.print_info()

    # Crop the obtained imagery such that they include just the buildings of
    # interest:
    filter_house = importer.get_class('HouseView')
    image_filter = filter_house({})
    filtered_images_street = image_filter.filter(images_street, 'tmp/filtered_images')
    print('\nCropped images are available in ',
          Path(filtered_images_street.dir_path).resolve())

    filtered_images_street.print_info()    

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()    
