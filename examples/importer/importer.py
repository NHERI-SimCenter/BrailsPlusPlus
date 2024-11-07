# Written: fmk 09/24
# License: BSD-2

"""
importer.py
================

This is a simple example to demonstrate the Importer class in BRAILS++

"""

import argparse
import sys

# the following line is not neeeded if brails is imported from pypi
#   .. it is included here as it allows us to test the code on a nightly basis
sys.path.insert(1, "../../")


from brails.utils.importer import Importer


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
    

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
    
