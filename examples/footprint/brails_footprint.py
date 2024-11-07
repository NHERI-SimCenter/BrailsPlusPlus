# Written: fmk 4/23
# License: BSD-2

"""
brails_footprint.py
===================

This is a simple NRAILS example to demonstrate different methods to obtain
building inventories for an area.

"""

import argparse
import sys

sys.path.insert(1, "../../")

from brails.utils.importer import Importer

def main():
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Demonstrate Importer.")
    parser.add_argument('location', type=str, help="Location")    

    # Parse the arguments
    args = parser.parse_args()
    
    # create an Import to get the classes
    
    importer = Importer()

    #
    # specify the BoundaryRegion
    #
    
    region_boundary_class = importer.get_class("RegionBoundary")
    region_boundary_object = region_boundary_class({"type": "locationName", "data":args.location})
    
    #
    # Get Footprints using OSM
    #
    
    print("Using OSM_FootprintsScraper ...")
    
    osm_class = importer.get_class("OSM_FootprintScraper")
    osm_scraper = osm_class({"length": "ft"})
    osm_inventory = osm_scraper.get_footprints(region_boundary_object)
    
    #
    # Get Footprints using Microsofts Database
    #
    
    print("Using Microsoft Footprint Database ...")
    ms_class = importer.get_class("MS_FootprintScraper")
    ms_scraper = ms_class({"length": "ft"})
    ms_inventory = ms_scraper.get_footprints(region_boundary_object)

    #
    # Get Footprints using USA Structures
    #
    
    print("Using USA_FootprintsScraper ...")
    usa_class = importer.get_class("USA_FootprintScraper")
    usa_scraper = usa_class({"length": "ft"})
    usa_inventory = usa_scraper.get_footprints(region_boundary_object)

    #
    # Print num buildings found
    #

    print("\n\n")
    print("-" * 27)
    print(f"{'Scraper':<15}  {'# building':<10}")
    print("-" * 27)    
    print(f"{'OSM':<15}  {len(osm_inventory.inventory):<10}")    
    print(f"{'Microsoft':<15}  {len(ms_inventory.inventory):<10}")
    print(f"{'USA':<15}  {len(usa_inventory.inventory):<10}")
    print("-" * 27)    

    #
    # Test obtaining a smaller random subset of each,
    #    method needed as we will not always want to get all the images
    #    print to see what we are getting from each

    print("\n\nSmall Subset of USA Inventory: ")
    small_inventory = usa_inventory.get_random_sample(2, 200)
    small_inventory.print_info()

    print("\n\nSmall Subset of OSM Inventory: ")
    small_inventory = osm_inventory.get_random_sample(2, 200)
    small_inventory.print_info()

    print("\n\nSmall Subset of MS Inventory: ")
    small_inventory = ms_inventory.get_random_sample(2, 200)
    small_inventory.print_info()
    

if __name__ == "__main__":
    main()
