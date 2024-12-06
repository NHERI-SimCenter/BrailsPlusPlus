# Written: fmk 4/24
# Modified: bacetiner 12/24
# License: BSD-2

"""
brails_footprint.py
===================

This is a simple BRAILS example to demonstrate different methods to obtain
building inventories for an area.

"""
import argparse
from brails.utils.importer import Importer


def download_footprints(location):
    """
    Download and analyze building footprint data for a specified location.

    Args:
        location (str):
            The name of the geographic region for which building footprints
            are to be downloaded. This is used to specify the boundary region.

    Functionality:
        1. Initializes a boundary region for the given location.
        2. Retrieves building footprints from:
           - OpenStreetMap using the `OSM_FootprintScraper`.
           - Microsoft Building Footprints data using the
            `MS_FootprintScraper`.
           - FEMA USA Structures database using the `USA_FootprintScraper`.
        3. Prints a summary of the number of buildings retrieved from each
           source.
        4. Extracts and prints details for a small random subset of the
           retrieved footprints from each data source for verification and
           sampling.

    Prints:
        - The number of buildings found in each dataset.
        - Information about small random subsets of the inventories for OSM,
          Microsoft, and FEMA USA Structures data sources.

    Note:
        The function uses an `Importer` class to dynamically load the required
        scraper classes.
        The random sampling is performed to give a quick overview of the data
        quality and coverage without loading the full dataset.
    """
    # Create an Importer to get the classes:
    importer = Importer()

    # Specify the BoundaryRegion:
    region_boundary_class = importer.get_class("RegionBoundary")
    region_boundary_object = region_boundary_class(
        {"type": "locationName", "data": location})

    # Get footprints using OpenStreetMap:
    print("Using OSM_FootprintsScraper...")

    osm_class = importer.get_class("OSM_FootprintScraper")
    osm_scraper = osm_class({"length": "ft"})
    osm_inventory = osm_scraper.get_footprints(region_boundary_object)

    # Get footprints using Microsoft Building Footprints data:
    print("\nUsing Microsoft Footprint Database...")

    ms_class = importer.get_class("MS_FootprintScraper")
    ms_scraper = ms_class({"length": "ft"})
    ms_inventory = ms_scraper.get_footprints(region_boundary_object)

    # Get footprints from FEMA USA Structures database:
    print("\nUsing USA_FootprintsScraper...")

    usa_class = importer.get_class("USA_FootprintScraper")
    usa_scraper = usa_class({"length": "ft"})
    usa_inventory = usa_scraper.get_footprints(region_boundary_object)

    # Print number of buildings found:
    seperator = '-' * 27
    print('\n\n')
    print(seperator)
    print(f"{'Scraper':<15}  {'# building':<10}")
    print(seperator)
    print(f"{'OSM':<15}  {len(osm_inventory.inventory):<10}")
    print(f"{'Microsoft':<15}  {len(ms_inventory.inventory):<10}")
    print(f"{'USA':<15}  {len(usa_inventory.inventory):<10}")
    print(seperator)

    # Test obtaining a smaller random subset of each, method needed as we will
    # not always want to get all the images. Print to see what we are getting
    # from each:
    print('\n\nSmall Subset of USA Inventory: ')
    small_inventory = usa_inventory.get_random_sample(2, 200)
    small_inventory.print_info()

    print('\n\nSmall Subset of OSM Inventory: ')
    small_inventory = osm_inventory.get_random_sample(2, 200)
    small_inventory.print_info()

    print('\n\nSmall Subset of MS Inventory: ')
    small_inventory = ms_inventory.get_random_sample(2, 200)
    small_inventory.print_info()


if __name__ == '__main__':
    # Set up command-line arguments:
    parser = argparse.ArgumentParser(description='Download footprints for a '
                                     'specified location using BRAILS++ '
                                     'footprint scrapers.')
    parser.add_argument('location', type=str, nargs='?', default='Tiburon, CA',
                        help="Name of the location to analyze.")

    # Parse the command-line arguments:
    args = parser.parse_args()

    # Get the building footprints for the specified location:
    download_footprints(args.location)
