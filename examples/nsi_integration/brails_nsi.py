# written: fmk, bacetiner
# Last updated: 03-12-2025
# License: BSD-2

"""
brails_nsi.py
=============
Example demonstrating the use of NSI scraper module.

 Purpose: Testing 1) get_class method of Importer
                  2) get_raw_data_given_boundary and
                     get_filtered_data_given_inventory methods of NSI_Parser
                  3) get_footprints method from scraper
                  4) print_info and write_to_geojson method of AssetInventory
                     objects
"""

import argparse
import sys
import os

from brails import Importer

# the following line is not neeeded if brails is imported from pypi
#   .. it is included here as it allows us to test the code on a nightly basis
sys.path.insert(1, "../../")

FILE_PATH = 'tmp/smallinv.geojson'


def main(location, scraper):
    # Create an importer to get the required classes:
    importer = Importer()

    # Select a region and create its RegionBoundary:
    region_boundary_class = importer.get_class('RegionBoundary')
    region_boundary_object = region_boundary_class(
        {'type': 'locationName', 'data': location})

    #
    # Use NSI_Parser to get the NSI raw data for the specified region
    #

    nsi_class = importer.get_class('NSI_Parser')
    nsi = nsi_class()
    nsi_inventory = nsi.get_raw_data(region_boundary_object)
    print('Total number of assets detected using NSI is '
          f'{len(nsi_inventory.inventory)}')

    print('\n******** SMALL NSI INVENTORY  *********')
    nsi_inventory_subset = nsi_inventory.get_random_sample(2, 200)
    nsi_inventory_subset.print_info()

    # Now Use the FootprintScraper to get footprints and integrate NSI
    # inventory data - will integrate on smaller inventory set:

    scraper_class = importer.get_class(scraper)
    scraper = scraper_class({'length': 'ft'})
    scraper_inventory = scraper.get_footprints(region_boundary_object)
    print(f'Total number of assets detected using {scraper} '
          f'is {len(scraper_inventory.inventory)}')

    # Create inventories that are a small subset of the obtained footprints &
    # print:
    inventory_subset = scraper_inventory.get_random_sample(5, 200)

    print('\n******** SMALL INVENTORY WITH NO NSI_DATA *********')
    inventory_subset.print_info()

    # integrate subset with raw and filtered NSI data
    subset_nsi_processed_data = nsi.get_filtered_data_given_inventory(
        inventory_subset, 'ft')

    # Print the inventories:
    print('\n******** SMALL INVENTORY WITH PROCESSES NSI DATA*********')
    subset_nsi_processed_data.print_info()

    # Write the extracted inventory to a GeoJSON file:
    directory = os.path.dirname(FILE_PATH)
    os.makedirs(directory, exist_ok=True)

    _ = subset_nsi_processed_data.write_to_geojson(FILE_PATH)


# Run the main function if this script is executed directly:
if __name__ == "__main__":
    # Set up command-line arguments:
    parser = argparse.ArgumentParser(description='Get NSI inventory for a '
                                     'location using the specified footprint '
                                     'scraper.')
    parser.add_argument('scraper', type=str, nargs='?',
                        default='OSM_FootprintScraper',
                        help="Name of the footprint scraper.")
    parser.add_argument('location', type=str, nargs='?', default='Tiburon, CA',
                        help="Name of the location to analyze.")

    # Parse the command-line arguments:
    args = parser.parse_args()
    print(args)
    try:
        # Run the main function
        main(args.location, args.scraper)
    except Exception as e:
        print(f'Error{e}')
