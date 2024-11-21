# Written: fmk 4/23
# Modified: bacetiner 11/21
# License: BSD-3

"""
brails_download_images.py
=========================

Purpose:
1) Test the `get_class` method of the Importer module.
2) Test the `get_footprints` method of the scraper modules.
3) Test the `get_images` and `print_info` methods of GoogleSatellite and
    GoogleStreetview.

This script demonstrates the use of BRAILS modules to download images for a
specified region.
"""

import os
import argparse
from brails.utils.importer import Importer


# Function to load the API key from a file:
def load_api_key(api_key_path):
    """
    Load the API key from the specified file.

    Args:
        api_key_path (str):
            Path to the file containing the API key.

    Returns:
        str:
            Google API key for accessing street-level image metadata.

    Raises:
        FileNotFoundError:
            If the API key file does not exist.
        ValueError:
            If the API key file is empty.
    """
    if not os.path.exists(api_key_path):
        raise FileNotFoundError(f"API key file not found at {api_key_path}")

    with open(api_key_path, 'r', encoding='utf-8') as file:
        api_key = file.readline().strip()

    if not api_key:
        raise ValueError("API key file is empty.")

    return api_key


# Main function for downloading images:
def download_images(location, scraper, api_key):
    """
    Download aerial/street-level images for a location using specified scraper.

    Args:
        api_key (str):
            Google API key for accessing street-level image metadata.
        scraper (str):
            Name of the footprint scraper to use.
        location (str):
            Name of the location to analyze.
    """
    # Create the importer:
    importer = Importer()

    # Select a region and create its RegionBoundary:
    region_boundary_class = importer.get_class('RegionBoundary')
    region_boundary_object = region_boundary_class(
        {'type': 'locationName', 'data': location})

    scraper_class = importer.get_class(scraper)
    scraper = scraper_class({"length": "ft"})
    inventory = scraper.get_footprints(region_boundary_object)
    print(f'Number of assets found: {len(inventory.inventory)} for {location} '
          'using {scraper}')

    # Subsample the assets for quick processing:
    small_inventory = inventory.get_random_sample(20, 40)

    # Get aerial imagery using GoogleSatellite:
    google_satellite_class = importer.get_class('GoogleSatellite')
    google_satellite = google_satellite_class()
    images_satellite = google_satellite.get_images(
        small_inventory, 'tmp/satellite/')

    # Get street-level imagery using GoogleStreetview:
    google_street_class = importer.get_class('GoogleStreetview')
    google_street = google_street_class({'apiKey': api_key})
    images_street = google_street.get_images(small_inventory, 'tmp/street/')

    # Print inventory info
    inventory.print_info()

    return small_inventory, images_satellite, images_street


# Entry point
if __name__ == "__main__":
    # Default API key file path:
    API_KEY_DIR = '../api_key.txt'

    # Set up command-line arguments:
    parser = argparse.ArgumentParser(description='Download images for a '
                                     'location using the specified footprint '
                                     'scraper.')
    parser.add_argument('scraper', type=str, nargs='?',
                        default='USA_FootprintScraper',
                        help="Name of the footprint scraper.")
    parser.add_argument('location', type=str, nargs='?', default='Tiburon, CA',
                        help="Name of the location to analyze.")
    parser.add_argument('--api_key_path', type=str, default=API_KEY_DIR,
                        help="Path to the Google API key file.")

    # Parse the command-line arguments:
    args = parser.parse_args()
    print(args)
    try:
        # Load the API key:
        parsed_api_key = load_api_key(args.api_key_path)

        # Run the main function
        fp_inventory, aerial_im, street_im = download_images(
            args.location, args.scraper, parsed_api_key)
    except Exception as e:
        print(f'Error{e}')
