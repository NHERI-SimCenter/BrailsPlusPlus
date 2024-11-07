# Written: fmk 4/23
# License: BSD-2

import argparse

from brails.types.asset_inventory import AssetInventory
from brails.types.region_boundary import RegionBoundary
from brails.scrapers.osm_footprint_scraper.osm_footprint_scraper import OSM_FootprintScraper
from brails.scrapers.ms_footprint_scraper.ms_footprint_handler import MS_FootprintScraper
from brails.scrapers.usa_footprint_scraper.usa_footprint_scraper import USA_FootprintScraper

def main():
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Demonstrate Importer.")
    parser.add_argument('scraper', type=str, help="Location")        
    parser.add_argument('location', type=str, help="Location")

    # Parse the arguments
    args = parser.parse_args()

    # create a RegionBoundary
    region_boundary_object = RegionBoundary({"type": "locationName", "data": args.location})

    scraper_type = args.scraper
    if scraper_type == "OSM_FootprintScraper":
        scraper = OSM_FootprintScraper({"length": "ft"})
    elif scraper_type == "MS_FootprintScraper":
        scraper = MS_FootprintScraper({"length": "ft"})
    elif scraper_type == "USA_FootprintScraper":
        scraper = USA_FootprintScraper({"length": "ft"})
    else:
        print(f"Unknown Scraper Type: {scraper_type}")
        
    inventory = scraper.get_footprints(region_boundary_object)

    print(f"num assets found: {len(inventory.inventory)} for {args.location} using {args.scraper}")

    small_inventory = inventory.get_random_sample(4, 200)
    small_inventory.print_info()
    
# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
