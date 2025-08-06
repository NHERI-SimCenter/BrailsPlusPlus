"""
Example demonstrating how to use RAPIDUtils with orthomosaic datasets.

This example covers how to:

- Load an orthomosaic raster dataset.
- Define a spatial region boundary from the raster extent.
- Retrieve building footprints from OpenStreetMap within that region.
- Extract aerial image patches for each building footprint,
  with optional overlay of building outlines.

Contributors:
Barbaros Cetiner

Last updated:
08-06-2025
"""

from brails.utils import RAPIDUtils, Importer

rapid_utils = RAPIDUtils('DIRECTORY-PATH-OF-RASTER-FILE-HERE')

# Instantiate the Importer to dynamically load various BRAILS classes:
importer = Importer()

# Define the region of interest using the extent of the raster image as a
# polygon:
region_data = {
    "type": "locationPolygon",
    "data": rapid_utils.dataset_extent
}

# Create a region boundary object from the raster extent polygon:
region_boundary_object = importer.get_class("RegionBoundary")(region_data)

# Load the OpenStreetMap (OSM) footprint scraper class and initialize it
# with the desired length unit (feet in this case):
osm_footprint_scraper = importer.get_class('OSM_FootprintScraper')(
    {'length': 'ft'}
)

# Retrieve building footprints within the defined region boundary:
scraper_inventory = osm_footprint_scraper.get_footprints(
    region_boundary_object
)

# Extract aerial imagery patches for each building footprint from the raster
# image. The extracted images are saved to the specified output folder,
# with an optional overlay of asset outlines:
image_set = rapid_utils.extract_aerial_imagery(
    scraper_inventory,
    'images_raster_test/overlaid_imagery',
    overlay_asset_outline=True
)
