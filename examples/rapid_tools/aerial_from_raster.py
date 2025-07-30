"""
Example demonstrating how to use RAPIDUtils with orthomosaic datasets.

This example covers how to:

- Load an orthomosaic raster dataset.
- Define a spatial region boundary from the raster extent.
- Retrieve building footprints from OpenStreetMap within that region.
- Extract aerial image patches for each building footprint,
  with optional overlay of building outlines.

# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 07-30-2025
"""

from brails.utils import RAPIDUtils, Importer

rapid_utils = RAPIDUtils('DIRECTORY-PATH-OF-RASTER-FILE-HERE')

importer = Importer()

region_data = {"type": "locationPolygon", "data": rapid_utils.dataset_extent}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)

scraper_class = importer.get_class('OSM_FootprintScraper')
scraper = scraper_class({'length': 'ft'})
scraper_inventory = scraper.get_footprints(region_boundary_object)

image_set = rapid_utils.extract_aerial_imagery(
    scraper_inventory,
    'images_raster_test/overlaid_imagery',
    overlay_asset_outline=True
)
