"""
Example demonstrating the capabilities of the USGSElevationScraper module.

This example showcases how to:
- Define a spatial region boundary using a location name.
- Retrieve NSI inventory and building footprint data for that region.
- Filter and enrich inventory data based on available building footprints.
- Fetch elevation data from the USGS Elevation Point Query Service for each
  asset using multiple statistical modes (e.g., centroid, average, min, max,
  etc.).
- Save the elevation-enriched asset inventory to a GeoJSON file.
- Fetch and save elevation surface data for the entire region by sampling
  points.

Contributors:
Barbaros Cetiner

Last updated:
08-06-2025
"""

# Import the Importer utility to dynamically load different data processing
# classes:
from brails.utils import Importer

# Define the target location name and output file paths:
LOCATION_NAME = 'Fort Myers Beach, FL'
INVENTORY_OUTPUT = 'FortMyersInventory_Elevation.geojson'
ELEVATION_SURFACE_OUTPUT = 'FortMyersElevationSurface.geojson'

# Instantiate the Importer class to load modules dynamically:
importer = Importer()

# Create a region boundary object using the location name:
region_data = {"type": "locationName", "data": LOCATION_NAME}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)

# Load and use the NSI_Parser class to retrieve the NSI inventory data
nsi_scraper = importer.get_class('NSI_Parser')()
nsi_inventory = nsi_scraper.get_raw_data(region_boundary_object)

# Load and use the USA_FootprintScraper to obtain building footprints for the
# region:
fema_usa_scraper = importer.get_class('USA_FootprintScraper')({'length': 'ft'})
fema_usa_inventory = fema_usa_scraper.get_footprints(region_boundary_object)

# Filter NSI inventory based on available building footprints and enrich with
# additional features:
nsi_inventory = nsi_scraper.get_filtered_data_given_inventory(
    fema_usa_inventory,
    "ft",  # Units
    get_extended_features=True,  # Whether to retrieve additional features
    add_features=[]  # No extra features specified
)

# Load the USGS elevation scraper to enrich asset data with elevation
# information:
usgs_elevation_scraper = importer.get_class('USGSElevationScraper')()

# Define the different statistical modes for elevation calculation:
modes = ['centroid', 'all', 'average', 'min', 'max', 'median', 'stddev']

# Get elevation data for each asset in the inventory using multiple statistical
# modes:
asset_inventory = usgs_elevation_scraper.get_asset_elevation_data(
    asset_inventory=nsi_inventory,
    modes=modes
)

# Write the updated asset inventory (with elevation data) to a GeoJSON file:
_ = asset_inventory.write_to_geojson(INVENTORY_OUTPUT)

# Separately fetch elevation data for the entire region (not per asset),
# sampling 2000 points:
region_elevation_data = usgs_elevation_scraper.get_region_elevation_data(
    region=region_boundary_object,
    num_points=2000
)

# Write the elevation surface data for the region to another GeoJSON file:
_ = region_elevation_data.write_to_geojson(ELEVATION_SURFACE_OUTPUT)
