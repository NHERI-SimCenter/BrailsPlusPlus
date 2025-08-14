"""
Example demonstrating the use of OvertureMapsFootprintScraper.

This example shows how to:

- Initialize the brails Importer.
- Define a spatial region boundary for a location (Berkeley, CA).
- List available Overture Maps releases.
- Retrieve building footprints from the specified Overture Maps release.
- Save the retrieved footprints to a GeoJSON file.

Contributors:
Barbaros Cetiner

Last updated:
08-14-2025
"""

from brails import Importer

# Initialize the importer:
importer = Importer()

# Create a region boundary object for Berkeley, CA:
region_data = {"type": "locationName", "data": "Berkeley, CA"}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)

# Import OvertureMapsFootprintScraper class and get a list of Overture Maps
# releases:
scraper_class = importer.get_class("OvertureMapsFootprintScraper")
_ = scraper_class.fetch_release_names(print_releases=True)

# Get building inventory data using the '2024-07-22.0' release of Overture
# Maps:
scraper_object = scraper_class(input_dict={"overtureRelease": "2024-07-22.0"})
inventory = scraper_object.get_footprints(region_boundary_object)

# Write the obtained inventory to a GeoJSON file:
_ = inventory.write_to_geojson("berkeley_buildings.geojson")
