"""
Filter a county inventory to a place.

This example keeps the buildings of a county inventory that lie inside a place
and writes them to a new GeoJSON file. It takes a county inventory of the
SimCenter San Francisco Bay Area testbed, or another GeoJSON inventory that
BRAILS++ wrote with ``write_to_geojson``, the place name and the output file:

    python filter_inventory_to_place.py county.geojson "Berkeley, CA" out.geojson

For a place that spans several counties, give each county's file before the
place name. The buildings inside the place are kept from every file into one
inventory:

    python filter_inventory_to_place.py a.geojson b.geojson "Place, CA" out.geojson

The place is a city, town or census-designated place, or a county, as the Census
Bureau's TIGER/Line files name it, written with its state, as in "Berkeley, CA" or
"Alameda County, CA". The script downloads the state's place file and, for a name
that ends in a word such as County or Island, the national county file. A
building is kept when the centroid of its footprint lies inside the place's
boundary. The boundary keeps its holes, so the buildings of an enclave, such as
Piedmont inside Oakland, are left out. For a place the census does not name, call
``filter_to_region(..., place_source='osm')`` from Python, which looks the name up in
OpenStreetMap.

For the integer-id form that R2D reads, write the result with
``write_to_geojson(..., integer_ids=True, asset_type='building')``. Close the
output file in any other program, such as QGIS, before the script writes it
again.
"""

import argparse

from brails.utils.region_filter import filter_to_region

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep the buildings of an inventory that lie inside a place.'
    )
    parser.add_argument(
        'county_files', nargs='+', help='one or more GeoJSON inventories to filter'
    )
    parser.add_argument(
        'place',
        help='the place or county with its state, such as "Berkeley, CA" or '
        '"Alameda County, CA"',
    )
    parser.add_argument('output_file', help='the GeoJSON file to write')
    args = parser.parse_args()

    # Fetch the boundary of the place and keep the buildings inside it:
    inventory = filter_to_region(args.county_files, args.place)

    # Write the buildings inside the place to a GeoJSON file:
    inventory.write_to_geojson(args.output_file)
