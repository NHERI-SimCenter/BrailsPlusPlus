"""
Filter a county inventory to a place.

This example keeps the buildings of a county inventory that lie inside a place
and writes them to a new GeoJSON file. It takes a county inventory of the
SimCenter San Francisco Bay Area testbed, or any other inventory that BRAILS++
wrote with ``write_to_geojson``, the place name and the output file:

    python filter_inventory_to_place.py county.geojson "Berkeley, CA" out.geojson

A building is kept when the centroid of its footprint lies inside the place's
boundary. A boundary looked up by name has no holes, so the buildings of an
enclave, such as Piedmont inside Oakland, are kept. Where holes matter, pass
the boundary to ``filter_to_region`` as a Shapely polygon instead of the name.

R2D reads the integer-id form of the file, ``write_to_geojson(...,
integer_ids=True)``, which needs a ``type`` on every building. The county
files carry none, so set ``asset.features['type'] = 'building'`` for each asset
of ``inventory.inventory.values()`` before that write. A file already in the
integer-id form cannot be written in that form again after filtering.

Close the output file in any other program, such as QGIS, before the script
writes it again.
"""

import argparse

from brails.utils.region_filter import filter_to_region

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep the buildings of an inventory that lie inside a place.'
    )
    parser.add_argument('county_file', help='the GeoJSON inventory to filter')
    parser.add_argument('place', help='the place name, such as "Berkeley, CA"')
    parser.add_argument('output_file', help='the GeoJSON file to write')
    args = parser.parse_args()

    # Fetch the boundary of the place and keep the buildings inside it:
    inventory = filter_to_region(args.county_file, args.place)

    # Write the buildings inside the place to a GeoJSON file:
    inventory.write_to_geojson(args.output_file)
