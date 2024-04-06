# Written: fmk 03/24
# Purpose: to test Importer and the footprint handlers

import sys
import inspect
from brails import Importer

importer = Importer()

region_data = {
    "type":"locationName",
    "data":"Tiburon, CA"
}

region_boundary_class = importer.get_class('RegionBoundary')
region_boundary_object = region_boundary_class(region_data);

print('Trying OSM_FootprintsScraper')
osm_class = importer.get_class('OSM_FootprintScraper')
osm_data = {'length':'ft'}
instance1 = osm_class(osm_data)
instance1.get_footprints(region_boundary_object)

print('Trying USA_FootprintsScraper')
usa_class = importer.get_class('OSM_FootprintScraper')
usa_data = {'length':'ft'}
instance2 = usa_class(usa_data)
instance2.get_footprints(region_boundary_object)











