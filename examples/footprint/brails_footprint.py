# Copyright (c) 2024 The Regents of the University of California
#
# This file is part of BRAILS++.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Barbaros Cetiner
#
# Last updated:
# 11-06-2024

"""
Example demonstrating BRAILS' automated footprint downloading capabilities.

 Purpose: Testing 1) get_class method of Importer
                  2) get_footprints method of OSM_FootprintScraper,
                        MS_FootprintScraper, and USA_FootprintScraper modules
                  3) get_random_sample of AssetInventory
"""

from brails import Importer

# create an Importer object to fetch BRAILS++ classes:
importer = Importer()

# Specify the BoundaryRegion:

region_data = {'type': 'locationName', 'data': 'Tiburon, CA'}
region_boundary_class = importer.get_class('RegionBoundary')
region_boundary_object = region_boundary_class(region_data)

# Get Footprints using OSM:
print('Trying OSM_FootprintsScraper...')

osm_class = importer.get_class('OSM_FootprintScraper')
osm_data = {'length': 'ft'}
instance1 = osm_class(osm_data)
osm_inventory = instance1.get_footprints(region_boundary_object)

print('Number of assets in OSM', len(osm_inventory.inventory))


# Get Footprints using Microsoft Footprints Database:
print('Trying Microsoft Footprint Database...')
ms_class = importer.get_class('MS_FootprintScraper')
ms_data = {'length': 'ft'}
instance2 = ms_class(ms_data)
ms_inventory = instance2.get_footprints(region_boundary_object)

print('Number of assets in Microsoft Footprint Database',
      len(ms_inventory.inventory))


# Get Footprints using USA Structures data:
print('Trying USA_FootprintsScraper...')
usa_class = importer.get_class('USA_FootprintScraper')
usa_data = {'length': 'ft'}
instance3 = usa_class(usa_data)
usa_inventory = instance3.get_footprints(region_boundary_object)

print('Number of assets in USA Structures', len(usa_inventory.inventory))


# Obtain a smaller subset of random values, as we will not always want to get
# all the images:
print('\nGetting a subset of the USA footprints...')
small_inventory = usa_inventory.get_random_sample(10, 200)
print('Number of assets in the subset', len(small_inventory.inventory))
