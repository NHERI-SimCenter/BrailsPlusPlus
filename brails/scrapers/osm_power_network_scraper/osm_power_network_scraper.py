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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 12-05-2024

"""
This module defines the class scraping power network data from OSM.

.. autosummary::

    OSM_PowerNetworkScraper
"""

import requests
from shapely.geometry import Polygon

from brails.scrapers.footprint_scraper import FootprintScraper
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import Asset, AssetInventory

OVERPASS_URL = "http://overpass-api.de/api/interpreter"


class OSM_PowerNetworkScraper(FootprintScraper):
    """
    A class for retrieving & processing power network data from OpenStreetMap.

    This class provides methods to query and extract power network elements,
    including power lines, substations, and transformers, from the
    OpenStreetMap API. It supports querying data for both bounding polygons
    and geographic names of regions.

    Attributes:
        length_unit (str):
            The unit of length for measurements. Default is 'ft'.

    Methods:
        get_elements(region: RegionBoundary) -> AssetInventory:
            Retrieves the power network elements and associated attributes for
            a given region using the Overpass API.
    """

    def __init__(self, input_dict: dict):
        """
        Initialize the class object with length units.

        Args:
            input_dict (dict):
                A dictionary specifying length units. If not provided, 'ft' is
                assumed by default..
        """
        self.length_unit = input_dict.get('length', 'ft')

    def get_elements(self, region: RegionBoundary) -> AssetInventory:
        """
        Get the OSM geometries and atrributes for the power network in an area.

        Args:
            region (RegionBoundary):
                The region of interest.

        Returns:
            AssetInventory:
                An inventory of power network elements in the region.

        """
        bpoly, queryarea_printname, osmid = region.get_boundary()

        # If the bounding polygon was obtained by calling a region name:
        if osmid is not None:

            queryarea_turboid = osmid + 3600000000
            query = f"""
            [out:json][timeout:5000][maxsize:2000000000];
            nwr["power"](area:{queryarea_turboid});
            out body;
            >;
            out skel qt;
            """

        else:
            # If the bounding polygon is rectangular:
            if self._is_box(bpoly):
                # Convert the bounding polygon coordinates to latitude and
                # longitude fashion:
                bbox_coords = bpoly.bounds
                bbox = f'{bbox_coords[1]},{bbox_coords[0]},{bbox_coords[3]},'\
                    f'{bbox_coords[2]}'
            else:
                bbox_coords = list(bpoly.exterior.coords)
                bbox = 'poly:"'
                for (lon, lat) in bbox_coords[:-1]:
                    bbox += f'{lat} {lon} '
                bbox = bbox[:-1] + '"'

            query = f"""
            [out:json][timeout:5000][maxsize:2000000000];
            nwr["power"]({bbox});
            out body;
            >;
            out skel qt;
            """

        datalist = self._fetch_data_from_api(query, queryarea_printname)
        assets_data = self._process_power_data(datalist)

        return self._create_inventory(assets_data)

    def _fetch_data_from_api(self,
                             query: str,
                             queryarea_printname: str) -> list[dict]:
        """
        Fetch power network data from OSM Overpass API.

        Args:
            query (str):
                The Overpass API query string.
            queryarea_printname (str):
                A human-readable name for the query area, used for error
                reporting.

        Returns:
             list:
                 A list of power network elements obtained from the API
                 response.
        """
        try:
            response = requests.get(OVERPASS_URL, params={'data': query})
            response.raise_for_status()
            return response.json().get('elements', [])
        except requests.exceptions.RequestException as e:
            print(f'Error fetching data for {queryarea_printname}: {e}')
            return []

    def _process_power_data(self,
                            datalist: list[dict]
                            ) -> dict[str, dict[str, list | dict]]:
        """
        Parse properties of power network elements obtained from Overpass API.

        Args:
            datalist (list):
                A list of data elements retrieved from the API.

        Returns:
            dict:
                A dictionary organized by asset type, containing asset
                geometries and attributes.
        """
        # Create dictinary of node coordinates for assembling way geometries
        # and create a dictionary of assets organized by asset type:
        nodedict = {}
        assets_data = {}
        for data in datalist:
            if 'tags' in data:
                asset_type = 'power_' + data['tags']['power']
                if asset_type not in assets_data:
                    assets_data[asset_type] = {'keys': set()}
                else:
                    keys = list(data['tags'].keys())
                    assets_data[asset_type]['keys'].update(keys)
            else:
                if data['type'] == 'node':
                    nodedict[data["id"]] = [data["lon"], data["lat"]]

        # Preallocate the dictionary of attributes and list of coordinates for
        # each asset type and remove the extracted asset-specific keys:
        for asset_type in assets_data.keys():
            keys = list(assets_data[asset_type]['keys'])
            assets_data[asset_type]['attributes'] = {key: [] for key in keys
                                                     if key != 'power'}
            assets_data[asset_type]['geometries'] = []
            del assets_data[asset_type]['keys']

        # Assemble asset geometries and dictionary of attributes and save them
        # in assets_data:
        for data in datalist:
            if data["type"] == "way":
                nodes = data["nodes"]
                geometry = []
                for node in nodes:
                    geometry.append(nodedict[node])
                asset_type = 'power_' + data['tags']['power']
                assets_data[asset_type]['geometries'].append(geometry)
                for key in assets_data[asset_type]['attributes'].keys():
                    assets_data[asset_type]['attributes'][key].append(
                        data['tags'].get(key, None))

            if data['type'] == 'node' and 'tags' in data:
                asset_type = 'power_' + data['tags']['power']
                geometry = [[data["lon"], data["lat"]]]
                assets_data[asset_type]['geometries'].append(
                    geometry)
                for key in assets_data[asset_type]['attributes'].keys():
                    assets_data[asset_type]['attributes'][key].append(
                        data['tags'].get(key, None))
        return assets_data

    def _create_inventory(self,
                          assets_data: dict[str: dict[str, list | dict]]
                          ) -> AssetInventory:
        """
        Create an AssetInventory from the processed OSM power network data.

        Parameters:
            assets_data (dict):
                A dictionary containing structured power network asset data.

        Returns:
            AssetInventory:
                An instance of AssetInventory populated with extracted power
                network elements.
        """
        # Save the results in an AssetInventory:
        inventory = AssetInventory()
        counter = 0

        for asset_type in assets_data.keys():
            for ind, geometry in enumerate(
                    assets_data[asset_type]['geometries']):
                asset_features = {'type': asset_type}

                for key in assets_data[asset_type]['attributes'].keys():
                    attr = assets_data[asset_type]['attributes'][key][ind]
                    asset_features[key] = "NA" if attr is None else attr

                # Create the Asset and add it to the inventory:
                asset = Asset(counter, geometry, asset_features)
                inventory.add_asset(counter, asset)
                counter += 1
        return inventory

    def _is_box(self, geom: Polygon) -> bool:
        """
        Determine whether a given Shapely geometry is a rectangular box.

        A box is defined as a Polygon with exactly four corners and opposite
        sides being equal. This function checks if the geometry is a Polygon
        with 5 coordinates (the 5th being a duplicate of the first to close the
        polygon), and verifies that opposite sides are equal, ensuring that the
        polygon is rectangular.

        Args:
            geom (Polygon):
                A Shapely Polygon object to be checked.

        Returns:
            bool:
                True if the Polygon is a rectangular box, False otherwise.
        """
        # Check if the geometry is a Polygon and has exactly 4 corners:
        if isinstance(geom, Polygon) and len(geom.exterior.coords) == 5:
            # Check if opposite sides are equal (box property):
            x1, y1 = geom.exterior.coords[0]
            x2, y2 = geom.exterior.coords[1]
            x3, y3 = geom.exterior.coords[2]
            x4, y4 = geom.exterior.coords[3]

            return (x1 == x2 and y1 == y4 and x3 == x4 and y2 == y3)
        return False
