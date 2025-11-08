# Copyright (c) 2025 The Regents of the University of California
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
# 08-28-2025

"""
Class for scraping FEMA FIRM information for individual assets in an inventory.

.. autosummary::

    FEMAFIRMScraper
"""

import copy
from typing import Any, Dict, List

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from brails import Importer
from brails.constants import DEFAULT_UNITS
from brails.types.asset_inventory import AssetInventory
from brails.utils import ArcgisAPIServiceHelper, UnitConverter
from brails.scrapers.asset_data_augmenter import AssetDataAugmenter

# API endpoint to access the National Flood Hazard Layer Flood Insurance Rate
# Map data:
API_ENDPOINT = (
    'https://hazards.fema.gov/arcgis/rest/services/FIRMette/NFHLREST_FIRMette/'
    'MapServer/20/query'
)


# List of core flood data fields:
CORE_FIELDS = ['FLD_ZONE', 'ZONE_SUBTY', 'STATIC_BFE']

# Attributes with associated units and the units they are defined in within
# the dataset:
DIMENSIONAL_ATTR = {'STATIC_BFE': 'ft'}
NULL_VALUE = -9999


class FEMAFIRMScraper(AssetDataAugmenter):
    """
    A class for scraping FEMA Flood Insurance Rate Map (FIRM) data.

    The scraper retrieves FEMA FIRM layers from the configured API endpoint,
    processes the flood hazard information, and assigns relevant attributes
    (e.g., flood zone classification, subtype, base flood elevation) to
    each asset in an :class:`AssetInventory`.

    To use :class:`FEMAFIRMScraper`, include the following lines in your
    code:

    .. code-block:: python

        from brails import Importer

        importer = Importer()
        firm_scraper = importer.get_class('FEMAFIRMScraper')

    Parameters:
        units (str):
            The length unit to use (default is 'ft').
        requested_attributes (List[str]):
            List of attributes selected for output.
        inventory (AssetInventory):
            An inventory object holding the assets for this instance.
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize an instance with length unit and attribute selection.

        Args:
            input_dict (dict, optional):
                Supported keys:
                    - 'length' (str): Length unit to use. Defaults to 'ft'.
                    - 'attribute_mode' (str): Attribute output mode.
                       Options:
                           - ``'all'``: include all API attributes
                           - ``'core_flood_fields'``: include only the
                             essential flood zone classification fields,
                             namely: flood zone, flood zone subtype, and static
                             base flood_elevation
                           - 'custom': include a specific subset
                        Defaults to 'core_flood_fields'.
                    - 'attributes' (list[str]): List of attributes to include
                      if ``attribute_mode == 'custom'``.
        """
        input_dict = input_dict or {}

        # Parse units:
        self.units = UnitConverter.parse_units(input_dict, DEFAULT_UNITS)

        # Determine requested attributes:
        self.requested_attributes = self._select_attributes(input_dict)

        # Initialize empty inventory:
        self.inventory = AssetInventory()

    def _select_attributes(self, input_dict: Dict[str, Any]):
        """Determine which attributes to request based on input_dict."""
        attribute_mode = input_dict.get(
            'attribute_mode',
            'core_flood_fields'
        ).lower()
        available_attributes = FEMAFIRMScraper.get_available_attributes()

        if attribute_mode == 'all':
            return available_attributes.copy()

        elif attribute_mode == 'core_flood_fields':
            return CORE_FIELDS.copy()

        elif attribute_mode == 'custom':
            requested = input_dict.get('attributes', [])
            valid_attrs = [a for a in requested if a in available_attributes]
            omitted = set(requested) - set(valid_attrs)
            if omitted:
                print(
                    'Warning: The following requested attributes were not '
                    f"found and will be omitted: {', '.join(omitted)}"
                )
            if not valid_attrs:
                print(
                    'Warning: No valid attributes found in the list of '
                    'requested attributes. Getting the following attributes'
                    f'{CORE_FIELDS} instead.'
                )
                return CORE_FIELDS.copy()
            return valid_attrs

        else:
            raise ValueError(
                f"Invalid attribute_mode '{attribute_mode}'. "
                "Must be one of: 'all', 'core_flood_fields', or 'custom'."
            )

    @staticmethod
    def get_available_attributes() -> List[str]:
        """
        Retrieve the list of available attribute names from the API layer.

        This method fetches all field names currently defined in the dataset
        at the configured API endpoint.

        Returns:
            List[str]: A list of attribute names provided by the API.


        Example:
            >>> from brails import Importer
            >>> importer = Importer()
            >>>
            >>> firm_Scraper = importer.get_class('FEMAFIRMScraper')
            >>> attributes = firm_Scraper.get_available_attributes()
            >>> print(attributes[:5])
            ['OBJECTID', 'DFIRM_ID', 'FLD_AR_ID', 'STUDY_TYP', 'FLD_ZONE']
        """
        return ArcgisAPIServiceHelper.fetch_api_fields(API_ENDPOINT)

    def populate_feature(
        self,
        input_inventory: AssetInventory
    ) -> AssetInventory:
        """
        Populate each asset in inventory with relevant FEMA FIRM attributes.

        This method:
            - Computes the bounding box of the assets in the inventory.
            - Downloads the FIRM data for the bounding box.
            - Processes the data and assigns requested attributes to each
              asset.

        Args:
            input_inventory (AssetInventory):
                An inventory of assets to augment with FEMA FIRM data.

        Returns:
            AssetInventory:
                The updated AssetInventory with FEMA FIRM attributes added.

        Raises:
            TypeError:
                If `input_inventory` is not an instance of AssetInventory.

        Example:
            >>> from brails.types.asset_inventory import Asset, AssetInventory
            >>> from brails import Importer
            >>>
            >>> asset1 = Asset(1, [[-91.5302, 41.6611]])
            >>> asset2 = Asset(2, [[-91.5340, 41.6625]])
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(1, asset1)
            >>> _ = inventory.add_asset(2, asset2)
            >>>
            >>> importer = Importer()
            >>> firm_scraper = importer.get_class('FEMAFIRMScraper')()
            >>> updated_assets = firm_scraper.populate_feature(inventory)
            No length unit specified. Using default: 'ft'.
            No weight unit specified. Using default: 'lb'.
            Meshing the defined area...
            Meshing complete. Covered the bounding box: (-91.53420000000001,
            41.660999999999994, -91.52999999999999, 41.662600000000005) with a
            single rectangular cell.
            Obtaining the FIRM information for the bounding box with
            coordinates (-91.53420000000001, 41.660999999999994,
            -91.52999999999999, 41.662600000000005): 100%|██████████| 1/1
            [00:00<00:00,  1.12it/s]AssetInventory
            >>> updated_assets.print_info()
            Inventory stored in:  dict
            Key:  1 Asset:
                 Coordinates:  [[-91.5302, 41.6611]]
                 Features:  {'FLD_ZONE': 'X', 'ZONE_SUBTY':
            'AREA OF MINIMAL FLOOD HAZARD', 'STATIC_BFE': -9999.0}
            Key:  2 Asset:
                 Coordinates:  [[-91.534, 41.6625]]
                 Features:  {'FLD_ZONE': 'X', 'ZONE_SUBTY':
            'AREA OF MINIMAL FLOOD HAZARD', 'STATIC_BFE': -9999.0}
        """
        bbox = input_inventory.get_extent().bounds

        importer = Importer()
        region_data = {"type": "locationPolygon", "data": bbox}
        region_boundary_class = importer.get_class("RegionBoundary")
        region = region_boundary_class(region_data)

        # Download FIRM data for the extracted bounding box:
        api_tools = ArcgisAPIServiceHelper(API_ENDPOINT)
        results, _ = api_tools.download_all_attr_for_region(
            region,
            task_description=(
                'Obtaining the FIRM information for the bounding box with '
                f'coordinates {bbox}'
            )
        )

        # Process the downloaded data and save it in an AssetInventory:
        self.inventory = copy.deepcopy(input_inventory)
        self._process_data(results)

        return self.inventory

    def _process_data(
        self,
        results: Dict[Polygon, List[Dict[str, Any]]]
    ) -> None:
        """
        Process the downloaded FEMA FIRM data and update the AssetInventory.

        This method:
            - Filters the FIRM attributes to only those requested.
            - Maps each asset's centroid to the corresponding polygon.
            - Updates asset features in place with the matched FIRM attributes.

        Args:
            results (Dict[Polygon, List[Dict[str, Any]]]):
                A mapping from each cell polygon to a list of dictionaries,
                each representing a FIRM feature.

        Notes:
            The method modifies ``self.inventory`` in place.
        """
        firm_data = list(results.values())[0]
        requested_fields_set = set(self.requested_attributes)

        # Filter attributes and preserve geometry:
        firm_data_filtered = [
            {
                'geometry': feature['geometry'],  # keep the geometry as-is
                'attributes': {k: v for k, v in feature['attributes'].items()
                               if k in requested_fields_set}
            }
            for feature in firm_data
        ]

        # Pre-build polygons and STRtree:
        polygons = [Polygon(f['geometry']['rings'][0])
                    for f in firm_data_filtered]
        tree = STRtree(polygons)

        # Precompute asset points:
        assets = list(self.inventory.inventory.values())
        points = [Point(asset.get_centroid()) for asset in assets]

        # Map points to polygons:
        point_to_polygon_map = {}
        for pt_index, pt in enumerate(points):
            candidate_indices = tree.query(pt)   # only nearby polygons
            for polygon_index in candidate_indices:
                if polygons[polygon_index].contains(pt):
                    point_to_polygon_map[pt_index] = polygon_index
                    break

        # Update asset features and convert units:
        for asset_index, asset in enumerate(assets):
            # Merge FIRM attributes if a matching polygon exists:
            polygon_index = point_to_polygon_map.get(asset_index)
            if polygon_index is not None:
                asset.features.update(
                    firm_data_filtered[polygon_index]['attributes']
                )
            # TODO: Remove null values.
            # Convert dimensional attributes to target units:
            for feature, original_unit in DIMENSIONAL_ATTR.items():
                value = asset.features.get(feature)
                if value is not None and value != NULL_VALUE:
                    unit_type = UnitConverter.get_unit_type(original_unit)
                    target_unit = self.units[unit_type]
                    asset.features[feature] = UnitConverter.convert_unit(
                        value,
                        original_unit,
                        target_unit
                    )
