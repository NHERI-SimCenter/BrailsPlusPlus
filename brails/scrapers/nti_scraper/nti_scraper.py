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
# 10-02-2025

"""
This module defines a class for retrieving tunnel data from NTI.

.. autosummary::

    NTIScraper
"""

from typing import Any, Dict, List
from shapely.geometry import Point, Polygon
from brails.constants import DEFAULT_UNITS
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import ArcgisAPIServiceHelper, UnitConverter

# Type of asset covered by the dataset:
ASSET_TYPE = 'tunnel'

# API endpoint to access National Tunnel Inventory data:
API_ENDPOINT = (
    'https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/'
    'NTAD_National_Tunnel_Inventory/FeatureServer/0/query'
)


# Attributes with associated units and the units they are defined in within
# the dataset:
DIMENSIONAL_ATTR = {'detour_length_a7': 'mi',
                    'tunnel_length_g1': 'ft',
                    'min_vert_clearance_over_tunnel_': 'ft',
                    'roadway_width_curb_to_curb_g3': 'ft',
                    'left_sidewalk_width_g4': 'ft',
                    'right_sidewalk_width_g5': 'ft',
                    'posting_load_gross_l5': 'ton_us',
                    'posting_load_axle_l6': 'ton_us',
                    'posting_load_type_3_l7': 'ton_us',
                    'posting_load_type_3S2_l8': 'ton_us',
                    'posting_load_type_33_l9': 'ton_us'}


class NTIScraper:
    """
    A class for scraping and processing National Tunnel Inventory (NTI) data.

    This class automates extracting tunnel data for a region from the NTI, 
    subdividing a region into smaller cells for efficient API calls, 
    filtering the results to include only the tunnels inside the specified
    region boundary, converting dimensional attributes into preferred 
    units, and organizing the data into an ``AssetInventory``.

    To import the :class:`NTIScraper`, use:

    .. code-block:: python

        from brails import Importer

        importer = Importer()
        nti_scraper_class = importer.get_class('NTIScraper')

    Parameters:
        units (Dict[str, str]):
            A dictionary mapping measurement types (e.g., length, weight)
            to their preferred output units. Parsed from the optional
            ``input_dict`` argument or defaulted to predefined units.
        inventory (AssetInventory):
            An inventory object that stores all processed tunnel assets for
            this scraper instance.
"""

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize an instance of the class with specified units.

        This constructor allows you to specify the length unit through the
        optional ``input_dict``. If ``input_dict`` is not provided, the length
        unit defaults to ``ft`` (feet). The inventory is also initialized when
        an instance is created.

        Args:
            input_dict (dict, optional):
                A dictionary that may contain keys 'length' and/or 'weight' 
                specifying the units to be in creating an ``AssetInventory`` 
                from NTI data. If provided, the values will be converted
                to lowercase and applied to the corresponding unit types. If 
                the dictionary is not provided or if a key is missing, default 
                units will be used: 'ft' (feet) for length and 'lb' (pounds) 
                for weight.
        """
        # Parse units from input_dict or fall back to default units:
        self.units = UnitConverter.parse_units(input_dict or {}, DEFAULT_UNITS)
        self.inventory = AssetInventory()

    def get_assets(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve tunnel inventory within a given region.

        This method processes the region boundary, splits it into smaller
        cells, fetches the necessary tunnel data for each cell, and returns the
        processed tunnel asset inventory.

        Args:
            region (RegionBoundary):
                A boundary object representing the geographic region for which
                to retrieve tunnel inventory data.

        Returns:
            AssetInventory:
                An inventory object containing the tunnel assets within the
                specified region.

        Raises:
            TypeError:
                If the ``region`` argument is not an instance of the
                ``RegionBoundary`` class. The error message will indicate that
                the `region` argument must be an instance of the
                ``RegionBoundary`` class.

        Example:
            >>> from brails.utils import Importer
            >>> importer = Importer()
            >>> region_data = {
            ...     'type': 'locationName',
            ...     'data': 'Santa Monica, CA'
            ... }
            >>> region_boundary = importer.get_class('RegionBoundary')
            >>> region = region_boundary(region_data)
            >>> nti_scraper_class = importer.get_class('NTIScraper')
            >>> nti_scraper = nti_scraper_class()
            No length unit specified. Using default: 'ft'.
            No weight unit specified. Using default: 'lb'.
            >>> inventory = nti_scraper.get_assets(region)
            Searching for Santa Monica, CA...
            Found Santa Monica, Los Angeles County, California, United States
            Meshing the defined area...
            Meshing complete. Covered Santa Monica with a single rectangular 
            cell.
            Obtaining the attributes of tunnels in each cell: 
            100%|██████████| 1/1 [00:00<00:00,  4.27it/s]
            Searching for Santa Monica, CA...
            Found Santa Monica, Los Angeles County, California, United States
            Found a total of 1 tunnel.
            >>> _ = inventory.write_to_geojson('nti_scraper_test.geojson')
            Wrote 1 asset to /home/bacetiner/nti_scraper_test.geojson
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  0 Asset:
            	    Coordinates:  [[-118.49455280024551, 34.011477779969816]]
            	    Features:  {'OBJECTID': 64, 'tunnel_number_i1': '53 0008', 
            'tunnel_name_i2': 'McClure Tunnel', 'state_code_i3': '06', 
            'year': 2025, 'federal_agency': 'N', 'county_code_i4': '037', 
            'place_code_i5': '70000', 'highway_district_i6': '07', 
            'route_number_i7': '00001', 'route_direction_i8': '0', 
            'route_type_i9': '3', 'facility_carried_i10': 'State Route 1', 
            'lrs_route_id_i11': 'SHS_001._P', 'lrs_mile_point_i12': 60.329, 
            'portal_latitude_i13': 34.01147778, 
            'portal_longitude_i14': -118.4945528, 'border_state_i15': '', 
            'border_financial_resp_i16': 0, 'border_tunnel_number_i17': '',
            'border_inspection_resp_i18': '', 'year_built_a1': 1935, 
            'year_rehabilitated_a2': 2021, 'total_number_of_lanes_a3': 4,
            'adt_a4': 58000, 'adtt_a5': 3150, 
            'year_of_average_daily_traffic_a': 2025, 
            'detour_length_a7': 5280.0, 'service_in_tunnel_a8': 1, 
            'owner_c1': 1, 'operator_c2': 1, 'direction_of_traffic_c3': 2, 
            'toll_c4': 0, 'nhs_designation_c5': 1, 
            'strahnet_designation_c6': 0, 'functional_classification_c7': 3, 
            'urban_code_c8': 51445, 'tunnel_length_g1': 478.0, 
            'min_vert_clearance_over_tunnel_': 14.6, 
            'roadway_width_curb_to_curb_g3': 23.5, 
            'left_sidewalk_width_g4': 3.5, 'right_sidewalk_width_g5': 0.0, 
            'routine_inspection_target_date_': 1502150400000, 
            'actual_routine_inspection_date_': 1696377600000, 
            'routine_inspection_interval_d3': 24, 'in_depth_inspection_d4': 0,
            'damage_inspection_d5': 0, 'special_inspection_d6': 0, 
            'load_rating_method_l1': '0', 
            'inventory_load_rating_factor_l2': 0.52, 
            'operating_load_rating_factor_l3': 0.87, 
            'tunnel_load_posting_status_l4': 'A', 
            'posting_load_gross_l5': 0.0, 'posting_load_axle_l6': 0.0, 
            'posting_load_type_3_l7': 0.0, 'posting_load_type_3S2_l8': 0.0, 
            'posting_load_type_33_l9': 0.0, 'height_restriction_l10': 1, 
            'hazardous_material_restriction_': 0, 'other_restrictions_l12': 0,
            'under_navigable_waterway_n1': 0, 
            'navigable_waterway_clearance_n2': 0, 
            'tunnel_or_portal_island_protect': 0, 'number_of_bores_s1': 1, 
            'tunnel_shape_s2': 1, 'portal_shapes_s3': 1, 
            'ground_conditions_s4': 1, 'complex_s5': 0, 'submitted_by': '06',
            'type': 'tunnel'}
        """
        if not isinstance(region, RegionBoundary):
            raise TypeError("The 'region' argument must be an instance of the "
                            "'RegionBoundary' class.")

        # Download tunnel data for each cell:
        api_tools = ArcgisAPIServiceHelper(API_ENDPOINT)
        results, final_cells = api_tools.download_all_attr_for_region(
            region,
            task_description='Obtaining the attributes of tunnels in each cell'
        )

        # Process the downloaded data and save it in an AssetInventory:
        self._process_data(region, final_cells, results)
        return self.inventory

    def _process_data(
        self,
        region: RegionBoundary,
        final_cells: List[Polygon],
        results: Dict[Polygon, List[Dict[str, Any]]]
    ) -> None:
        """
        Process the downloaded NTI data and store it in an AssetInventory.

        This method filters tunnel data retrieved from each cell based on
        whether the geometry lies within the region boundary, converts units if
        needed, and constructs asset objects for the inventory.

        Args:
            region (RegionBoundary):
                The region object containing the boundary polygon used to
                filter relevant tunnel data.
            final_cells (List[Polygon]):
                List of polygonal cells that subdivide the region and contain
                tunnel data.
            results (Dict[Polygon, List[Dict[str, Any]]]):
                A mapping from each cell polygon to a list of dictionaries,
                each representing a tunnel and its associated attributes.
        """
        # Obtain the boundary polygon for the region:
        boundary_polygon, _, _ = region.get_boundary(print_progress=False)

        # Identify the cells that are inside the bounding polygon and record
        # their data:
        data, data_to_filter = [], []
        for cell in final_cells:
            if boundary_polygon.contains(cell):
                data.extend(results.get(cell, []))
            else:
                data_to_filter.extend(results.get(cell, []))

        # Filter the data within the cells that are not contained in the
        # bounding polygon such that only the points within the bounding
        # polygon are retained:
        for item in data_to_filter:
            if boundary_polygon.contains(Point(item['geometry']['x'],
                                               item['geometry']['y'])):
                data.append(item)

        # Display the number of elements detected:
        element_number = len(data)
        element = 'tunnel' if element_number==1 else 'tunnels'
        print(f'\nFound a total of {element_number} {element}.')

        # Save the results in the inventory:
        for index, item in enumerate(data):
            geometry = [[item['geometry']['x'], item['geometry']['y']]]
            asset_features = {**item['attributes'], 'type': ASSET_TYPE}
            for feature, feature_unit in DIMENSIONAL_ATTR.items():
                feature_Value = asset_features.get(feature)
                if feature_Value is not None:
                    unit_type = UnitConverter.get_unit_type(feature_unit)
                    target_unit = self.units[unit_type]
                    asset_features[feature] = UnitConverter.convert_unit(
                        feature_Value,
                        feature_unit,
                        target_unit
                    )

            # Create the Asset and add it to the inventory:
            asset = Asset(index, geometry, asset_features)
            self.inventory.add_asset(index, asset)
