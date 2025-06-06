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
# 06-06-2025

"""
This module defines the class object for downloading ASCE Hazard data.

.. autosummary::

    ASCE_HAZARD_DATA_SCRAPER
"""

from typing import Dict, Any, TYPE_CHECKING
from shapely.geometry import box
from brails.utils.api import ArcgisAPIServiceHelper
from brails.utils.inventory_validator import InventoryValidator

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


API_ENDPOINT = ('https://gis.asce.org/arcgis/rest/services/ASCE722/'
                'w2022_Tile_RC_I/MapServer/2/query')


class ASCE_HAZARD_DATA_SCRAPER:
    """
    A class to retrieve wind speed hazard data.

    Attributes:
        length_unit (str):
            Unit of length for building attributes (default is 'ft').

    Methods:
        get_windspeeds(inventory):
            Retrieves wind speed data and attaches it to the asset inventory.
    """

    def __init__(self, input_dict: Dict[str, Any]):
        """
        Initialize the ASCE_HAZARD_DATA_SCRAPER instance.

        Args:
            input_dict (dict):
                A dictionary specifying configuration parameters. If the key
                'length' is present, it defines the unit of length (e.g., 'ft',
                'm'). Defaults to 'ft'.
        """
        self.length_unit = input_dict.get('length', 'ft')

    def get_windspeeds(self, inventory: "AssetInventory") -> "AssetInventory":
        """
        Retrieve wind speed data for each asset in the inventory and attach it.

        This method downloads wind speed data using the ArcGIS API for a
        bounding polygon (currently hardcoded) and is intended to associate
        hazard attributes with assets.

        Args:
            inventory (AssetInventory):
                The asset inventory containing building locations.

        Returns:
            AssetInventory:
                Updated inventory with appended wind speed attributes.

        Raises:
            TypeError:
                If the 'inventory' argument is not an instance of
                AssetInventory.
        """
        if not InventoryValidator.is_inventory(inventory):
            raise TypeError(
                "The 'inventory' argument must be an instance of "
                "'AssetInventory'.")

        # Define a hardcoded bounding polygon
        bpoly = box(-118.5534678, 33.9666583, 34.0505825, -118.4435336)

        # Get the number of elements allowed per cell by the API
        api_tools = ArcgisAPIServiceHelper(API_ENDPOINT)

        datalist = api_tools.download_attr_from_api(bpoly, 'all')

        # TO-DO: Implement data post-processing to match to inventory

        return datalist
