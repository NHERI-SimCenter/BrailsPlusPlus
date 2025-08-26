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
# 08-26-2025

"""
Ruleset that maps USGS NLCD classes to a coarser land cover categories.

.. autosummary::

    NLCDToCoarseLandCover
"""


from brails.types.asset_inventory import AssetInventory
from brails.inferers.inference_engine import InferenceEngine


# NLCD class name → coarse category
LAND_COVER_MAPPING = {
    'Open Water': 'Open',
    'Perennial Ice/Snow': 'Open',
    'Developed, Open Space': 'Open',
    'Developed, Low Intensity': 'Light Suburban',
    'Developed, Medium Intensity': 'Suburban',
    'Developed, High Intensity': 'Suburban',
    'Barren Land (Rock/Sand/Clay)': 'Open',
    'Unconsolidated Shore': 'Open',
    'Deciduous Forest': 'Trees',
    'Evergreen Forest': 'Trees',
    'Mixed Forest': 'Trees',
    'Dwarf Scrub': 'Light Trees',
    'Shrub/Scrub': 'Light Trees',
    'Grasslands/Herbaceous': 'Open',
    'Sedge/Herbaceous': 'Open',
    'Lichens': 'Open',
    'Moss': 'Open',
    'Pasture/Hay': 'Open',
    'Cultivated Crops': 'Open',
    'Woody Wetlands': 'Trees',
    'Emergent Herbaceous Wetlands': 'Open'
}

INPUT_KEY = 'land_cover'
MISSING_VALUE_PLACEHOLDER = 'Unknown'
OUTPUT_KEY = 'LandCover'

KEYS_UTILIZED = [INPUT_KEY, OUTPUT_KEY]


class NLCDToCoarseLandCover(InferenceEngine):
    """
    Convert detailed USGS NLCD classes to a set of coarse categories.

    This class maps USGS National Land Cover Dataset (NLCD) classes to one of
    five coarse categories: ``'Open'``, ``'Light Suburban'``, ``'Suburban'``,
    ``'Light Trees'``, or ``'Trees'``.

    It operates exclusively on the ``'land_cover'`` field of each asset in an
    inventory and adds a new field ``'LandCover'`` containing the mapped coarse
    category. Missing or unmapped values are assigned a placeholder
    (default: ``'Unknown'``).

    To use :class:`NLCDToCoarseLandCover`, include the following lines in your
    code:

    .. code-block:: python

        from brails import Importer

        importer = Importer()
        nlcd_inferer = importer.get_class(NLCDToCoarseLandCover)

    """

    def __init__(self):
        """Initialize the converter ruleset."""

    def infer(self, input_inventory: AssetInventory) -> AssetInventory:
        """
        Update each asset in the inventory with a coarse land cover category.

        The method reads the ``'land_cover'`` field of each asset, maps it to a
        coarse category using the provided mapping, and writes the result to
        the ``'coarse_land_cover'`` field. If the original value is missing or
        unmapped, a placeholder is used.

        Args:
            asset_inventory (AssetInventory):
                An inventory object containing assets with a ``'land_cover'``
                field.
        """
        for asset in input_inventory.inventory.values():
            fine_val = asset.features.get(INPUT_KEY)
            if fine_val is None:
                coarse_val = MISSING_VALUE_PLACEHOLDER
            else:
                coarse_val = self.mapping.get(
                    fine_val,
                    MISSING_VALUE_PLACEHOLDER
                )
            asset.features[OUTPUT_KEY] = coarse_val
