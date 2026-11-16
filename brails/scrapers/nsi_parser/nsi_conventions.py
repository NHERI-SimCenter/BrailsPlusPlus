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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.

"""
Declare the raw NSI fields and map NSI records to SimCenter conventions.

.. autosummary::

      to_simcenter
"""

from __future__ import annotations

import pyarrow as pa

# The feature type written on every asset built from an NSI record.
ASSET_TYPE = 'building'

# The fields the NSI 2026 API serves and the type each is stored under in a
# raw NSI parquet table. A float field may be served as an integer and is
# stored as a float.
NSI_RAW_SCHEMA = pa.schema(
    [
        ('bid', pa.string()),
        ('bldgtype', pa.string()),
        ('bldheight', pa.float64()),
        ('cbfips', pa.string()),
        ('creprcnt', pa.float64()),
        ('crerank', pa.float64()),
        ('depindex', pa.float64()),
        ('fd_id', pa.int64()),
        ('firmzone', pa.string()),
        ('found_ht', pa.float64()),
        ('found_type', pa.string()),
        ('ftprntid', pa.string()),
        ('ftprntsqft', pa.int64()),
        ('ftprntsrc', pa.string()),
        ('fullrep', pa.float64()),
        ('grnd_elv_m', pa.float64()),
        ('ground_elv', pa.float64()),
        ('med_yr_blt', pa.int64()),
        ('novehprob', pa.float64()),
        ('num_story', pa.int64()),
        ('o65disable', pa.float64()),
        ('occtype', pa.string()),
        ('pctlowclr', pa.float64()),
        ('pop2amo65', pa.int64()),
        ('pop2amu65', pa.int64()),
        ('pop2pmo65', pa.int64()),
        ('pop2pmu65', pa.int64()),
        ('resunits', pa.int64()),
        ('source', pa.string()),
        ('sqft', pa.float64()),
        ('st_damcat', pa.string()),
        ('static_bfe', pa.float64()),
        ('students', pa.int64()),
        ('u65disable', pa.float64()),
        ('usastrucid', pa.string()),
        ('val_cont', pa.float64()),
        ('val_struct', pa.float64()),
        ('val_vehic', pa.float64()),
        ('vehperunit', pa.float64()),
        ('x', pa.float64()),
        ('y', pa.float64()),
        ('zone_sub', pa.string()),
    ]
)

# NSI field names and the SimCenter feature names assigned to them. Fields
# that are not listed pass through with their NSI names.
# The fields and the features they become are described in README.rst.
#
# med_yr_blt becomes YearBuiltTractMedian: NSI reports it as the median year
# built of the structures in the Census tract, so every building in a tract
# carries the same value and none of them was observed. Use the
# YearBuiltSampler to draw a per-building year from it.
#
# sqft becomes GrossArea: NSI reports it as the structure's estimated
# square footage, its gross area, not its plan area.
#
# fullrep becomes ReplacementCost: NSI reports it as the full, not
# depreciated, estimated replacement value of the structure.
NSI_TO_SIMCENTER = {
    'x': 'Longitude',
    'y': 'Latitude',
    'sqft': 'GrossArea',
    'num_story': 'NumberOfStories',
    'med_yr_blt': 'YearBuiltTractMedian',
    'fullrep': 'ReplacementCost',
    'bldgtype': 'BuildingType',
    'occtype': 'OccupancyClass',
    'found_type': 'FoundationType',
    'found_ht': 'FoundationHeight',
    'ground_elv': 'GroundElevation',
    'firmzone': 'FloodZone',
}

# NSI general building type codes kept as general classes. The H code is
# mapped to MH; the other codes (W, S, C, M) match the Hazus general classes
# as they are.
BUILDING_TYPE_MAP = {'H': 'MH'}


def _unpack_occtype(occtype: str) -> dict:
    """
    Decode the packed NSI occupancy string into separate features.

    NSI uses the extended RES1 occupancy code that includes story count,
    split-level and basement information, for example ``RES1-2SWB`` or
    ``RES1-SLNB``. The occupancy class is the part before the dash, which keeps
    the RES3 subtypes (``RES3A``-``RES3F``) verbatim since they carry no dash.

    Args:
        occtype (str):
            The NSI occupancy code.

    Returns:
        dict:
            ``OccupancyClass``, plus boolean ``SplitLevel`` and
            ``Basement`` where the code carries them. A feature the code
            does not state is omitted, not defaulted, so downstream
            imputation sees a genuine gap.
    """
    unpacked = {'OccupancyClass': occtype.split('-')[0]}

    if occtype.startswith('RES1') and '-' in occtype:
        suffix = occtype.partition('-')[2]
        unpacked['SplitLevel'] = suffix.startswith('SL')
        if suffix.endswith('WB'):
            unpacked['Basement'] = True
        elif suffix.endswith('NB'):
            unpacked['Basement'] = False

    return unpacked


def to_simcenter(properties: dict) -> dict:
    """
    Rename raw NSI fields to SimCenter names and decode packed values.

    The conversion renames the fields listed in :data:`NSI_TO_SIMCENTER`,
    maps the general building type codes through :data:`BUILDING_TYPE_MAP`,
    and unpacks the occupancy code. Fields it does not know pass through
    with their NSI names.

    Args:
        properties (dict):
            The raw NSI properties of one point.

    Returns:
        dict:
            A new dictionary with the properties under SimCenter
            conventions. The input is not modified.
    """
    converted = {
        NSI_TO_SIMCENTER.get(key, key): value for key, value in properties.items()
    }

    building_type = converted.get('BuildingType')
    if isinstance(building_type, str):
        converted['BuildingType'] = BUILDING_TYPE_MAP.get(
            building_type, building_type
        )

    occtype = converted.get('OccupancyClass')
    if isinstance(occtype, str):
        converted.update(_unpack_occtype(occtype))

    return converted
