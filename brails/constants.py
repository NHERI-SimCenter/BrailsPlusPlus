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
# 08-12-2025

"""This defines the global constants used in BRAILS++."""

from typing import Dict, Tuple

DEFAULT_UNITS: Dict[str, str] = {
    'length': 'ft',
    'weight': 'lb',
}

# Earth's radius: R_EARTH_KM, in kilometers, is the rounded value GeoTools
# uses; EARTH_RADIUS_M is the mean radius in meters that the asset ids
# (through a copy in asset_id.py) and the distances of the donor search and
# the unit budget use.
R_EARTH_KM = 6371
EARTH_RADIUS_M = 6371008.8

# The provenance feature the footprint scrapers write, naming the provider
# of each footprint.
FOOTPRINT_PROVIDER_KEY = 'footprint_provider'

# The features holding each footprint provider's own ids, by the value the
# footprint scrapers record as FOOTPRINT_PROVIDER_KEY. The first of them
# ranks footprints of equal geometry and identifies a dropped footprint.
PROVIDER_ID_FEATURES: Dict[str, Tuple[str, ...]] = {
    'usa_structures': ('usa_build_id', 'usa_uuid'),
    'microsoft': (),
    'osm': ('osm_id',),
    'overture': ('overture_id',),
}

# The features that record a footprint's source: every provider id and the
# provenance feature.
PROVIDER_FEATURES: Tuple[str, ...] = (
    *(
        feature
        for id_features in PROVIDER_ID_FEATURES.values()
        for feature in id_features
    ),
    FOOTPRINT_PROVIDER_KEY,
)

# A footprint's release and the datasets of its sources. A footprint of any
# provider may carry them; only HazusEQInventoryCompiler writes them, for an
# Overture footprint.
FOOTPRINT_RELEASE_KEY = 'footprint_release'
FOOTPRINT_DATASETS_KEY = 'footprint_datasets'

# The features that state where a footprint came from. Unlike
# PROVIDER_FEATURES, they hold no provider id.
FOOTPRINT_PROVENANCE_FEATURES: Tuple[str, ...] = (
    FOOTPRINT_PROVIDER_KEY,
    FOOTPRINT_RELEASE_KEY,
    FOOTPRINT_DATASETS_KEY,
)

# The features the census tract scraper writes, holding an asset's Census
# tract and block group.
TRACT_KEY = 'TRACT_GEOID'
BLOCK_GROUP_KEY = 'BLOCKGROUP_GEOID'

# Characters in a block-group GEOID: state 2, county 3, tract 6, group 1. A
# tract's GEOID is the first TRACT_GEOID_LENGTH characters of its block
# groups' GEOIDs.
GEOID_LENGTH = 12
TRACT_GEOID_LENGTH = 11

# What replaces a secret, such as an API key, wherever text that would hold it is
# shown: a repr, a log record or a message.
REDACTED = '<redacted>'

# The geographic CRS asset coordinates are stored in.
GEOGRAPHIC_CRS = 'EPSG:4326'

# SimCenter feature names shared by several modules
OCCUPANCY_KEY = 'OccupancyClass'
UNITS_KEY = 'NumberOfUnits'
STORIES_KEY = 'NumberOfStories'
PLAN_AREA_KEY = 'PlanArea'
GROSS_AREA_KEY = 'GrossArea'
YEAR_BUILT_KEY = 'YearBuilt'
BUILDING_TYPE_KEY = 'BuildingType'
REPLACEMENT_COST_KEY = 'ReplacementCost'

# SimCenter feature names of a Hazus earthquake inventory
DESIGN_LEVEL_KEY = 'DesignLevel'
STRUCTURE_TYPE_KEY = 'StructureType'
HEIGHT_CLASS_KEY = 'HeightClass'

# The feature HazusEQInventoryCompiler writes on every building: true in a
# world the building is excluded from, false in the others.
EXCLUDED_KEY = 'Excluded'

# Features under which HazusEQInventoryCompiler keeps the GROSS_AREA_KEY
# feature that NSIFootprintJoin writes and the one a local inventory gives.
NSI_GROSS_AREA_KEY = 'NSIGrossArea'
LOCAL_GROSS_AREA_KEY = 'LocalGrossArea'

# Feature of HazusEQInventoryCompiler's record of the footprints its unit
# budget dropped.
DROPPED_KEY = 'Dropped'

# SimCenter source string that captures data provenance
SOURCE_STRING_KEY = 'SourceString'

# Feature holding, per world, the id of the donor whose values a target kept.
IMPUTATION_DONOR_KEY = 'ImputationDonor'

# Feature holding number of stories inferred from height.
STORIES_FROM_HEIGHT_KEY = 'StoriesFromHeight'

# Where a building's gross area comes from, as the unit budget and the
# replacement cost imputer log it: its GROSS_AREA_KEY, its plan area times
# its STORIES_KEY, its plan area as one story, or the feature a caller of the
# imputer names as holding it.
AREA_FROM_GROSS_AREA = 'gross_area'
AREA_FROM_STORIES = 'stories'
AREA_FROM_PLAN_AREA = 'plan_area'
AREA_FROM_FEATURE = 'feature'

# The year of the TIGER/Line release read when no vintage is given.
DEFAULT_TIGER_VINTAGE = 2023
