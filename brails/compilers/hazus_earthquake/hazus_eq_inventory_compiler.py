# Copyright (c) 2026 The Regents of the University of California
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
Compile a Hazus earthquake building inventory over several possible worlds.

The section "Hazus earthquake inventory compiler" of the README beside this
module describes the run.

.. autosummary::

      HazusEQInventoryCompiler
      CompilerInputs
      CompilationResult
"""

from __future__ import annotations

import inspect
import json
import math
import numbers
import operator
import secrets
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping

from brails.aggregators.points_to_polygons.nsi_footprint_join import NSIFootprintJoin
from brails.aggregators.stacked_footprints.stacked_footprints import (
    RULE_COPY,
    RULE_STACKED,
    StackedFootprintMerge,
)
from brails.compilers.compilation_report import (
    CONFLICT_CATEGORY,
    DIAGNOSIS_NOT_SELECTED,
    DROPPED_CATEGORY,
    FAILS_SCHEMA_CATEGORY,
    MISSING_INPUT_CATEGORY,
    NO_PLAUSIBLE_VALUE_CATEGORY,
    UNDETERMINED_CATEGORY,
    UNSUPPORTED_CATEGORY,
    CompilationReport,
)
from brails.compilers.hazus_earthquake.pair_redraw import PairRedraw
from brails.compilers.hazus_earthquake.resolution_loop import (
    STAGE_DESIGN_LEVEL,
    STAGE_RESOLUTION,
    STAGE_STRUCTURE_TYPE,
    ComposedScreen,
    ResolutionLoop,
)
from brails.compilers.source_composition import (
    _build_inventory,
    _select_assets,
    compose_source_parts,
)
from brails.constants import (
    BLOCK_GROUP_KEY,
    BUILDING_TYPE_KEY,
    DEFAULT_UNITS,
    DESIGN_LEVEL_KEY,
    DROPPED_KEY,
    EXCLUDED_KEY,
    FOOTPRINT_DATASETS_KEY,
    FOOTPRINT_PROVIDER_KEY,
    FOOTPRINT_RELEASE_KEY,
    GEOGRAPHIC_CRS,
    GROSS_AREA_KEY,
    HEIGHT_CLASS_KEY,
    LOCAL_GROSS_AREA_KEY,
    NSI_GROSS_AREA_KEY,
    OCCUPANCY_KEY,
    PLAN_AREA_KEY,
    REPLACEMENT_COST_KEY,
    SOURCE_STRING_KEY,
    STORIES_FROM_HEIGHT_KEY,
    STORIES_KEY,
    STRUCTURE_TYPE_KEY,
    TRACT_GEOID_LENGTH,
    UNITS_KEY,
    YEAR_BUILT_KEY,
)
from brails.imputers.observed_donor_imputer.donor_records import (
    OUTCOME_NO_AREA,
    OUTCOME_NO_DONOR,
)
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    DEFAULT_DONOR_CAP_M,
    ObservedDonorImputer,
    imputation_donors_from_log,
)
from brails.imputers.replacement_cost.replacement_cost_imputer import (
    OUTCOME_NO_CLASS,
    OUTCOME_VALUE_CODES,
    ReplacementCostImputer,
    footprint_gross_area,
)
from brails.imputers.residential_neighbor_year_sampler.residential_neighbor_year_sampler import (
    ResidentialNeighborYearSampler,
)
from brails.inferers.acs_unit_budget.acs_unit_budget import (
    DEFAULT_MH_PARK_RADIUS_M,
    DROP_PROBABILITY,
    GAP_NO_ACS,
    GEOID_COLUMN,
    ACSUnitBudget,
    UnitBudgetResult,
)
from brails.inferers.acs_year_built_sampler.acs_year_built_sampler import (
    B25034_BINS,
    B25127_CELLS,
    ACSYearBuiltSampler,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
    GENERIC_DESIGN_LEVELS,
    STATE_DESIGN_LEVELS,
    HazusEQDesignLevelInferer,
    apply_w1_carveout,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_height_class_inferer import (
    HazusEQHeightClassInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_structure_type_inferer import (
    HazusEQStructureTypeInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    WEST_COAST,
    get_hazus_general_type_membership,
    get_hazus_height_classes_by_type,
    get_hazus_state_region_mapping,
)
from brails.inferers.hazus_inferer_earthquake.hazus_table_completion import (
    is_implausible,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_NO_DONOR,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
)
from brails.inferers.stories_from_height_inferer.stories_from_height_inferer import (
    StoriesFromHeightInferer,
    parse_sources,
)
from brails.scrapers.mobile_home_park_scraper.mobile_home_park_scraper import (
    MobileHomeParkScraper,
)
from brails.scrapers.nsi_parser.nsi_parser import NSI_Parser
from brails.scrapers.overture_maps_scraper.overture_maps_footprint_scraper import (
    OvertureMapsFootprintScraper,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    OVERTURE_PROVIDER,
    SOURCES_KEY,
)
from brails.scrapers.us_census_scrapers.acs_table_scraper import (
    DEFAULT_ACS_VINTAGE,
    ACSTableScraper,
)
from brails.scrapers.us_census_scrapers.census_tract_scraper import (
    LEVEL_BLOCK_GROUP,
    LEVEL_TRACT,
    CensusTractScraper,
)
from brails.scrapers.us_census_scrapers.tiger_line_scraper import (
    STATE_NAME_COLUMN,
    TIGERLineScraper,
)
from brails.types.asset_inventory import (
    DEFAULT_VECTOR_FEATURES,
    HAZUS_EQ_SOURCE_FEATURES,
    SOURCE_CODES,
    SOURCE_LABELS,
    AssetInventory,
    is_missing_value,
    read_source_string,
)
from brails.types.census_boundary import CensusBoundary
from brails.types.geometry_region import GeometryRegion
from brails.utils.region_filter import (
    find_inside,
    read_boundary,
    reference_point,
    resolve_region,
)
from brails.utils.stage_freeze import StageFreeze
from brails.utils.tiger_files import DEFAULT_TIGER_VINTAGE, read_tiger_file
from brails.utils.unit_converter import UNIT_TO_METER
from brails.utils.unit_count_rules import UNIT_COUNT_CLASSES
from brails.validators.schema_validator import SchemaValidator

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from shapely.geometry.base import BaseGeometry

    from brails.types.region_boundary import RegionBoundary

# The number of possible worlds compiled by default.
DEFAULT_N_WORLDS = 10

# The source labels of a world's merge, highest precedence first.
PRECEDENCE = ('local', 'footprint', 'nsi', 'derived', 'imputed', 'drawn')

# The label of the footprints, the spine of every merge: a merged inventory
# holds the footprints' assets and geometries.
SPINE_LABEL = 'footprint'

# The features the compiler writes for itself and reads back, which a local
# inventory may not hold: its precedence would put them in place of the
# compiler's.
RESERVED_LOCAL_FEATURES = (BLOCK_GROUP_KEY, NSI_GROSS_AREA_KEY)

# The properties a written building holds besides its id and its geometry,
# unless write is asked for more. Their choice is described in README.rst.
DEFAULT_PROPERTIES = (
    OCCUPANCY_KEY,
    UNITS_KEY,
    STORIES_KEY,
    YEAR_BUILT_KEY,
    DESIGN_LEVEL_KEY,
    STRUCTURE_TYPE_KEY,
    HEIGHT_CLASS_KEY,
    REPLACEMENT_COST_KEY,
    BLOCK_GROUP_KEY,
    PLAN_AREA_KEY,
    GROSS_AREA_KEY,
    BUILDING_TYPE_KEY,
    SOURCE_STRING_KEY,
)

# The value of write's extra_properties that writes every feature the
# compiled inventory holds.
ALL_PROPERTIES = 'all'

# The property the integer-id form of the file numbers its features within.
_TYPE_KEY = 'type'

# The log naming each default property a compiled building lacks where it
# applies, as (asset id, world, property); write refuses a file it names.
MISSING_DEFAULTS_LOG = 'missing_defaults'

# Every log a run can hold, in the order the steps write them. A run holds
# stacked_parts and stacked_asset_ids only with merge_stacked. write without
# detailed removes a file at the stem under any of these names.
LOG_NAMES = (
    'stacked_parts',
    'stacked_asset_ids',
    'footprint_ties',
    'join_links',
    'join_unlinked',
    'join_branch_counts',
    'stories',
    'budget',
    'budget_diagnostic',
    'budget_gaps',
    'budget_nonresidential',
    'imputer',
    'year_fit',
    'pair_redraw',
    'resolution',
    'neighbor_year',
    'nonresidential_resolution',
    'schema_check',
    'gross_area',
    'value',
    MISSING_DEFAULTS_LOG,
)

# The frozen stages, in the order they are frozen.
STAGE_S01_FOOTPRINTS = 's01_footprints'
STAGE_S01_LOCAL = 's01_local'
STAGE_S02_NSI_JOIN = 's02_nsi_join'
STAGE_S02_DERIVED = 's02_derived'
STAGE_S03_STORY_COUNTS = 's03_story_counts'
STAGE_S03_WITHHELD = 's03_withheld'
STAGE_S04_UNIT_BUDGET = 's04_unit_budget'
STAGE_S04_DROPPED = 's04_dropped'
STAGE_S04_WITHHELD = 's04_withheld'
STAGE_S05_IMPUTATION = 's05_imputation'
STAGE_S05_VIEW = 's05_view'
STAGE_S05_DONORS = 's05_donors'
STAGE_S06_RESIDENTIAL_YEAR = 's06_residential_year'
STAGE_S07_WITHHELD = 's07_withheld'
STAGE_S07_PAIR_REDRAW = 's07_pair_redraw'
STAGE_S07_DERIVED = 's07_derived'
STAGE_S08_STRUCTURE = 's08_structure'
STAGE_S08_DERIVED = 's08_derived'
STAGE_S08_VIEW = 's08_view'
STAGE_S09_NONRESIDENTIAL_YEAR = 's09_nonresidential_year'
STAGE_S10_NONRESIDENTIAL_STRUCTURE = 's10_nonresidential_structure'
STAGE_S10_DERIVED = 's10_derived'
STAGE_S12_WITHHELD_LOCAL = 's12_withheld_local'
STAGE_S12_WITHHELD_JOIN = 's12_withheld_join'
STAGE_S12_WITHHELD_FOOTPRINT = 's12_withheld_footprint'
STAGE_S13_WITHHELD = 's13_withheld'
STAGE_S14_WITHHELD = 's14_withheld'
STAGE_S14_WITHHELD_FOOTPRINT = 's14_withheld_footprint'
STAGE_S12_GROSS_AREA = 's12_gross_area'
STAGE_S13_REPLACEMENT_COST = 's13_replacement_cost'
STAGE_S14_BUILDING_TYPE = 's14_building_type'
STAGE_S15_OUTPUT = 's15_output'

# The stages the report records actions and exclusions under.
REPORT_STAGES = (
    's01_footprints',
    's02_nsi_join',
    's03_story_counts',
    's04_unit_budget',
    's05_imputation',
    's06_residential_year',
    's07_pair_redraw',
    's08_design_level',
    's08_resolution',
    's08_structure_type',
    's08_height_class',
    's08_carve_out',
    's09_nonresidential_year',
    's10_design_level',
    's10_resolution',
    's10_structure_type',
    's10_height_class',
    's10_carve_out',
    's11_schema_check',
    's13_replacement_cost',
    's14_building_type',
)

# The categories the run files an exclusion under, besides the report's
# DROPPED_CATEGORY. The section "Exclusion categories" of the compilers'
# README defines each.
EXCLUSION_CATEGORIES = (
    MISSING_INPUT_CATEGORY,
    UNSUPPORTED_CATEGORY,
    CONFLICT_CATEGORY,
    NO_PLAUSIBLE_VALUE_CATEGORY,
    UNDETERMINED_CATEGORY,
    FAILS_SCHEMA_CATEGORY,
)

# The Hazus input schema every compiled realization is checked against.
DLML_DATASET = 'seismic/building/portfolio/Hazus v5.1'

# The state and design level method whose table assumes Moderate-Code for a
# Pre-Code W1 building (Table C-10 of the Hazus 5.1 technical manual).
CARVE_OUT_STATE = 'California'
CARVE_OUT_METHOD = '5.1'

# The state a compilation applies where neither the compiler nor its inputs name
# one, and the 1997 UBC seismic zone it applies where the compiler is given none.
DEFAULT_STATE = 'California'
DEFAULT_SEISMIC_ZONE = '4'

# The design level method a compiler applies by default.
DEFAULT_DESIGN_LEVEL_METHOD = '5.1'

# The NSI join's distance cap in meters that a compiler applies by default.
DEFAULT_JOIN_CAP_M = 15.0

# The share of a boundary's area that a second state may not reach in
# CompilerInputs.from_place, measured against the TIGER/Line state polygons, which
# hold coastal and inland water.
STATE_AREA_SHARE = 0.01

# The most block groups the area of CompilerInputs.from_place may intersect unless
# the caller lifts the limit.
DEFAULT_MAX_BLOCK_GROUPS = 800

# The equal-area projection the state shares of a boundary are measured in.
_EQUAL_AREA_CRS = 'EPSG:6933'

# The exclusion category of each diagnosis by which a year sampler, the
# design level inferer or the height class inferer refuses an asset. An asset
# ACSYearBuiltSampler finds infeasible lies in a block group the census tables
# do not cover. ResidentialNeighborYearSampler gives no_donor to an asset that
# has no residential neighbor with a construction year, which is filed, as
# s13's building without a donor is, as having no plausible value.
_DIAGNOSIS_CATEGORY = {
    DIAGNOSIS_INVALID: MISSING_INPUT_CATEGORY,
    DIAGNOSIS_UNSUPPORTED: UNSUPPORTED_CATEGORY,
    DIAGNOSIS_INFEASIBLE: MISSING_INPUT_CATEGORY,
    DIAGNOSIS_NO_DONOR: NO_PLAUSIBLE_VALUE_CATEGORY,
}

# The feature of the stage of s13's values holding the source label of each
# value.
_VALUE_SOURCE_KEY = 'ReplacementCostSource'

# The occupancy class whose NSI story count is kept over the height count.
NSI_STORIES_CLASS = 'RES1'

# The settings the compiler gives ObservedDonorImputer, which imputer_options
# override key by key. Their origin is described in README.rst.
DEFAULT_IMPUTER_OPTIONS: Mapping[str, object] = MappingProxyType(
    {'k': 10, 'location_scale_m': 500.0, 'area_weight': 2.0}
)

# The option names the compiler sets itself, which an options mapping may
# not hold, per configured class.
_RESERVED_OPTIONS = {
    # The budget and the pair re-draw need the imputer's default status.
    'imputer_options': ('seed', 'status_key', 'status_values', 'status_name'),
    'budget_options': ('b25024', 'mh_park_points', 'imputer', 'seed'),
    # The compiler reads the features under their default names.
    'sampler_options': (
        'b25034_block_groups',
        'b25034_tracts',
        'b25127_tracts',
        'seed',
        'world',
        'occupancy_key',
        'units_key',
        'block_group_key',
        'year_built_key',
    ),
}

# The ACS table and the level the compiler reads for each table of CompilerInputs
# given as None.
_ACS_TABLE_REQUESTS = {
    'b25024_block_groups': ('B25024', LEVEL_BLOCK_GROUP),
    'b25034_block_groups': ('B25034', LEVEL_BLOCK_GROUP),
    'b25034_tracts': ('B25034', LEVEL_TRACT),
    'b25127_tracts': ('B25127', LEVEL_TRACT),
}

# The estimates ACSYearBuiltSampler reads of each table it takes.
_SAMPLER_VARIABLES = {
    'b25034_block_groups': [row[3] for row in B25034_BINS],
    'b25034_tracts': [row[3] for row in B25034_BINS],
    'b25127_tracts': [name for _, _, pair in B25127_CELLS for name in pair],
}

# Position of each feature in the source string.
_CODE_POSITION = {
    feature: position for position, feature in enumerate(HAZUS_EQ_SOURCE_FEATURES)
}

# The sources of an observed occupancy class: the local inventory, NSI, and the
# classes the NSI join sets from a unit count band. An asset whose occupancy
# class comes from one of them has an observed occupancy class: it is not a
# target of the unit budget, and with a story count it is an observed donor.
# The second tuple holds the sources whose occupancy class the unit budget's
# mobile-home rule may relabel.
_OCCUPANCY_SOURCE_LABELS = ('local', 'nsi', 'derived')
_RELABELED_SOURCES = ('nsi', 'derived')

# The structure types for which Hazus defines no height class.
_WITHOUT_HEIGHT_CLASS = frozenset(
    structure_type
    for structure_type, classes in get_hazus_height_classes_by_type().items()
    if classes is None
)

# The general building type letter of each Hazus structure type, by the
# pairing the structure type inferer restricts its draw with.
_LETTER_OF_TYPE = {
    structure_type: letter
    for letter, structure_types in get_hazus_general_type_membership().items()
    for structure_type in structure_types
}


class CompilerInputs(NamedTuple):
    """
    The data of one study area.

    Attributes:
        footprints (AssetInventory):
            The building footprints, polygons in EPSG:4326 keyed by string
            asset ids, with one possible world. Each holds
            ``footprint_provider``, for an Overture footprint ``sources``,
            and ``type`` where the file is written with ``integer_ids``. The
            ids are taken as given, unless the compiler merges stacked
            footprints, which re-keys them.
        nsi_points (AssetInventory):
            The NSI points keyed by ``fd_id``, as
            :class:`~brails.scrapers.NSIFileReader` reads them.
        b25024_block_groups (pandas.DataFrame | None):
            Table B25024 per block group, as
            :class:`~brails.inferers.acs_unit_budget.acs_unit_budget.ACSUnitBudget`
            takes it. ``None``, the default, has the compiler read it from the Census
            Data API, which needs a Census API key.
        b25034_block_groups (pandas.DataFrame | None):
            Table B25034 per block group, as
            :class:`~brails.inferers.acs_year_built_sampler.acs_year_built_sampler.ACSYearBuiltSampler`
            takes it. ``None``, the default, has the compiler read it, as for
            ``b25024_block_groups``.
        b25034_tracts (pandas.DataFrame | None):
            Table B25034 per tract, in the same form. ``None``, the default, has the
            compiler read it, as for ``b25024_block_groups``.
        b25127_tracts (pandas.DataFrame | None):
            Table B25127 per tract, in the same form. ``None``, the default, has the
            compiler read it, as for ``b25024_block_groups``.
        mh_park_points (object | None):
            The mobile-home park points, in any form
            :class:`~brails.inferers.acs_unit_budget.acs_unit_budget.ACSUnitBudget`
            accepts for ``mh_park_points``, or an ``AssetInventory`` of points such
            as :class:`~brails.scrapers.MobileHomeParkScraper` returns. An empty
            array compiles without park points. ``None``, the default, has the
            compiler read the HIFLD points within the budget's ``mh_park_radius_m``
            of the footprints' bounding box, which needs network access.
        footprint_release (str | None):
            The Overture release the footprints come from, written to every
            Overture footprint as ``overture_<release>``. ``None`` writes
            no release.
        local (AssetInventory | None):
            Values known locally, keyed by asset ids of ``footprints``,
            with one possible world. They take precedence over every other
            source and are never changed. They may hold no feature of
            :data:`RESERVED_LOCAL_FEATURES`. Not accepted when the compiler
            merges stacked footprints.
        study_area (str):
            A label of the study area for the report, such as a FIPS code.
        height_unit (str):
            The length unit of the footprints' heights, as the footprint
            scraper wrote them. Defaults to the length unit of
            :data:`brails.constants.DEFAULT_UNITS`.
        block_groups (geopandas.GeoDataFrame | None):
            One polygon per census block group, with a ``GEOID`` column, as a
            TIGER/Line file holds them. ``None``, the default, has the compiler read
            the block groups that intersect the footprints' bounding box from the
            TIGER/Line files of its ``block_group_vintage``, which needs network
            access. Every footprint must lie in a block group either way.
        boundary (shapely.geometry.Polygon | shapely.geometry.MultiPolygon | None):
            The place to compile, in EPSG:4326, holes kept. Give the footprints of
            the place and of an area around it. The result's inventory and the
            files :meth:`CompilationResult.write` saves hold the buildings inside
            the place. ``None``, the default, keeps every footprint.
        buffer_m (float | None):
            How far in meters the area around ``boundary`` reaches past it, which
            the report records and compares with the donor cap. Read only with
            ``boundary``.
        boundary_record (Mapping[str, object] | None):
            Where ``boundary`` comes from, such as the provider and the GEOID of a
            census place, which the report records. Read only with ``boundary``.
        state (str | None):
            The full name of the study area's state, which a compiler built
            without a state applies. ``None``, the default, leaves the state to the
            compiler.
        merge_stacked (bool):
            Whether a compiler built without ``merge_stacked`` merges the stacked
            footprints. Defaults to ``False``.
        notices (tuple[str, ...]):
            Notices about the inputs, which the report lists first, such as those
            :meth:`from_place` writes. Defaults to an empty tuple.
    """

    footprints: AssetInventory
    nsi_points: AssetInventory
    b25024_block_groups: pd.DataFrame | None = None
    b25034_block_groups: pd.DataFrame | None = None
    b25034_tracts: pd.DataFrame | None = None
    b25127_tracts: pd.DataFrame | None = None
    mh_park_points: object = None
    footprint_release: str | None = None
    local: AssetInventory | None = None
    study_area: str = ''
    height_unit: str = DEFAULT_UNITS['length']
    block_groups: gpd.GeoDataFrame | None = None
    boundary: Polygon | MultiPolygon | None = None
    buffer_m: float | None = None
    boundary_record: Mapping[str, object] | None = None
    state: str | None = None
    merge_stacked: bool = False
    notices: tuple[str, ...] = ()

    @classmethod
    def from_place(
        cls,
        place: str | RegionBoundary | Polygon | MultiPolygon,
        *,
        source: str | None = None,
        buffer_m: float = DEFAULT_DONOR_CAP_M,
        overture_options: Mapping[str, object] | None = None,
        block_group_vintage: int = DEFAULT_TIGER_VINTAGE,
        max_block_groups: int | None = DEFAULT_MAX_BLOCK_GROUPS,
        compiler: HazusEQInventoryCompiler | None = None,
    ) -> CompilerInputs:
        """
        Gather the inputs of a compilation of a place from its name or boundary.

        The section "A compilation from a place" of the README beside this module
        states the steps. The area read is the place and the block groups with land
        that overlap it, widened by ``buffer_m``; its Overture footprints are read
        once. The area compiled is the place and the block groups that hold its
        buildings, widened by ``buffer_m``. The inputs hold the footprints and NSI
        points of the area compiled and the block groups of the area read, leave the
        census tables and the park points as ``None`` for the compiler to read, and
        ask it to merge stacked footprints.

        Args:
            place (str | RegionBoundary | Polygon | MultiPolygon):
                A place name with its state, as in ``'Berkeley, CA'``, looked up
                where ``source`` says; a :class:`RegionBoundary` of any kind; or a
                Shapely polygon or multipolygon in longitude and latitude.
            source (str, optional):
                Where a place name is looked up: ``None`` or ``'census'`` for the
                census places and counties, ``'osm'`` for OpenStreetMap.
            buffer_m (float, optional):
                How far in meters both areas reach past the place and its block
                groups. Defaults to the observed-donor imputer's default ``cap_m``.
            overture_options (Mapping[str, object], optional):
                Options of :class:`~brails.scrapers.OvertureMapsFootprintScraper`,
                such as ``'overtureRelease'`` or ``'local_path'``. The footprints
                are keyed by geohash at the scraper's area floor, so
                ``'id_scheme'`` and ``'min_area'`` are refused.
            block_group_vintage (int, optional):
                The year of the TIGER/Line release of the block groups and of a
                census boundary looked up by name. Defaults to
                :data:`~brails.utils.tiger_files.DEFAULT_TIGER_VINTAGE`.
            max_block_groups (int | None, optional):
                The most block groups the area read may intersect, counted before
                the footprints are read. ``None`` sets no limit. Defaults to
                :data:`DEFAULT_MAX_BLOCK_GROUPS`.
            compiler (HazusEQInventoryCompiler, optional):
                The compiler that will compile the inputs, whose
                :meth:`~HazusEQInventoryCompiler.check_state` is called for the
                place's state before the footprints are read, as
                :meth:`HazusEQInventoryCompiler.compile_place` passes it. The inputs
                are checked against that compiler only; another compiler checks
                them in ``compile``. ``None``, the default, leaves the check to
                ``compile``.

        Returns:
            CompilerInputs:
                The footprints and NSI points of the area compiled, the block groups
                of the area read, the boundary with ``buffer_m`` and its
                ``boundary_record``, the state, the Overture release,
                ``merge_stacked`` set on, and the notices for the report.

        Raises:
            TypeError:
                If ``place`` is none of the accepted types, or ``buffer_m`` is not
                a real number.
            ValueError:
                If ``buffer_m`` is less than 0 or not finite, or
                ``overture_options`` sets ``'id_scheme'`` or ``'min_area'``; if
                :func:`~brails.utils.region_filter.resolve_region` or the census
                lookup refuses the place; if the boundary encloses no area, even as
                repaired; if no state, or a second state, holds
                :data:`STATE_AREA_SHARE` of a boundary that is not a census one;
                if the compiler cannot take the place's state, or ``compiler`` was
                built for another state or has no seismic zone for this one; if the
                area read intersects more than ``max_block_groups`` block groups;
                or if no footprint lies inside the place.
            LookupError, RuntimeError:
                If no Census API key is found or the Census Data API refuses it.
            requests.exceptions.RequestException:
                If a request fails, as the scrapers raise it.
        """
        if isinstance(buffer_m, bool) or not isinstance(buffer_m, numbers.Real):
            msg = f'buffer_m must be a number of meters, not {buffer_m!r}.'
            raise TypeError(msg)
        if not (math.isfinite(buffer_m) and buffer_m >= 0):
            msg = f'buffer_m must be a finite number of at least 0, not {buffer_m}.'
            raise ValueError(msg)
        fixed = sorted(set(overture_options or {}) & {'id_scheme', 'min_area'})
        if fixed:
            msg = (
                f'overture_options sets {" and ".join(fixed)}, which from_place '
                "fixes: the footprints are keyed by geohash at the scraper's area "
                'floor, on which the stacked merge rests. Remove it from '
                'overture_options.'
            )
            raise ValueError(msg)
        # Checked first, so that a missing or refused key stops before a download.
        ACSTableScraper().check_key(DEFAULT_ACS_VINTAGE)

        region = resolve_region(place, source, vintage=block_group_vintage)
        boundary, name, identifier = read_boundary(region)
        record = _record_boundary(region, name, identifier)
        notices = []
        if (
            isinstance(region, CensusBoundary)
            and region.vintage != block_group_vintage
        ):
            notices.append(
                f'The boundary comes from the TIGER/Line {region.vintage} files and '
                f'the block groups from the {block_group_vintage} files.'
            )
        state = _find_state(boundary, name, record, block_group_vintage, compiler)

        tiger = TIGERLineScraper(block_group_vintage)
        overlapping = tiger.get_block_groups(GeometryRegion(boundary, name))
        # The block groups likely to hold a building of the place have land and
        # overlap it, and the area read is the place and those, widened. A building
        # in a block group without land is noted below.
        land = overlapping.loc[overlapping['ALAND'] > 0].to_crs(GEOGRAPHIC_CRS)
        reach = _widen(shapely.union_all([boundary, *land.geometry]), buffer_m)
        block_groups = tiger.get_block_groups(GeometryRegion(reach, name))
        if max_block_groups is not None and len(block_groups) > max_block_groups:
            msg = (
                f'The area read around {name} intersects {len(block_groups)} block '
                f'groups, more than max_block_groups={max_block_groups}. Pass '
                'max_block_groups=None to compile it, with the memory it needs.'
            )
            raise ValueError(msg)

        scraper = OvertureMapsFootprintScraper(
            {'id_scheme': 'geohash', **(overture_options or {})}
        )
        read = scraper.get_footprints(GeometryRegion(reach, name)).inventory
        inside = find_inside(read, boundary)
        if not inside:
            msg = (
                f'No Overture footprint read lies inside {name}. For a polygon, '
                'check that it is in longitude and latitude (EPSG:4326); with '
                "'local_path', check that the files cover the place."
            )
            raise ValueError(msg)
        # The block group of each footprint read, by the point find_inside reads.
        ids = list(read)
        xy = np.array(
            [reference_point(read[a], a) for a in ids], dtype=float
        ).reshape(-1, 2)
        located = (
            gpd.sjoin(
                gpd.GeoDataFrame(
                    {'asset_id': ids},
                    geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
                    crs=GEOGRAPHIC_CRS,
                ),
                block_groups[['GEOID', 'ALAND', 'geometry']].to_crs(GEOGRAPHIC_CRS),
                predicate='within',
                how='left',
            )
            .drop_duplicates('asset_id')
            .set_index('asset_id')
        )
        held = located.loc[inside]
        unread = held.index[held['ALAND'] == 0]
        if len(unread):
            notices.append(
                f'{len(unread)} building(s) inside {name} lie in block groups '
                'without land, which can reach past the area read, so those block '
                f'groups may be compiled in part: {sorted(unread)[:10]}.'
            )
        chosen = block_groups[block_groups['GEOID'].isin(held['GEOID'])]
        area = _widen(
            shapely.union_all([boundary, *chosen.to_crs(GEOGRAPHIC_CRS).geometry]),
            buffer_m,
        )
        compiled = located.loc[find_inside(read, area)]
        unplaced = compiled['GEOID'].isna()
        if unplaced.any():
            notices.append(
                f'{int(unplaced.sum())} footprint(s) of the area compiled lie in no '
                'block group, such as across a national border or offshore, and '
                'were left out.'
            )
        footprints = AssetInventory()
        footprints.inventory = {a: read[a] for a in compiled.index[~unplaced]}

        nsi_points = NSI_Parser().get_raw_data(
            GeometryRegion(_widen(area, DEFAULT_JOIN_CAP_M), name), omit_nulls=True
        )
        return cls(
            footprints=footprints,
            nsi_points=nsi_points,
            footprint_release=scraper.overture_release,
            study_area='' if isinstance(region, (Polygon, MultiPolygon)) else name,
            height_unit=scraper.units['length'],
            block_groups=block_groups,
            boundary=boundary,
            buffer_m=buffer_m,
            boundary_record=record,
            state=state,
            merge_stacked=True,
            notices=tuple(notices),
        )


class CompilationResult(NamedTuple):
    """
    The compiled inventory, with the report, frozen stages and logs of the run.

    Attributes:
        inventory (AssetInventory):
            Every footprint the run kept inside the inputs' ``boundary``, or every
            footprint it kept where the inputs give none, over the worlds 0 to
            ``n_worlds - 1``, with every feature the run gave it. A feature equal in
            every world is held as a scalar and otherwise as a list in world order.
            Each asset holds :data:`~brails.constants.EXCLUDED_KEY`, true in a world
            it was excluded from. A run that keeps no footprint gives an inventory
            without assets, which holds one world. :meth:`write` still writes the
            report rows of every world the run computed.
        report (CompilationReport):
            The actions, the exclusions, the timings and the summary of the
            run. With a boundary, its summary, the rows it saves and its exclusion
            groups are those of the buildings inside, unless given ``keep_outside``.
        freeze (StageFreeze):
            Every frozen stage, in freeze order.
        manifest (dict):
            The freeze's manifest at the end of the run.
        logs (Mapping[str, pandas.DataFrame]):
            The components' logs, by name.
        seed (int):
            The seed every component drew from.
        area_inventory (AssetInventory):
            Every footprint the run kept, inside the boundary and around it, in the
            form of :attr:`inventory`. The assets of :attr:`inventory` are this
            inventory's assets, not copies. Without a boundary it is
            :attr:`inventory`.
        boundary_assets (frozenset[str] | None):
            The ids of the footprints of ``s01_footprints`` inside the boundary,
            those the run kept and those it dropped. ``None`` without a boundary.
    """

    inventory: AssetInventory
    report: CompilationReport
    freeze: StageFreeze
    manifest: dict
    logs: Mapping[str, pd.DataFrame]
    seed: int
    area_inventory: AssetInventory
    boundary_assets: frozenset[str] | None

    def write(  # noqa: C901
        self,
        path: str | os.PathLike[str],
        *,
        world: int | None = None,
        integer_ids: bool = False,
        keep_partly_excluded: bool = False,
        extra_properties: str | Sequence[str] = (),
        detailed: bool = False,
        keep_outside: bool = False,
    ) -> None:
        """
        Write every world, or one, as GeoJSON, with the report and the manifest.

        The GeoJSON at ``path`` holds every world the run computed, a
        property whose worlds differ as a list in which position ``i`` is
        world ``i``. Given ``world``, it holds that one world only, and a
        building's property whose value is ``None`` in that world is left
        out, so that the file does not depend on how many worlds the run
        computed. A building excluded in every written world, by the
        report's exclusion rows of those worlds, is not written, and one
        excluded in some is written only with ``keep_partly_excluded``. With
        ``keep_partly_excluded``, and only then, each building also holds
        :data:`~brails.constants.EXCLUDED_KEY`; for one world, the option
        only adds that property, false for every building written.
        Each building holds every property of :data:`DEFAULT_PROPERTIES`
        that applies to it, as the section "The run" of the README beside
        this module states, and those of ``extra_properties``. In a world
        it is excluded from, a kept building may lack them. With
        ``integer_ids`` the file takes the form the R2D tools read: each
        building also holds
        ``type``, within which :meth:`AssetInventory.write_to_geojson`
        numbers the features, and its asset id as ``unique_id``.

        Beside the GeoJSON, under the stem of ``path``, :meth:`write` saves
        the report's summary as ``<stem>_report_summary.json`` and its
        exclusion rows as ``<stem>_report_exclusions.parquet``, through
        :meth:`CompilationReport.save`, and the manifest through
        :meth:`StageFreeze.save_manifest` as ``<stem>_manifest.json``. With
        ``detailed`` it also saves the report's action rows as
        ``<stem>_report.parquet`` and every log of :attr:`logs` through
        :meth:`CompilationReport.save_logs` as ``<stem>_<name>.parquet``.
        Before anything is written, every call removes the file at
        ``<stem>_<name>.parquet`` for each name of :data:`LOG_NAMES` that
        :attr:`logs` does not hold, and a call without ``detailed`` that is not
        refused also removes ``<stem>_report.parquet`` and the files of every
        other name of :data:`LOG_NAMES`, whoever wrote them. The stem then
        holds no action rows and no log beside the files this call writes. A
        refused file leaves a GeoJSON that an earlier call wrote at ``path`` in
        place. Where a write fails after the removal, the stem keeps what an
        earlier call wrote there, less the removed files. The report and the
        logs keep the rows of the written worlds. The summary's notices,
        timings and schema check cover every world the run computed.
        :attr:`logs` holds every log whatever ``detailed`` says.

        After a compilation with a boundary, :meth:`write` writes the buildings of
        :attr:`inventory`, those inside the boundary, and saves the report rows and
        the summary of those buildings. The refusal for a missing default property
        concerns them only. With ``keep_outside`` it writes :attr:`area_inventory`
        and the rows of every building, as after a compilation without a boundary.
        The logs saved with ``detailed`` hold every building either way, since
        several of them are keyed by block group.

        Args:
            path (str or os.PathLike):
                The GeoJSON file to write.
            world (int, optional):
                The number of the world to write alone, from 0 to one less
                than the compiler's ``n_worlds``. ``None`` writes every world.
            integer_ids (bool, optional):
                Whether to number the features, as
                :meth:`AssetInventory.write_to_geojson` does with it.
            keep_partly_excluded (bool, optional):
                Whether to write the buildings excluded in some, but not
                all, of the written worlds.
            extra_properties (str or Sequence[str], optional):
                Features of the written inventory to write beside the default
                properties, or :data:`ALL_PROPERTIES` for every feature it
                holds. They change no value of the default properties.
            detailed (bool, optional):
                Whether to save the report's action rows and the logs too.
                Defaults to ``False``. A refused file saves them either way.
            keep_outside (bool, optional):
                Whether to write the buildings around the boundary too. Defaults to
                ``False``. Without a boundary it changes nothing.

        Raises:
            TypeError:
                If ``world`` is a bool or not an integer, or
                ``extra_properties`` is a string other than
                :data:`ALL_PROPERTIES`; nothing is then written.
            ValueError:
                If ``world`` is not one of the worlds the run computed, or
                ``extra_properties`` names a feature no building of the
                written inventory holds or names
                :data:`~brails.constants.EXCLUDED_KEY`; nothing is then
                written.
            RuntimeError:
                If the log :data:`MISSING_DEFAULTS_LOG` names a default
                property that a compiled building lacks in a written world,
                once the report with its action rows, every log and the
                manifest are written, whatever ``detailed`` says, and before
                the GeoJSON is; or if a frozen stage changed after it was
                frozen, in which case every file this call writes but the
                manifest has been written.
            OSError:
                If a file the call removes cannot be removed, before anything
                is written. The message names the file, and the files
                removed before it stay removed.
        """
        compiled = self.area_inventory if keep_outside else self.inventory
        # The report holds the run's worlds and checks world numbers against
        # them. An inventory without assets holds one world, whatever the run
        # computed, so the worlds are not read from the inventory.
        worlds = (
            tuple(
                self.report._selected_worlds(None)  # noqa: SLF001
                or range(compiled.n_pw)
            )
            if world is None
            else (self.report._checked_world(world),)  # noqa: SLF001
        )
        if isinstance(extra_properties, str) and extra_properties != ALL_PROPERTIES:
            msg = (
                'extra_properties must be a sequence of feature names or '
                f'{ALL_PROPERTIES!r}, not {extra_properties!r}.'
            )
            raise TypeError(msg)
        everything = extra_properties == ALL_PROPERTIES
        extra = () if everything else tuple(extra_properties)
        if EXCLUDED_KEY in extra:
            msg = (
                f'{EXCLUDED_KEY} is written with keep_partly_excluded, not as an '
                'extra property.'
            )
            raise ValueError(msg)
        held = {
            key for asset in compiled.inventory.values() for key in asset.features
        }
        unknown = sorted(set(extra) - held)
        if unknown:
            msg = f'No building of the inventory holds {unknown}.'
            raise ValueError(msg)
        selected = {
            *DEFAULT_PROPERTIES,
            *extra,
            *([EXCLUDED_KEY] if keep_partly_excluded else []),
            *([_TYPE_KEY] if integer_ids else []),
        }

        path = Path(path)
        missing = self.logs[MISSING_DEFAULTS_LOG]
        lacking = missing[missing['world'].isin(worlds)]
        if self.boundary_assets is not None and not keep_outside:
            lacking = lacking[lacking['asset_id'].isin(self.boundary_assets)]
        # Every call removes the logs of another run, which this result does
        # not hold, and a lean call also the action rows and every log, all
        # before it writes anything.
        lean = not detailed and lacking.empty
        for name in (
            ('report', *LOG_NAMES)
            if lean
            else [name for name in LOG_NAMES if name not in self.logs]
        ):
            stale = path.with_name(f'{path.stem}_{name}.parquet')
            try:
                stale.unlink(missing_ok=True)
            except OSError as error:
                msg = (
                    f'{stale} cannot be removed ({error}). write removes it '
                    'first, and nothing has been written.'
                )
                raise OSError(msg) from error
        if not lacking.empty:
            self.report.save(
                path.with_name(f'{path.stem}_report.parquet'),
                worlds=worlds,
                keep_outside=keep_outside,
            )
            self.report.save_logs(path.parent, path.stem, self.logs, worlds=worlds)
            self.freeze.save_manifest(path.with_name(f'{path.stem}_manifest.json'))
            counts = lacking['property'].value_counts().to_dict()
            msg = (
                f'Compiled buildings lack {len(lacking)} default value(s) that '
                f'apply to them ({counts}), as (asset, world, property): '
                f'{[tuple(row) for row in lacking.head(10).itertuples(index=False)]}'
                f'; the log {MISSING_DEFAULTS_LOG} names them all. No GeoJSON '
                'is written.'
            )
            raise RuntimeError(msg)
        groups = self.report.exclusion_groups(
            worlds=worlds, keep_outside=keep_outside
        )
        groups = groups[groups['group'] != 'dropped']
        left_out = set(groups.loc[groups['group'] == 'every', 'asset_id'])
        if not keep_partly_excluded:
            left_out |= set(groups.loc[groups['group'] == 'some', 'asset_id'])
        excluded_in = {
            w: set(rows) for w, rows in groups.groupby('world')['asset_id']
        }
        # Every world is written from the inventory as it is, unless a building
        # to be written has an Excluded value that differs from the report's
        # exclusion rows, as after a caller adds an exclusion to the report.
        # One world, and every world after such a change, is assembled from
        # the written worlds' realizations, each building's Excluded value
        # taken from the report.
        written = {
            asset_id: asset
            for asset_id, asset in compiled.inventory.items()
            if asset_id not in left_out
        }
        as_is = world is None
        for asset_id, asset in written.items() if as_is else ():
            excluded_values = asset.features[EXCLUDED_KEY]
            listed = [asset_id in excluded_in.get(w, ()) for w in worlds]
            if (
                excluded_values
                if isinstance(excluded_values, list)
                else [excluded_values] * len(listed)
            ) != listed:
                as_is = False
                break
        if not as_is:
            assembled = AssetInventory()
            for w in worlds:
                realization = compiled.get_world_realization(w)
                for asset_id in left_out:
                    realization.inventory.pop(asset_id, None)
                excluded = excluded_in.get(w, set())
                for asset_id, asset in realization.inventory.items():
                    asset.features[EXCLUDED_KEY] = asset_id in excluded
                assembled.add_world(realization)
                del realization
            written = assembled.inventory
        inventory = _build_inventory(
            written,
            {asset_id: asset.coordinates for asset_id, asset in written.items()},
            {
                asset_id: {
                    key: value
                    for key, value in asset.features.items()
                    if (key in selected or (everything and key != EXCLUDED_KEY))
                    # In one world, None stands for a feature that only
                    # another world holds.
                    and (world is None or value is not None)
                }
                for asset_id, asset in written.items()
            },
            n_pw=len(worlds),
            vector_features=compiled.vector_features,
        )
        inventory.write_to_geojson(str(path), integer_ids=integer_ids)
        self.report.save(
            path.with_name(f'{path.stem}_report.parquet'),
            worlds=worlds,
            actions=detailed,
            keep_outside=keep_outside,
        )
        if detailed:
            self.report.save_logs(path.parent, path.stem, self.logs, worlds=worlds)
        self.freeze.save_manifest(path.with_name(f'{path.stem}_manifest.json'))


class _ImputationOutcome(NamedTuple):
    """
    What the imputation from observed donors hands to the steps after it.

    Attributes:
        imputer (ObservedDonorImputer):
            The imputer that ran, configured by the compiler's options. The
            pair re-draw draws with it, and the year draw of non-residential
            buildings (s09) and the replacement cost imputation (s13) take its
            settings.
        targets (AssetInventory):
            The imputer's targets, carrying the budget's values.
        log (pandas.DataFrame):
            The imputer's log, as ``impute_stage`` returned it.
    """

    imputer: ObservedDonorImputer
    targets: AssetInventory
    log: pd.DataFrame


class _WorldSources:
    """
    The frozen parts of each source, and the records that act on them.

    A part is registered under the label of the source it belongs to. A
    withheld record names, per asset and feature, the source a value is
    withheld from. The dropped record, named by :attr:`dropped`, holds the
    footprints removed from every source. :meth:`merge_world` builds one
    world's merged inventory from them.
    """

    def __init__(self, freeze: StageFreeze) -> None:
        """
        Start with no stage registered.

        Args:
            freeze (StageFreeze):
                The freeze the stages are frozen in and read from.
        """
        self.freeze = freeze
        self.parts: dict[str, list[str]] = {label: [] for label in PRECEDENCE}
        self.withheld: list[str] = []
        self.dropped: str | None = None

    def add_part(self, label: str, stage: str, inventory: AssetInventory) -> None:
        """
        Freeze a part and register it under its source.

        Args:
            label (str):
                The source label, one of :data:`PRECEDENCE`.
            stage (str):
                The name the part is frozen under.
            inventory (AssetInventory):
                The inventory to freeze as the part.
        """
        self.freeze.freeze(stage, inventory)
        self.parts[label].append(stage)

    def add_withheld(self, stage: str, inventory: AssetInventory) -> None:
        """
        Freeze a record of withheld values and register it.

        Args:
            stage (str):
                The name the record is frozen under.
            inventory (AssetInventory):
                One asset per asset with a withheld value, one feature per
                withheld feature, holding the label of its source.
        """
        self.freeze.freeze(stage, inventory)
        self.withheld.append(stage)

    def merge_world(self, world: int) -> AssetInventory:
        """
        Compose every registered source in one world and merge them.

        Args:
            world (int):
                The index of the world whose parts are merged, from 0.

        Returns:
            AssetInventory:
                The world's merged inventory, with one possible world and
                its source string written. A feature whose value is
                ``None`` is left out, so the merged inventory of a world
                holds the same features whatever the number of worlds.
        """
        sources = {
            label: compose_source_parts(
                self.freeze,
                stages,
                world=world,
                withheld=self.withheld,
                label=label,
                dropped=self.dropped,
            )
            for label, stages in self.parts.items()
            if stages
        }
        spine = sources.pop(SPINE_LABEL)
        precedence = [
            label for label in PRECEDENCE if label in sources or label == SPINE_LABEL
        ]
        merged = spine.merge(sources, precedence, name=SPINE_LABEL)
        for asset in merged.inventory.values():
            asset.features = {
                key: value
                for key, value in asset.features.items()
                if value is not None
            }
        return merged


def _read_occupancy_sources(inventory: AssetInventory) -> dict[str, str | None]:
    """
    Read the source of each asset's occupancy class from its source string.

    Args:
        inventory (AssetInventory):
            A merged inventory of one world, each asset with a source string.

    Returns:
        dict[str, str | None]:
            The label of the source of each asset's occupancy class, by
            asset id in the inventory's order, ``None`` where no source
            holds one.

    Raises:
        ValueError:
            If an asset's source string is not one
            :func:`~brails.types.asset_inventory.read_source_string` reads.
    """
    return {
        asset_id: SOURCE_LABELS.get(
            read_source_string(asset_id, asset.features)[
                _CODE_POSITION[OCCUPANCY_KEY]
            ]
        )
        for asset_id, asset in inventory.inventory.items()
    }


def _select_observed_donors(
    inventory: AssetInventory,
    occupancy_source: Mapping[str, str | None],
    *,
    vector_features: Sequence[str] | None = None,
) -> AssetInventory:
    """
    Select the observed donors of a merged world.

    An observed donor is a footprint whose occupancy class comes from one of
    :data:`_OCCUPANCY_SOURCE_LABELS` and that holds a story count.

    Args:
        inventory (AssetInventory):
            A merged inventory of one world.
        occupancy_source (Mapping[str, str | None]):
            The source of each asset's occupancy class in ``inventory``, as
            :func:`_read_occupancy_sources` reads it.
        vector_features (Sequence[str], optional):
            The features whose values are vectors in the result. ``None``
            declares those of ``inventory``.

    Returns:
        AssetInventory:
            The donors, the asset objects of ``inventory``, in its order.
    """
    return _select_assets(
        inventory,
        [
            asset_id
            for asset_id, asset in inventory.inventory.items()
            if occupancy_source[asset_id] in _OCCUPANCY_SOURCE_LABELS
            and not is_missing_value(asset.features.get(STORIES_KEY))
        ],
        vector_features=vector_features,
    )


def _is_plausible_pair(
    asset_id: object,  # noqa: ARG001
    world: int,  # noqa: ARG001
    features: Mapping[str, Any],
) -> bool:
    """
    Say whether an imputed occupancy class and story count may stand together.

    The compiler passes this function to
    :meth:`ObservedDonorImputer.impute_stage` as ``validate``, and the imputer
    calls it with the asset id, the world and the features once it has copied
    a donor's values. It therefore takes all three arguments and reads only
    ``features``.

    Args:
        asset_id (object):
            The id of the asset checked, unused.
        world (int):
            The index of the world in which the donor's values were copied,
            unused.
        features (Mapping[str, Any]):
            The asset's features in that world after the copy.

    Returns:
        bool:
            False where both are present and form an implausible pair,
            otherwise True.
    """
    occupancy = features.get(OCCUPANCY_KEY)
    stories = features.get(STORIES_KEY)
    if is_missing_value(occupancy) or is_missing_value(stories):
        return True
    return not is_implausible(occupancy, stories)


def _write_coded_values(
    asset_id: str, features: dict[str, Any], values: Mapping[str, Any], label: str
) -> None:
    """
    Write values from one source into a world's merged inventory, with codes.

    Args:
        asset_id (str):
            The asset's id, which an error message names.
        features (dict[str, Any]):
            The asset's features in the merged inventory, its source
            string included, changed in place.
        values (Mapping[str, Any]):
            The values to write, by feature name.
        label (str):
            The label of the source they belong to.

    Raises:
        ValueError:
            If the source string is not one that
            :func:`~brails.types.asset_inventory.read_source_string` reads.
    """
    codes = list(read_source_string(asset_id, features))
    for key, value in values.items():
        features[key] = value
        if key in _CODE_POSITION:
            codes[_CODE_POSITION[key]] = SOURCE_CODES[label]
    features[SOURCE_STRING_KEY] = ''.join(codes)


def _record_boundary(
    region: RegionBoundary | Polygon | MultiPolygon, name: str, identifier: object
) -> dict[str, object]:
    """
    Return the record of where a boundary comes from, for the report's settings.

    Args:
        region (RegionBoundary | Polygon | MultiPolygon):
            The region, as :func:`~brails.utils.region_filter.resolve_region`
            returns it.
        name (str):
            The region's name, as its boundary gives it.
        identifier (object):
            The id the region's boundary gives.

    Returns:
        dict[str, object]:
            :attr:`~brails.types.census_boundary.CensusBoundary.record` for a census
            boundary; ``provider`` ``'osm'`` with the ``name`` and ``osm_id`` for a
            name looked up in OpenStreetMap; ``provider`` ``'polygon'`` otherwise.
    """
    if isinstance(region, CensusBoundary):
        return dict(region.record)
    if getattr(region, 'input_type', None) == 'locationName':
        return {'provider': 'osm', 'name': name, 'osm_id': identifier}
    return {'provider': 'polygon'}


def _find_state(
    boundary: Polygon | MultiPolygon,
    name: str,
    record: Mapping[str, object],
    vintage: int,
    compiler: HazusEQInventoryCompiler | None,
) -> str:
    """
    Return the one state a boundary lies in, after the refusals its state calls for.

    A census boundary lies in the state its record names. Another boundary lies in
    each state that holds at least :data:`STATE_AREA_SHARE` of its area, measured in
    an equal-area projection against the TIGER/Line state polygons. A boundary that
    encloses no area, or that lies in no state or in two, is refused first. Then,
    without ``compiler``, a state the compiler cannot take is refused, and with it,
    what :meth:`HazusEQInventoryCompiler.check_state` refuses, in its order: a state
    other than the compiler's, a state it cannot take, a missing zone.

    Args:
        boundary (Polygon | MultiPolygon):
            The boundary, in EPSG:4326.
        name (str):
            The region's name, for the error messages.
        record (Mapping[str, object]):
            Where the boundary comes from, as
            :func:`_record_boundary` returns it.
        vintage (int):
            The year of the TIGER/Line release of the state file.
        compiler (HazusEQInventoryCompiler | None):
            The compiler whose state and seismic zone are checked against each
            state, or ``None`` to check only that the compiler can take it.

    Returns:
        str:
            The state's full name.

    Raises:
        ValueError:
            If the boundary encloses no area; if no state holds
            :data:`STATE_AREA_SHARE` of the area of a boundary that is not a census
            one; or for the refusals above.
    """
    if boundary.area == 0:
        msg = (
            f'The boundary of {name} encloses no area, even as repaired. Pass a '
            'polygon that encloses the place.'
        )
        raise ValueError(msg)
    if record['provider'] == 'census':
        states = [str(record['state'])]
    else:
        frame = read_tiger_file('STATE', 'us', vintage)
        shape = gpd.GeoSeries([boundary], crs=GEOGRAPHIC_CRS).to_crs(_EQUAL_AREA_CRS)
        shares = (
            frame.to_crs(_EQUAL_AREA_CRS).intersection(shape.iloc[0]).area
            / shape.iloc[0].area
        )
        states = frame.loc[shares >= STATE_AREA_SHARE, 'NAME'].tolist()
    if not states:
        msg = (
            f'Less than {STATE_AREA_SHARE:.0%} of the area of {name} lies in any '
            f'state of the TIGER/Line {vintage} state file. Check that the boundary '
            'is in longitude and latitude (EPSG:4326) and lies in the United '
            'States.'
        )
        raise ValueError(msg)
    if len(states) > 1:
        msg = (
            f'{name[0].upper()}{name[1:]} has {STATE_AREA_SHARE:.0%} or more of its '
            f'area in each of {" and ".join(states)}, and a compilation covers one '
            'state. Pass the part of the boundary that lies in one state.'
        )
        raise ValueError(msg)
    (state,) = states
    if compiler is None:
        _check_state(state, DEFAULT_SEISMIC_ZONE, DEFAULT_DESIGN_LEVEL_METHOD, 0)
    else:
        compiler.check_state(state)
    return state


def _widen(geometry: BaseGeometry, meters: float) -> BaseGeometry:
    """
    Return a geometry in EPSG:4326 widened by a distance, in its UTM zone.

    Args:
        geometry (BaseGeometry):
            The geometry, in longitude and latitude.
        meters (float):
            How far to widen the geometry, in meters.

    Returns:
        BaseGeometry:
            The widened geometry, in longitude and latitude.
    """
    series = gpd.GeoSeries([geometry], crs=GEOGRAPHIC_CRS)
    widened = series.to_crs(series.estimate_utm_crs()).buffer(meters)
    return widened.to_crs(GEOGRAPHIC_CRS).iloc[0]


def _check_state(
    state: str, seismic_zone: str, design_level_method: str, seed: int
) -> None:
    """
    Raise if the inferers a compilation builds would refuse the state or zone.

    The inferers are built only for the check. Each world builds its own.

    Args:
        state (str):
            The full state name the compilation applies.
        seismic_zone (str):
            The 1997 UBC seismic zone it applies.
        design_level_method (str):
            The design level method of :class:`HazusEQDesignLevelInferer`.
        seed (int):
            The seed the structure type inferer is built with.

    Raises:
        ValueError:
            If :class:`HazusEQStructureTypeInferer` refuses the state with its
            completed tables, or :class:`HazusEQDesignLevelInferer` the state, the
            zone or the method.
        NotImplementedError:
            If the design level method is not implemented.
    """
    try:
        HazusEQStructureTypeInferer(state, seed=seed, complete_tables=True)
    except ValueError as error:
        group = get_hazus_state_region_mapping().get(state, {}).get('RegionGroup')
        msg = (
            'The compiler draws structure types from the completed West Coast '
            'tables, which cover the states of the Census West region only, '
            f'not {state}.'
            if group not in (None, WEST_COAST)
            else f'The compiler cannot draw structure types for {state}: {error}'
        )
        raise ValueError(msg) from error
    HazusEQDesignLevelInferer(design_level_method, state, seismic_zone)


class HazusEQInventoryCompiler:
    """
    Compile a Hazus earthquake inventory of one study area over several worlds.

    The section "Hazus earthquake inventory compiler" of the README beside
    this module describes the run, its frozen stages, its worlds and the
    written file.

    Every constructor argument is stored as an attribute of the same name,
    read-only by convention; ``seed`` holds the seed used, never ``None``,
    and the options hold the mappings as resolved. ``state``, ``seismic_zone`` and
    ``merge_stacked`` keep ``None`` when left at it. Each compilation resolves them
    and records the values applied in its summary's settings.
    """

    def __init__(  # noqa: C901
        self,
        *,
        seed: int | None = None,
        n_worlds: int = DEFAULT_N_WORLDS,
        state: str | None = None,
        seismic_zone: str | None = None,
        design_level_method: str = DEFAULT_DESIGN_LEVEL_METHOD,
        join_cap_m: float = DEFAULT_JOIN_CAP_M,
        projected_crs: object = None,
        imputer_options: Mapping[str, object] | None = None,
        budget_options: Mapping[str, object] | None = None,
        sampler_options: Mapping[str, object] | None = None,
        merge_stacked: bool | None = None,
        acs_vintage: int = DEFAULT_ACS_VINTAGE,
        block_group_vintage: int = DEFAULT_TIGER_VINTAGE,
    ) -> None:
        """
        Store the compiler's settings and refuse any that a component would refuse.

        The constructor builds the structure type inferer, the design level
        inferer, the NSI join and the observed-donor imputer from the
        settings so that each can refuse a bad setting. It keeps none of
        them, since :meth:`compile` builds its own. A bad setting therefore
        raises here, before any data is read, and not partway through a
        compilation. The constructor also loads the input schema of
        :data:`DLML_DATASET`, against which every compilation checks its
        buildings.

        Args:
            seed (int, optional):
                The seed of every component. ``None`` draws one from system
                entropy.
            n_worlds (int, optional):
                The number of possible worlds :meth:`compile` computes.
                Defaults to :data:`DEFAULT_N_WORLDS`.
            state (str, optional):
                The full state name, as in ``'California'``, which every inferer
                that takes a state receives. Structure types are drawn from the
                completed West Coast tables, so the state must be one of that
                region with a RES1 distribution. ``None``, the default, applies the
                state of :class:`CompilerInputs` and :data:`DEFAULT_STATE` where the
                inputs name none.
            seismic_zone (str, optional):
                The 1997 UBC seismic zone, which every inferer that takes one
                receives. ``None``, the default, applies
                :data:`DEFAULT_SEISMIC_ZONE`, which :meth:`compile` refuses for
                inputs that name a state other than :data:`DEFAULT_STATE`.
            design_level_method (str, optional):
                The design level method of
                :class:`HazusEQDesignLevelInferer`.
            join_cap_m (float, optional):
                The distance cap of the NSI join in meters.
            projected_crs (object, optional):
                The projected CRS of the join's distances, as
                :class:`NSIFootprintJoin` takes it. ``None`` estimates a UTM
                zone from the footprints.
            imputer_options (Mapping[str, object], optional):
                Keyword arguments of :class:`ObservedDonorImputer`, each
                overriding :data:`DEFAULT_IMPUTER_OPTIONS`. The compiler
                passes the donors directly and sets the seed, so
                ``n_batches`` and ``seed`` are refused.
            budget_options (Mapping[str, object], optional):
                Keyword arguments of :class:`ACSUnitBudget` other than its
                table, mobile-home park points, imputer and seed, which the
                compiler passes.
            sampler_options (Mapping[str, object], optional):
                Keyword arguments of :class:`ACSYearBuiltSampler` other than
                its tables, seed and world, which the compiler passes.
            merge_stacked (bool, optional):
                Whether s01 first merges the stacked footprints of one
                building with :class:`StackedFootprintMerge`, in the
                footprints' height unit, which re-keys them by geohash. The
                footprints must then all be Overture footprints with an
                ``overture_id``, and no local inventory is accepted. ``None``, the
                default, applies the ``merge_stacked`` of :class:`CompilerInputs`.
            acs_vintage (int, optional):
                The last year of the ACS five-year release from which :meth:`compile`
                reads each census table that :class:`CompilerInputs` gives as
                ``None``. Defaults to
                :data:`~brails.scrapers.us_census_scrapers.acs_table_scraper.DEFAULT_ACS_VINTAGE`.
            block_group_vintage (int, optional):
                The year of the TIGER/Line release from which :meth:`compile` reads
                the block groups when :class:`CompilerInputs` gives none. Where the
                census tables are read too, it must be of the decade of
                ``acs_vintage``. Defaults to
                :data:`~brails.utils.tiger_files.DEFAULT_TIGER_VINTAGE`.

        Raises:
            TypeError:
                If ``seed``, ``n_worlds``, ``acs_vintage`` or ``block_group_vintage``
                is not an integer, ``merge_stacked`` is neither a boolean nor
                ``None``, an options argument is not a mapping, or an options
                mapping names an argument its class does not take or that the
                compiler sets.
            ValueError:
                If ``n_worlds`` is below 1; if
                :class:`HazusEQStructureTypeInferer` refuses the state with
                its completed tables (a state outside the West Coast region,
                or one without a RES1 distribution); if the zone or the method
                is refused by :class:`HazusEQDesignLevelInferer`; if
                ``imputer_options`` holds ``n_batches``; or if the join or
                the imputer refuses its settings.
            NotImplementedError:
                If the design level method is not implemented.
        """
        if seed is None:
            seed = secrets.randbits(64)
        elif isinstance(seed, bool):
            msg = f'seed must be an integer or None, not {seed!r}.'
            raise TypeError(msg)
        if isinstance(n_worlds, bool):
            msg = f'n_worlds must be an integer, not {n_worlds!r}.'
            raise TypeError(msg)
        self.seed = operator.index(seed)
        self.n_worlds = operator.index(n_worlds)
        if self.n_worlds < 1:
            msg = f'n_worlds must be at least 1, not {self.n_worlds}.'
            raise ValueError(msg)
        if merge_stacked is not None and not isinstance(merge_stacked, bool):
            msg = f'merge_stacked must be a boolean or None, not {merge_stacked!r}.'
            raise TypeError(msg)
        self.merge_stacked = merge_stacked
        for name, vintage in (
            ('acs_vintage', acs_vintage),
            ('block_group_vintage', block_group_vintage),
        ):
            if isinstance(vintage, bool) or not isinstance(vintage, int):
                msg = f'{name} must be an integer year, not {vintage!r}.'
                raise TypeError(msg)
        self.acs_vintage = acs_vintage
        self.block_group_vintage = block_group_vintage

        # A compiler without a state is checked on DEFAULT_STATE, and compile
        # checks another state the inputs name.
        _check_state(
            state or DEFAULT_STATE,
            seismic_zone or DEFAULT_SEISMIC_ZONE,
            design_level_method,
            self.seed,
        )
        self.state = state
        self.seismic_zone = seismic_zone
        self.design_level_method = design_level_method

        # Built only to check the cap and the CRS. The run builds its own join.
        NSIFootprintJoin(
            cap_m=join_cap_m, seed=self.seed, projected_crs=projected_crs
        )
        self.join_cap_m = join_cap_m
        self.projected_crs = projected_crs

        if 'n_batches' in (imputer_options or {}):
            msg = (
                'imputer_options must not hold n_batches: the compiler passes '
                'the donors directly and runs the imputer without batch '
                'feedback.'
            )
            raise ValueError(msg)
        resolved = {}
        for name, given, target in (
            ('imputer_options', imputer_options, ObservedDonorImputer),
            ('budget_options', budget_options, ACSUnitBudget),
            ('sampler_options', sampler_options, ACSYearBuiltSampler),
        ):
            if given is not None and not isinstance(given, Mapping):
                msg = f'{name} must be a mapping, not {type(given).__name__}.'
                raise TypeError(msg)
            options = dict(given or {})
            reserved = [key for key in options if key in _RESERVED_OPTIONS[name]]
            accepted = inspect.signature(target).parameters
            unknown = [key for key in options if key not in accepted]
            if reserved or unknown:
                msg = (
                    f'{name} may hold only arguments of {target.__name__} the '
                    f'compiler does not set: {sorted(reserved + unknown)} are '
                    'not among them.'
                )
                raise TypeError(msg)
            resolved[name] = options
        resolved['imputer_options'] = {
            **DEFAULT_IMPUTER_OPTIONS,
            **resolved['imputer_options'],
        }
        # Built only to check the options. The run builds its own imputer.
        ObservedDonorImputer(**resolved['imputer_options'], seed=self.seed)
        # Built here so that a missing schema stops the compiler before a run.
        self._validator = SchemaValidator(DLML_DATASET)
        self.imputer_options = MappingProxyType(resolved['imputer_options'])
        self.budget_options = MappingProxyType(resolved['budget_options'])
        # The distance from the footprints within which park points are read.
        self._mh_park_radius_m = self.budget_options.get(
            'mh_park_radius_m', DEFAULT_MH_PARK_RADIUS_M
        )
        self.sampler_options = MappingProxyType(resolved['sampler_options'])

    def compile_place(
        self,
        place: str | RegionBoundary | Polygon | MultiPolygon,
        **from_place_options: object,
    ) -> CompilationResult:
        """
        Compile a place from its name or boundary.

        The call is ``compile(CompilerInputs.from_place(place, compiler=self,
        **from_place_options))``, so a state or zone the compiler would refuse stops
        it before the footprints are read. ``buffer_m`` defaults to the imputer's
        ``cap_m``. The Census key is checked by both calls. No inputs are kept, so a
        compilation that fails loses the downloads; call
        :meth:`CompilerInputs.from_place` and :meth:`compile` apart to keep them.
        It raises what those two raise.

        Args:
            place (str | RegionBoundary | Polygon | MultiPolygon):
                The place, as :meth:`CompilerInputs.from_place` takes it.
            **from_place_options:
                The keyword arguments of :meth:`CompilerInputs.from_place` other
                than ``compiler``.

        Returns:
            CompilationResult:
                The compiled inventory of the place, with the report, frozen stages
                and logs of the run.
        """
        from_place_options.setdefault(
            'buffer_m', self.imputer_options.get('cap_m', DEFAULT_DONOR_CAP_M)
        )
        return self.compile(
            CompilerInputs.from_place(place, compiler=self, **from_place_options)
        )

    def compile(self, inputs: CompilerInputs) -> CompilationResult:
        """
        Compile the study area's building inventory over :attr:`n_worlds` worlds.

        The inputs are not modified.

        Args:
            inputs (CompilerInputs):
                The study area's data.

        Returns:
            CompilationResult:
                The compiled inventory, with the report, frozen stages and
                logs of the run.

        Raises:
            TypeError:
                If a footprint's asset id is not a string, the boundary is not a
                Shapely Polygon or MultiPolygon, or the notices are not a tuple of
                strings.
            ValueError:
                If the local inventory holds an asset that the footprints
                lack, a feature of :data:`RESERVED_LOCAL_FEATURES` or more
                than one possible world, or is given with ``merge_stacked``;
                if no footprint lies inside the boundary, found right after
                ``s01_footprints`` and before the census tables are read;
                if the inputs name a state other than the compiler's, or other than
                :data:`DEFAULT_STATE` for a compiler built without a seismic zone,
                or one the inferers refuse with the compiler's seismic zone;
                or as the components raise, among them a footprint outside
                every block group and, with ``merge_stacked``, a footprint
                that is not Overture's; or, for a census table given as ``None``, if
                a key store holds a value that cannot be a key or a table the year
                sampler reads holds an estimate the Census Bureau does not publish
                where the run reads it; or, where both the census tables and the
                block groups are read, if ``acs_vintage`` and ``block_group_vintage``
                are of different decades.
            LookupError:
                If a census table is given as ``None`` and no Census API key is
                found.
            RuntimeError:
                If a frozen stage changed during the run; or, for a census table
                given as ``None``, if the Census Data API refuses the key or answers
                a request with something other than the table asked; or, for park
                points given as ``None``, if the HIFLD archive shows no complete
                write.
            requests.exceptions.RequestException:
                If the key check or a request for a census table fails, as
                :meth:`~brails.scrapers.ACSTableScraper.get_table` raises, or the
                download of a TIGER/Line file of block groups or of the park points
                fails.
        """
        state, seismic_zone, merge_stacked, notices = self._check_inputs(inputs)

        n_computed = self.n_worlds
        settings = {
            'study_area': inputs.study_area,
            'footprint_release': inputs.footprint_release,
            'height_unit': inputs.height_unit,
            'n_worlds': self.n_worlds,
            'worlds': list(range(n_computed)),
            'state': state,
            'seismic_zone': seismic_zone,
            'design_level_method': self.design_level_method,
            'join_cap_m': self.join_cap_m,
            'projected_crs': None
            if self.projected_crs is None
            else str(self.projected_crs),
            'imputer_options': dict(self.imputer_options),
            'budget_options': dict(self.budget_options),
            'sampler_options': dict(self.sampler_options),
            'merge_stacked': merge_stacked,
        }
        report = CompilationReport(
            self.seed,
            stages=REPORT_STAGES,
            exclusion_categories=EXCLUSION_CATEGORIES,
            settings=settings,
        )
        report.set_world_count(n_computed)
        for notice in notices:
            report.note(notice)
        freeze = StageFreeze()
        sources = _WorldSources(freeze)
        logs: dict[str, pd.DataFrame] = {}

        inputs, acs_scraper = self._read_inputs_before_s01(inputs, report)
        started = time.perf_counter()
        centroids = self._prepare_footprints(
            inputs, sources, logs, report, merge_stacked=merge_stacked
        )
        boundary_assets = self._select_inside_boundary(inputs, freeze, report)
        report.set_timing(STAGE_S01_FOOTPRINTS, time.perf_counter() - started)
        inputs = self._read_census_tables(inputs, acs_scraper, freeze, report)

        started = time.perf_counter()
        self._join_nsi_to_footprints(inputs, sources, logs, centroids)
        report.set_timing('s02_nsi_join', time.perf_counter() - started)

        started = time.perf_counter()
        stories = self._infer_story_counts(
            inputs, sources, logs, centroids, report, n_computed
        )
        report.set_timing('s03_story_counts', time.perf_counter() - started)

        # The unit budget and the imputation read world 0 as s03 leaves it,
        # with the source of each footprint's occupancy class, and the budget
        # is given the imputer that s05 then runs on the budget's targets.
        started = time.perf_counter()
        merged = sources.merge_world(0)
        occupancy_source = _read_occupancy_sources(merged)
        imputer = ObservedDonorImputer(**self.imputer_options, seed=self.seed)
        allocation = self._allocate_units(
            inputs,
            sources,
            logs,
            centroids,
            report,
            n_computed,
            merged,
            occupancy_source,
            imputer,
        )
        report.set_timing('s04_unit_budget', time.perf_counter() - started)
        started = time.perf_counter()
        imputation = self._impute_from_donors(
            sources,
            logs,
            centroids,
            n_computed,
            merged,
            occupancy_source,
            imputer,
            allocation,
        )
        report.set_timing('s05_imputation', time.perf_counter() - started)
        del merged, allocation
        self._freeze_view(STAGE_S05_VIEW, sources, report, n_computed)
        sources.freeze.freeze(
            STAGE_S05_DONORS,
            imputation_donors_from_log(
                imputation.log, imputation.targets, n_worlds=n_computed
            ),
        )
        imputer = imputation.imputer
        del imputation

        self._draw_residential_years(
            inputs, sources, logs, centroids, report, n_computed
        )
        self._redraw_implausible_pairs(
            sources, logs, centroids, report, n_computed, imputer, stories
        )
        self._assign_structure_in_each_world(
            sources,
            logs,
            centroids,
            report,
            n_computed,
            state=state,
            seismic_zone=seismic_zone,
        )
        self._freeze_view(STAGE_S08_VIEW, sources, report, n_computed)
        self._impute_nonresidential_years(
            sources,
            logs,
            centroids,
            report,
            n_computed,
            imputer,
            state=state,
            seismic_zone=seismic_zone,
        )

        self._withhold_replaced_values(sources, centroids)

        # Steps s11 to s15 run world by world on the world's merged inventory,
        # which s15 completes, and each world's results are added to those of
        # the earlier worlds.
        excluded_in = {
            world: set(ids)
            for world, ids in report.exclusions.groupby('world')['asset_id']
        }
        areas, costs, letters, inventory = (
            AssetInventory(),
            AssetInventory(),
            AssetInventory(),
            AssetInventory(),
        )
        violations, area_rows, value_logs, lacking = [], [], [], []
        for world in range(n_computed):
            started = time.perf_counter()
            merged = sources.merge_world(world)
            report.set_timing(
                f'{STAGE_S15_OUTPUT}_merge',
                time.perf_counter() - started,
                world=world,
            )
            excluded = set(excluded_in.get(world, ()))
            violations.append(self._check_schema(world, merged, excluded, report))
            world_areas = self._compute_gross_areas(
                world, sources, centroids, area_rows, report
            )
            world_costs, value_log = self._impute_replacement_costs(
                world, sources, centroids, imputer, excluded, world_areas, report
            )
            world_letters = self._derive_building_types(
                world, merged, sources, centroids, excluded, report
            )
            lacking.extend(
                self._assemble_output(
                    world,
                    merged,
                    excluded,
                    world_areas,
                    world_costs,
                    world_letters,
                    report,
                )
            )
            areas.add_world(world_areas)
            costs.add_world(world_costs)
            letters.add_world(world_letters)
            inventory.add_world(merged)
            value_logs.append(value_log)
            del merged

        # s11: the schema's violations and the count of each world's refusals.
        logs['schema_check'] = pd.concat(violations, ignore_index=True)
        report.set_schema_check(
            {'dataset': DLML_DATASET}, refused=logs['schema_check']
        )
        # s12: the source of each area, in every world.
        logs['gross_area'] = pd.DataFrame(
            area_rows, columns=['asset_id', 'world', 'area_source']
        ).astype({'asset_id': object, 'area_source': 'category'})
        # s13: the value imputer's log of every world.
        logs['value'] = pd.concat(value_logs, ignore_index=True)
        # s15: the default properties compiled buildings lack.
        logs[MISSING_DEFAULTS_LOG] = pd.DataFrame(
            lacking, columns=['asset_id', 'world', 'property'], dtype=object
        )
        if lacking:
            of_inside = (
                ''
                if boundary_assets is None
                else f', {sum(a in boundary_assets for a, _, _ in lacking)} of them '
                'of buildings inside the boundary'
            )
            report.note(
                f'{len(lacking)} default value(s) that apply to compiled '
                f'buildings are missing{of_inside}; the log {MISSING_DEFAULTS_LOG} '
                'names them, and write refuses '
                + (
                    'the file.'
                    if boundary_assets is None
                    else 'a file that holds them.'
                )
            )
        # The stages of s12 to s15, over the worlds.
        for stage, assembled in (
            (STAGE_S12_GROSS_AREA, areas),
            (STAGE_S13_REPLACEMENT_COST, costs),
            (STAGE_S14_BUILDING_TYPE, letters),
            (STAGE_S15_OUTPUT, inventory),
        ):
            started = time.perf_counter()
            freeze.freeze(stage, assembled)
            report.set_timing(f'{stage}_freeze', time.perf_counter() - started)

        started = time.perf_counter()
        manifest = freeze.manifest()
        report.set_timing('manifest', time.perf_counter() - started)
        report.set_manifest_stages(entry['stage'] for entry in manifest['stages'])
        inside = inventory
        if boundary_assets is not None:
            inside = _select_assets(
                inventory,
                [a for a in inventory.inventory if a in boundary_assets],
                n_pw=inventory.n_pw,
            )
        return CompilationResult(
            inventory=inside,
            report=report,
            freeze=freeze,
            manifest=manifest,
            logs=MappingProxyType(logs),
            seed=self.seed,
            area_inventory=inventory,
            boundary_assets=boundary_assets,
        )

    def _check_inputs(
        self, inputs: CompilerInputs
    ) -> tuple[str, str, bool, list[str]]:
        """
        Refuse inputs the compiler cannot compile, and resolve its settings for them.

        It runs before anything is read.

        Args:
            inputs (CompilerInputs):
                The study area's data.

        Returns:
            tuple[str, str, bool, list[str]]:
                The state, the seismic zone and ``merge_stacked`` the compilation
                applies, and the notices for the report: those of the inputs, and one
                naming the zone where the inputs name :data:`DEFAULT_STATE` and the
                compiler was built without a zone.

        Raises:
            TypeError:
                If a footprint's asset id is not a string, the boundary is not a
                Shapely Polygon or MultiPolygon, or the notices are not a tuple of
                strings.
            ValueError:
                If the local inventory holds an asset that the footprints lack, a
                feature of :data:`RESERVED_LOCAL_FEATURES` or more than one possible
                world, or is given with ``merge_stacked``; or as
                :meth:`check_state` raises.
        """
        not_strings = [
            a for a in inputs.footprints.inventory if not isinstance(a, str)
        ]
        if not_strings:
            msg = f'Footprint asset ids must be strings, not {not_strings[:10]!r}.'
            raise TypeError(msg)
        if not isinstance(inputs.boundary, (Polygon, MultiPolygon, type(None))):
            msg = (
                'The boundary must be a Shapely Polygon or MultiPolygon, not '
                f'{type(inputs.boundary).__name__}.'
            )
            raise TypeError(msg)
        if not isinstance(inputs.notices, tuple) or not all(
            isinstance(notice, str) for notice in inputs.notices
        ):
            msg = (
                'CompilerInputs.notices must be a tuple of strings, not '
                f'{inputs.notices!r}.'
            )
            raise TypeError(msg)
        state, seismic_zone = self.check_state(inputs.state)
        merge_stacked = (
            inputs.merge_stacked
            if self.merge_stacked is None
            else self.merge_stacked
        )
        if inputs.local is not None:
            self._check_local(inputs, merge_stacked=merge_stacked)
        notices = list(inputs.notices)
        if inputs.state == DEFAULT_STATE and self.seismic_zone is None:
            notices.append(
                f'The inputs name {DEFAULT_STATE} and the compiler was built '
                f'without seismic_zone, so zone {DEFAULT_SEISMIC_ZONE} applies.'
            )
        return state, seismic_zone, merge_stacked, notices

    @staticmethod
    def _check_local(inputs: CompilerInputs, *, merge_stacked: bool) -> None:
        """
        Refuse a local inventory the compiler cannot take.

        Args:
            inputs (CompilerInputs):
                The study area's data, with a local inventory.
            merge_stacked (bool):
                Whether the compilation merges stacked footprints.

        Raises:
            ValueError:
                If the local inventory holds an asset that the footprints lack, a
                feature of :data:`RESERVED_LOCAL_FEATURES` or more than one possible
                world, or is given with ``merge_stacked``.
        """
        if merge_stacked:
            msg = (
                'A local inventory cannot be given with merge_stacked: its '
                'ids are those of the footprints, which the merge re-keys.'
            )
            raise ValueError(msg)
        stray = sorted(
            set(inputs.local.inventory) - set(inputs.footprints.inventory),
            key=str,
        )
        if stray:
            msg = (
                f'The local inventory holds {len(stray)} asset(s) the '
                f'footprints lack: {stray[:10]!r}.'
            )
            raise ValueError(msg)
        if inputs.local.n_pw != 1:
            msg = (
                'The local inventory must hold one possible world, not '
                f'{inputs.local.n_pw}.'
            )
            raise ValueError(msg)
        reserved = sorted(
            {
                key
                for asset in inputs.local.inventory.values()
                for key in asset.features
            }
            & set(RESERVED_LOCAL_FEATURES)
        )
        if reserved:
            msg = (
                f'The local inventory holds {reserved}, of '
                f'RESERVED_LOCAL_FEATURES {list(RESERVED_LOCAL_FEATURES)}, '
                'which the compiler writes for itself and reads back.'
            )
            raise ValueError(msg)

    def check_state(self, named: str | None) -> tuple[str, str]:
        """
        Return the state and the seismic zone a compilation of a study area applies.

        The compiler's state applies where it was given one, the state the study
        area is in where it was not, and :data:`DEFAULT_STATE` where neither is
        named. The compiler's zone applies where it was given one, and
        :data:`DEFAULT_SEISMIC_ZONE` otherwise. :meth:`compile` calls it for the
        state its inputs name, and :meth:`CompilerInputs.from_place` for the state
        of the place when given the compiler.

        Args:
            named (str | None):
                The state the study area is in, such as the inputs name, or
                ``None``.

        Returns:
            tuple[str, str]:
                The full state name and the 1997 UBC seismic zone.

        Raises:
            ValueError:
                In this order: if the compiler's state and ``named`` differ; for a
                compiler built without a state, if the inferers refuse ``named``
                with the zone; and if ``named`` is a state other than
                :data:`DEFAULT_STATE` and the compiler was built without a zone.
        """
        if self.state is not None and named not in (None, self.state):
            msg = (
                f'The compiler was built for {self.state} and the study area is in '
                f'{named}. Build the compiler without a state, and it takes the '
                'state of the study area.'
            )
            raise ValueError(msg)
        state = self.state or named or DEFAULT_STATE
        seismic_zone = self.seismic_zone or DEFAULT_SEISMIC_ZONE
        # The constructor checked the compiler's state, or DEFAULT_STATE without one.
        if state != (self.state or DEFAULT_STATE):
            _check_state(state, seismic_zone, self.design_level_method, self.seed)
        if named not in (None, DEFAULT_STATE) and self.seismic_zone is None:
            zones = STATE_DESIGN_LEVELS.get(named, GENERIC_DESIGN_LEVELS)
            msg = (
                f'The study area is in {named}, and the compiler was built without '
                'seismic_zone. Build it with seismic_zone, the 1997 UBC zone of the '
                f'study area, one of {", ".join(zones)}; zone '
                f'{DEFAULT_SEISMIC_ZONE} applies without one only to inputs that '
                f'name {DEFAULT_STATE} or no state.'
            )
            raise ValueError(msg)
        return state, seismic_zone

    def _prepare_footprints(
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        report: CompilationReport,
        *,
        merge_stacked: bool,
    ) -> dict[str, list[list[float]]]:
        """
        Run s01: freeze the footprints with their block groups and plan areas.

        With ``merge_stacked``, the stacked footprints are merged first,
        and the rest of the step reads the merged set.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            sources (_WorldSources):
                The sources, to which the footprints and a local inventory
                are added as :data:`STAGE_S01_FOOTPRINTS` and
                :data:`STAGE_S01_LOCAL`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``footprint_ties`` and, with
                ``merge_stacked``, the merge's parts table as
                ``stacked_parts`` and its id log as ``stacked_asset_ids``.
            report (CompilationReport):
                The report, whose notices count the merges.
            merge_stacked (bool):
                Whether to merge the stacked footprints, as the compilation
                resolved :attr:`merge_stacked`.

        Returns:
            dict[str, list[list[float]]]:
                Each footprint's centroid as coordinates, the exterior
                rings' centroid of
                :meth:`AssetInventory.centroid_and_plan_area`.

        Raises:
            ValueError:
                If a footprint lies outside every block group or encloses no
                area, or as :meth:`StackedFootprintMerge.merge` raises.
        """
        footprints = inputs.footprints
        if merge_stacked:
            merge = StackedFootprintMerge(length_unit=inputs.height_unit)
            merged = merge.merge(footprints)
            footprints = merged.inventory
            logs['stacked_parts'] = merged.parts
            # The keys of an assign_asset_ids log entry, so that the log
            # keeps its columns when nothing collides.
            logs['stacked_asset_ids'] = pd.DataFrame(
                merged.id_log,
                columns=['position', 'geohash', 'area_m2', 'rank', 'id', 'reason'],
            )
            # The drawing kept of a set of duplicates is a head too, so the
            # rule tells it from the head of a stack.
            rule = merged.parts['rule']
            head = merged.parts['is_head']
            report.note(
                'Stacked footprints: '
                f'{int(((rule == RULE_STACKED) & ~head).sum())} footprint(s) '
                f'merged into {int(((rule == RULE_STACKED) & head).sum())} '
                f'asset(s); {int(((rule == RULE_COPY) & ~head).sum())} '
                'duplicate drawing(s) dropped.'
            )
        spine, ties = CensusTractScraper().get_census_geographies(
            footprints,
            levels=(LEVEL_BLOCK_GROUP,),
            polygons=inputs.block_groups,
        )
        logs['footprint_ties'] = ties
        release = inputs.footprint_release
        centroids = {}
        for asset_id, asset in spine.inventory.items():
            lon, lat, area_m2 = AssetInventory.centroid_and_plan_area(
                asset.coordinates
            )
            # The parts share these lists; nothing in the run changes them.
            centroids[asset_id] = [[lon, lat]]
            features = asset.features
            datasets = sorted(
                {
                    entry['dataset']
                    for entry in parse_sources(features.get(SOURCES_KEY))
                    if isinstance(entry.get('dataset'), str) and entry['dataset']
                }
            )
            overture = features.get(FOOTPRINT_PROVIDER_KEY) == OVERTURE_PROVIDER
            if overture and release is not None:
                features[FOOTPRINT_RELEASE_KEY] = f'{OVERTURE_PROVIDER}_{release}'
            if datasets:
                features[FOOTPRINT_DATASETS_KEY] = ';'.join(datasets)
            features[PLAN_AREA_KEY] = area_m2 / UNIT_TO_METER['ft'] ** 2
        sources.add_part(SPINE_LABEL, STAGE_S01_FOOTPRINTS, spine)
        if inputs.local is not None:
            local = deepcopy(inputs.local)
            for asset_id, asset in local.inventory.items():
                asset.coordinates = centroids[asset_id]
                # The local GrossArea also under a name of its own, which the
                # output keeps where the GrossArea of s12 takes its place.
                if not is_missing_value(asset.features.get(GROSS_AREA_KEY)):
                    asset.features[LOCAL_GROSS_AREA_KEY] = asset.features[
                        GROSS_AREA_KEY
                    ]
            sources.add_part('local', STAGE_S01_LOCAL, local)
        return centroids

    def _read_inputs_before_s01(
        self, inputs: CompilerInputs, report: CompilationReport
    ) -> tuple[CompilerInputs, ACSTableScraper | None]:
        """
        Read the park points and block groups left as ``None``, and check the key.

        All run before s01, so that a run without a usable key or without the files
        stops before any step. The census tables left as ``None`` are read after s01,
        by :meth:`_read_census_tables`.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            report (CompilationReport):
                The report, whose settings record the release of each input read and
                the write of the HIFLD archive, whose notices count the park points
                and name the states of the block groups, and which times the two
                reads as ``s01_mh_park_points`` and ``s01_block_groups``.

        Returns:
            tuple[CompilerInputs, ACSTableScraper | None]:
                A copy of the inputs holding the inputs read, and the scraper holding
                the checked key, or ``None`` when every table is given.

        Raises:
            ValueError:
                If both the tables and the block groups are read and their vintages
                are of different decades, or as :class:`ACSTableScraper` and
                :meth:`TIGERLineScraper.get_block_groups` raise.
            LookupError, RuntimeError, ValueError:
                If no key is found, a store holds a value that cannot be one or the
                API refuses it, with a sentence naming the tables that need it; or,
                for a RuntimeError, if the HIFLD archive shows no complete write.
            requests.exceptions.RequestException:
                If the key check fails, or the download of the park points or of a
                TIGER/Line file, with a sentence naming the input that was being
                read.
        """
        missing = [
            name for name in _ACS_TABLE_REQUESTS if getattr(inputs, name) is None
        ]
        # A GEOID can carry over from one decade to the next with a changed boundary,
        # so tables and polygons of two decades would be joined without an error.
        if (
            missing
            and inputs.block_groups is None
            and self.acs_vintage // 10 != self.block_group_vintage // 10
        ):
            decade = self.acs_vintage // 10 * 10
            msg = (
                f'acs_vintage {self.acs_vintage} and block_group_vintage '
                f'{self.block_group_vintage} are of different decades. The tables '
                "of an ACS release count the block groups of its decade's census, "
                'so set block_group_vintage to a TIGER/Line release from '
                f'{decade} to {decade + 9}.'
            )
            raise ValueError(msg)
        scraper = None
        if missing:
            try:
                scraper = ACSTableScraper()
                scraper.check_key(self.acs_vintage)
            except (LookupError, RuntimeError, ValueError) as error:
                msg = (
                    f'{error} CompilerInputs leaves {", ".join(missing)} as None, '
                    'so the compiler reads them from the Census Data API, which '
                    'needs a key. Give the tables in CompilerInputs to compile '
                    'without one.'
                )
                raise type(error)(msg) from error
            report.settings['acs_vintage'] = self.acs_vintage

        if inputs.mh_park_points is None:
            started = time.perf_counter()
            radius = self._mh_park_radius_m
            print(
                f'Reading the mobile-home park points within {radius:g} m of the '
                "footprints' bounding box from the HIFLD archive."
            )
            area = _widen(inputs.footprints.get_extent(buffer='none'), radius)
            parks = MobileHomeParkScraper()
            try:
                points = parks.get_assets(GeometryRegion(area))
            except (requests.RequestException, RuntimeError) as error:
                msg = (
                    'Reading the mobile-home park points from the HIFLD archive '
                    f'failed: {error} To compile without the download, give the '
                    'points as CompilerInputs.mh_park_points.'
                )
                raise type(error)(msg) from error
            inputs = inputs._replace(mh_park_points=points)
            report.settings['mh_park_write'] = {
                'marker': parks.marker,
                'written': parks.written,
            }
            report.note(
                f'{len(points.inventory)} mobile-home park point(s) read from the '
                f"HIFLD archive within {radius:g} m of the footprints' bounding "
                f'box, from its write {parks.marker} of {parks.written}.'
            )
            report.set_timing('s01_mh_park_points', time.perf_counter() - started)

        if inputs.block_groups is None:
            started = time.perf_counter()
            vintage = self.block_group_vintage
            area = GeometryRegion(inputs.footprints.get_extent(buffer='none'))
            try:
                block_groups = TIGERLineScraper(vintage).get_block_groups(area)
            except requests.RequestException as error:
                msg = (
                    f'Reading the block groups from the TIGER/Line {vintage} files '
                    f'failed: {error} To compile without the download, give the '
                    'polygons as CompilerInputs.block_groups.'
                )
                raise type(error)(msg) from error
            inputs = inputs._replace(block_groups=block_groups)
            report.settings['block_group_vintage'] = vintage
            report.note(
                f'Block groups read from the TIGER/Line {vintage} files of '
                f'{", ".join(sorted(set(block_groups[STATE_NAME_COLUMN])))} for the '
                "footprints' bounding box."
            )
            report.set_timing('s01_block_groups', time.perf_counter() - started)
        return inputs, scraper

    def _select_inside_boundary(
        self, inputs: CompilerInputs, freeze: StageFreeze, report: CompilationReport
    ) -> frozenset[str] | None:
        """
        Find the footprints of ``s01_footprints`` that lie inside the boundary.

        A footprint is inside by the rule of
        :func:`~brails.utils.region_filter.find_inside`, which
        :func:`~brails.utils.region_filter.filter_to_region` applies too.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            freeze (StageFreeze):
                The freeze holding :data:`STAGE_S01_FOOTPRINTS`.
            report (CompilationReport):
                The report, which receives the footprints inside the boundary, the
                boundary among its settings, and a notice of an area narrower than
                the donor cap.

        Returns:
            frozenset[str] | None:
                The ids of the footprints inside the boundary, or ``None`` where the
                inputs give no boundary.

        Raises:
            ValueError:
                If no footprint lies inside the boundary.
        """
        if inputs.boundary is None:
            return None
        spine = freeze.read(STAGE_S01_FOOTPRINTS).inventory
        boundary, _, _ = read_boundary(inputs.boundary)
        inside = frozenset(find_inside(spine, boundary))
        if not inside:
            msg = (
                f'None of the {len(spine)} footprints lies inside the boundary. '
                'Pass a boundary and footprints of one place, both in longitude '
                'and latitude (EPSG:4326).'
            )
            raise ValueError(msg)
        report.set_boundary(inside, len(spine))
        report.settings.update(
            boundary=mapping(inputs.boundary),
            buffer_m=inputs.buffer_m,
            boundary_record=None
            if inputs.boundary_record is None
            else dict(inputs.boundary_record),
        )
        cap = self.imputer_options.get('cap_m', DEFAULT_DONOR_CAP_M)
        if inputs.buffer_m is None or inputs.buffer_m < cap:
            width = (
                'of unknown width'
                if inputs.buffer_m is None
                else f'{inputs.buffer_m:g} m wide'
            )
            report.note(
                f'The area around the boundary is {width}, and the donor cap is '
                f'{cap:g} m, so a building near the boundary may take values other '
                'than those a compilation of a wider area gives it.'
            )
        return inside

    def _read_census_tables(
        self,
        inputs: CompilerInputs,
        scraper: ACSTableScraper | None,
        freeze: StageFreeze,
        report: CompilationReport,
    ) -> CompilerInputs:
        """
        Read the census tables given as ``None`` for the counties of the footprints.

        A county is read whole. The year sampler receives the rows of B25034 for the
        footprints' block groups and tracts only, and every tract of B25127, whose
        county totals it sums.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            scraper (ACSTableScraper | None):
                The scraper the tables are read with, holding the checked key, or
                ``None`` when every table is given, which reads nothing.
            freeze (StageFreeze):
                The freeze holding :data:`STAGE_S01_FOOTPRINTS`, whose block groups
                name the counties.
            report (CompilationReport):
                The report, whose notice names the release, the tables and the
                counties, and which times the reading as ``s01_acs_tables``.

        Returns:
            CompilerInputs:
                A copy of ``inputs`` holding the tables read, each in the form its
                field takes.

        Raises:
            ValueError:
                If a table the year sampler reads holds, in the rows it receives, an
                estimate the Census Bureau does not publish, or as
                :meth:`ACSTableScraper.get_table` raises.
            RuntimeError, requests.exceptions.RequestException:
                As :meth:`ACSTableScraper.get_table` raises.
        """
        if scraper is None:
            return inputs
        started = time.perf_counter()
        block_groups = {
            asset.features[BLOCK_GROUP_KEY]
            for asset in freeze.read(STAGE_S01_FOOTPRINTS).inventory.values()
        }
        # A county's FIPS code is the first five digits of a GEOID.
        counties = sorted({geoid[:5] for geoid in block_groups})
        rows = {
            'b25034_block_groups': block_groups,
            'b25034_tracts': {geoid[:TRACT_GEOID_LENGTH] for geoid in block_groups},
        }
        requested = [
            name for name in _ACS_TABLE_REQUESTS if getattr(inputs, name) is None
        ]
        print(
            f'Reading the ACS {self.acs_vintage} five-year tables of '
            f'{", ".join(requested)} for the counties {", ".join(counties)} from '
            'the Census Data API.'
        )
        tables = {}
        for name in requested:
            table, level = _ACS_TABLE_REQUESTS[name]
            frame = scraper.get_table(
                table, level, counties, vintage=self.acs_vintage
            )
            if name == 'b25024_block_groups':
                # The budget takes the table with a GEOID column.
                tables[name] = frame.rename_axis(GEOID_COLUMN).reset_index()
                continue
            if name in rows:
                frame = frame.loc[frame.index.isin(rows[name])]
            estimates = frame[_SAMPLER_VARIABLES[name]]
            unfit = estimates.index[~np.isfinite(estimates).all(axis=1)]
            if len(unfit):
                msg = (
                    f'The ACS {self.acs_vintage} table {table} read for {name}, by '
                    f'{level.replace("_", " ")}, holds an estimate the Census '
                    f'Bureau does not publish for {len(unfit)} GEOID(s), among them '
                    f'{list(unfit[:5])}. Give {name} in CompilerInputs with those '
                    'estimates filled, or choose another acs_vintage.'
                )
                raise ValueError(msg)
            tables[name] = frame
        report.note(
            f'ACS {self.acs_vintage} five-year tables read from the Census Data API '
            f'for {", ".join(requested)}, covering the counties '
            f'{", ".join(counties)}.'
        )
        report.set_timing('s01_acs_tables', time.perf_counter() - started)
        return inputs._replace(**tables)

    def _join_nsi_to_footprints(
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
    ) -> None:
        """
        Run s02: join the NSI points to the frozen footprints.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            sources (_WorldSources):
                The sources, to which the join's observed and derived values
                are added as :data:`STAGE_S02_NSI_JOIN` and
                :data:`STAGE_S02_DERIVED`, with centroids as their
                coordinates.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive the join's links, unlinked points
                and branch counts.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
        """
        result = NSIFootprintJoin(
            cap_m=self.join_cap_m, seed=self.seed, projected_crs=self.projected_crs
        ).join(sources.freeze.read(STAGE_S01_FOOTPRINTS), inputs.nsi_points)
        logs['join_links'] = result.links
        logs['join_unlinked'] = result.unlinked
        logs['join_branch_counts'] = result.branch_counts.reset_index()
        for part in (result.observed, result.derived):
            for asset_id, asset in part.inventory.items():
                asset.coordinates = centroids[asset_id]
                # NSI's GrossArea also under a name of its own, which no local
                # inventory may hold.
                if not is_missing_value(asset.features.get(GROSS_AREA_KEY)):
                    asset.features[NSI_GROSS_AREA_KEY] = asset.features[
                        GROSS_AREA_KEY
                    ]
        sources.add_part('nsi', STAGE_S02_NSI_JOIN, result.observed)
        sources.add_part('derived', STAGE_S02_DERIVED, result.derived)

    def _infer_story_counts(
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
    ) -> StoriesFromHeightInferer:
        """
        Run s03: story counts from the footprint heights.

        The section "The run" of the README beside this module states the
        procedure.

        Args:
            inputs (CompilerInputs):
                The study area's data, whose height unit the inferer reads.
            sources (_WorldSources):
                The sources, to which the counts are added as the part
                :data:`STAGE_S03_STORY_COUNTS` and the NSI counts they
                replace as the record :data:`STAGE_S03_WITHHELD`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``stories``.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            report (CompilationReport):
                The report, which receives the withheld NSI counts of every
                world.
            n_computed (int):
                The number of worlds computed.

        Returns:
            StoriesFromHeightInferer:
                The inferer of story counts from footprint heights, which the
                pair re-draw reads too.

        Raises:
            ValueError:
                If an asset's source string is not one
                :func:`~brails.types.asset_inventory.read_source_string`
                reads.
        """
        merged = sources.merge_world(0)
        inferer = StoriesFromHeightInferer(length_unit=inputs.height_unit)
        verdicts = inferer.check(merged)
        logs['stories'] = verdicts.rename_axis('asset_id').reset_index()
        passing = _select_assets(
            merged,
            verdicts.index[verdicts['diagnosis'] == DIAGNOSIS_OK],
            vector_features=DEFAULT_VECTOR_FEATURES,
        )
        counts = inferer.draw(passing)[STORIES_FROM_HEIGHT_KEY]

        derived: dict[str, dict[str, Any]] = {}
        withheld: dict[str, dict[str, str]] = {}
        # (asset id, old count) of each NSI count replaced.
        replaced: list[tuple] = []
        for asset_id, count in counts.items():
            features = merged.inventory[asset_id].features
            values = {STORIES_FROM_HEIGHT_KEY: count}
            code = read_source_string(asset_id, features)[
                _CODE_POSITION[STORIES_KEY]
            ]
            if is_missing_value(features.get(STORIES_KEY)):
                values[STORIES_KEY] = count
            elif (
                code == SOURCE_CODES['nsi']
                and features.get(OCCUPANCY_KEY) != NSI_STORIES_CLASS
            ):
                values[STORIES_KEY] = count
                withheld[asset_id] = {STORIES_KEY: 'nsi'}
                replaced.append((asset_id, features[STORIES_KEY]))
            derived[asset_id] = values

        sources.add_part(
            'derived',
            STAGE_S03_STORY_COUNTS,
            _build_inventory(sorted(derived, key=str), centroids, derived),
        )
        sources.add_withheld(
            STAGE_S03_WITHHELD,
            _build_inventory(sorted(withheld, key=str), centroids, withheld),
        )
        for world in range(n_computed):
            for asset_id, old in replaced:
                report.record_action(
                    asset_id,
                    STORIES_KEY,
                    stage='s03_story_counts',
                    action='withheld',
                    world=world,
                    old=old,
                    new=derived[asset_id][STORIES_KEY],
                    source='nsi',
                    reason=(
                        'the footprint height gives the story count of an '
                        f'occupancy class other than {NSI_STORIES_CLASS}'
                    ),
                )
        return inferer

    def _allocate_units(
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
        merged: AssetInventory,
        occupancy_source: Mapping[str, str | None],
        imputer: ObservedDonorImputer,
    ) -> UnitBudgetResult:
        """
        Run s04: the unit budget, which drops footprints left without units.

        The section "The run" of the README beside this module states the
        procedure.

        Args:
            inputs (CompilerInputs):
                The study area's data, whose B25024 table and mobile-home
                park points the budget reads.
            sources (_WorldSources):
                The sources, to which the budget's values are added as the
                part :data:`STAGE_S04_UNIT_BUDGET`, the dropped footprints as
                the record :data:`STAGE_S04_DROPPED`, and the occupancy
                classes it relabels as the record :data:`STAGE_S04_WITHHELD`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``budget``, ``budget_diagnostic``,
                ``budget_gaps`` and ``budget_nonresidential``.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            report (CompilationReport):
                The report, which receives the relabeled classes and the
                dropped footprints of every world.
            n_computed (int):
                The number of worlds computed.
            merged (AssetInventory):
                World 0 merged after s03, which s05 reads as well. Read and
                not changed.
            occupancy_source (Mapping[str, str | None]):
                The source of each footprint's occupancy class in ``merged``,
                as :func:`_read_occupancy_sources` reads it.
            imputer (ObservedDonorImputer):
                The imputer whose status round the budget takes and which
                s05 runs.

        Returns:
            UnitBudgetResult:
                The budget's allocation, whose values s05's targets carry.
        """
        # A footprint has an observed occupancy class where one of
        # _OCCUPANCY_SOURCE_LABELS holds its class.
        with_occupancy_source = _select_assets(
            merged,
            (
                a
                for a, source in occupancy_source.items()
                if source in _OCCUPANCY_SOURCE_LABELS
            ),
            vector_features=DEFAULT_VECTOR_FEATURES,
        )
        without_occupancy_source = _select_assets(
            merged,
            (
                a
                for a, source in occupancy_source.items()
                if source not in _OCCUPANCY_SOURCE_LABELS
            ),
            vector_features=DEFAULT_VECTOR_FEATURES,
        )
        relabel = {
            a: source
            for a, source in occupancy_source.items()
            if source in _RELABELED_SOURCES
        }
        parks = inputs.mh_park_points
        if isinstance(parks, AssetInventory):
            # The budget takes the points' longitude and latitude.
            parks = np.array(
                [asset.coordinates[0] for asset in parks.inventory.values()]
            )
        budget = ACSUnitBudget(
            inputs.b25024_block_groups,
            parks,
            imputer=imputer,
            seed=self.seed,
            **self.budget_options,
        )
        allocation = budget.allocate(
            without_occupancy_source,
            with_occupancy_source,
            n_worlds=n_computed,
            relabel_sources=relabel,
        )
        logs['budget'] = allocation.log
        logs['budget_diagnostic'] = allocation.diagnostic
        logs['budget_gaps'] = allocation.gaps
        logs['budget_nonresidential'] = pd.DataFrame(
            {'asset_id': list(allocation.nonresidential)}, dtype=object
        )
        withheld: dict[str, dict[str, str]] = {}
        for row in allocation.withheld.itertuples(index=False):
            withheld.setdefault(row.asset_id, {})[row.feature] = row.source
        dropped_ids = set(allocation.dropped)
        no_acs = set(
            allocation.gaps.loc[allocation.gaps['kind'] == GAP_NO_ACS, 'block_group']
        )
        kept = sorted((a for a in merged.inventory if a not in dropped_ids), key=str)

        sources.freeze.freeze(
            STAGE_S04_DROPPED,
            _build_inventory(
                sorted(dropped_ids, key=str),
                centroids,
                {asset_id: {DROPPED_KEY: True} for asset_id in dropped_ids},
            ),
        )
        sources.dropped = STAGE_S04_DROPPED
        sources.add_withheld(
            STAGE_S04_WITHHELD,
            _build_inventory(sorted(withheld, key=str), centroids, withheld),
        )
        sources.add_part(
            'imputed',
            STAGE_S04_UNIT_BUDGET,
            _build_inventory(
                kept,
                centroids,
                {
                    asset_id: asset.features
                    for asset_id, asset in allocation.imputed.inventory.items()
                },
                n_pw=n_computed,
                vector_features=allocation.imputed.vector_features,
            ),
        )

        for world in range(n_computed):
            for row in allocation.withheld.itertuples(index=False):
                new = allocation.imputed.inventory[row.asset_id].features.get(
                    row.feature
                )
                report.record_action(
                    row.asset_id,
                    row.feature,
                    stage='s04_unit_budget',
                    action='withheld',
                    world=world,
                    old=row.value,
                    new=new[world] if isinstance(new, list) else new,
                    source=row.source,
                    reason=(
                        'a footprint near a mobile-home park point made a mobile '
                        'home'
                    ),
                )
            for asset_id in sorted(dropped_ids, key=str):
                block_group = merged.inventory[asset_id].features.get(
                    BLOCK_GROUP_KEY
                )
                detail = (
                    f'plan area under {budget.threshold_sqft:g} sq ft and '
                    f'beyond {budget.mh_park_radius_m:g} m of every mobile-home '
                    'park point, '
                    + (
                        'in a block group without B25024 counts, with a '
                        f'residential probability under {DROP_PROBABILITY:g}'
                        if block_group in no_acs
                        else 'not chosen as a dwelling'
                    )
                )
                report.record_exclusion(
                    asset_id,
                    world=world,
                    stage='s04_unit_budget',
                    category=DROPPED_CATEGORY,
                    diagnosis=DIAGNOSIS_NOT_SELECTED,
                    detail=detail,
                )
        return allocation

    def _impute_from_donors(
        self,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        n_computed: int,
        merged: AssetInventory,
        occupancy_source: Mapping[str, str | None],
        imputer: ObservedDonorImputer,
        allocation: UnitBudgetResult,
    ) -> _ImputationOutcome:
        """
        Run s05: fill what is still missing from observed donors.

        The targets carry the values of s04's allocation, and the imputer
        takes each target's residential status from it in place of its own
        status round. The section "The run" of the README beside this module
        states the procedure.

        Args:
            sources (_WorldSources):
                The sources, to which the imputed values are added as the
                part :data:`STAGE_S05_IMPUTATION`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``imputer``.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            n_computed (int):
                The number of worlds computed.
            merged (AssetInventory):
                World 0 merged after s03, as s04 read it. Read and not
                changed.
            occupancy_source (Mapping[str, str | None]):
                The source of each footprint's occupancy class in ``merged``,
                as s04 read it.
            imputer (ObservedDonorImputer):
                The imputer s04's budget was given.
            allocation (UnitBudgetResult):
                The allocation of s04, whose values the targets carry.

        Returns:
            _ImputationOutcome:
                The imputer, its targets and its log.
        """
        # Every footprint the budget kept that is not an observed donor is a
        # target.
        donors = _select_observed_donors(
            merged, occupancy_source, vector_features=DEFAULT_VECTOR_FEATURES
        )
        dropped_ids = set(allocation.dropped)
        # Each target holds a copy of its merged features, updated with the
        # features the unit budget allocated.
        target_features = {}
        for asset_id, asset in merged.inventory.items():
            if asset_id in dropped_ids or asset_id in donors.inventory:
                continue
            features = dict(asset.features)
            budgeted = allocation.imputed.inventory.get(asset_id)
            if budgeted is not None:
                features.update(budgeted.features)
            target_features[asset_id] = features
        targets = _build_inventory(
            target_features,
            {a: merged.inventory[a].coordinates for a in target_features},
            target_features,
            n_pw=n_computed,
        )
        imputation = imputer.impute_stage(
            targets,
            donors,
            n_worlds=n_computed,
            status=allocation.residential,
            validate=_is_plausible_pair,
        )
        imputer_log = imputation.log.copy()
        imputer_log['values'] = [
            json.dumps(dict(values), default=str, sort_keys=True)
            for values in imputer_log['values']
        ]
        logs['imputer'] = imputer_log
        # A footprint with an observed occupancy class outside the residential
        # classes is a target for its story count only, since the join gives
        # units only to the NSI points of housing.
        for asset_id, asset in imputation.imputed.inventory.items():
            class_of = merged.inventory[asset_id].features.get(OCCUPANCY_KEY)
            if (
                occupancy_source[asset_id] in _OCCUPANCY_SOURCE_LABELS
                and class_of not in UNIT_COUNT_CLASSES
            ):
                asset.features.pop(UNITS_KEY, None)

        kept = sorted((a for a in merged.inventory if a not in dropped_ids), key=str)
        sources.add_part(
            'imputed',
            STAGE_S05_IMPUTATION,
            _build_inventory(
                kept,
                centroids,
                {
                    asset_id: asset.features
                    for asset_id, asset in imputation.imputed.inventory.items()
                },
                n_pw=n_computed,
                vector_features=imputation.imputed.vector_features,
            ),
        )
        return _ImputationOutcome(imputer, targets, imputation.log)

    def _draw_residential_years(
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
    ) -> None:
        """
        Run s06 in every world: the year built of residential buildings.

        The section "The run" of the README beside this module states the
        procedure.

        Args:
            inputs (CompilerInputs):
                The study area's data, whose census tables the year sampler
                reads.
            sources (_WorldSources):
                The sources, to which the years are added as the part
                :data:`STAGE_S06_RESIDENTIAL_YEAR`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``year_fit``, with a ``world``
                column.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            report (CompilationReport):
                The report, which receives the exclusions of every world.
            n_computed (int):
                The number of worlds computed.
        """
        view = sources.freeze.read(STAGE_S05_VIEW)
        kept = list(view.inventory)
        assembled = AssetInventory()
        fits = []
        for world in range(n_computed):
            started = time.perf_counter()
            frozen = view.get_world_realization(world)
            sampler = ACSYearBuiltSampler(
                inputs.b25034_block_groups,
                inputs.b25034_tracts,
                inputs.b25127_tracts,
                seed=self.seed,
                world=world,
                **self.sampler_options,
            ).fit(frozen)
            fits.append(sampler.fit_report.reset_index().assign(world=world))
            targets = [
                a
                for a, asset in frozen.inventory.items()
                if asset.features.get(OCCUPANCY_KEY) in UNIT_COUNT_CLASSES
                and is_missing_value(asset.features.get(YEAR_BUILT_KEY))
            ]
            verdicts = sampler.check(_select_assets(frozen, targets))
            for row in verdicts[verdicts['diagnosis'] != DIAGNOSIS_OK].itertuples():
                report.record_exclusion(
                    row.Index,
                    world=world,
                    stage='s06_residential_year',
                    category=_DIAGNOSIS_CATEGORY[row.diagnosis],
                    diagnosis=row.diagnosis,
                    detail=row.detail,
                    resolvable_by=row.resolvable_by,
                )
            passing = list(verdicts.index[verdicts['diagnosis'] == DIAGNOSIS_OK])
            years = sampler.draw(_select_assets(frozen, passing))
            assembled.add_world(
                _build_inventory(
                    kept,
                    centroids,
                    {
                        asset_id: {YEAR_BUILT_KEY: int(year)}
                        for asset_id, year in years.items()
                    },
                    vector_features=(),
                )
            )
            del frozen
            report.set_timing(
                's06_residential_year', time.perf_counter() - started, world=world
            )

        logs['year_fit'] = pd.concat(fits, ignore_index=True)
        started = time.perf_counter()
        sources.add_part('drawn', STAGE_S06_RESIDENTIAL_YEAR, assembled)
        report.set_timing(
            's06_residential_year_freeze', time.perf_counter() - started
        )

    def _redraw_implausible_pairs(
        self,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
        imputer: ObservedDonorImputer,
        stories: StoriesFromHeightInferer,
    ) -> None:
        """
        Run s07 in every world: re-draw implausible pairs of class and stories.

        Each world's re-draw reads that world merged after s06, and its
        donors are the observed donors of the world's realization of the
        view :data:`STAGE_S05_VIEW`. The section "The run" of the README beside
        this module states the procedure.

        Args:
            sources (_WorldSources):
                The sources, to which the re-drawn values are added as the
                part :data:`STAGE_S07_PAIR_REDRAW`, the height counts that
                replace NSI's as the part :data:`STAGE_S07_DERIVED`, and the
                values they replace as the record :data:`STAGE_S07_WITHHELD`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``pair_redraw``, with a ``world``
                column.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            report (CompilationReport):
                The report, which receives the actions of every world.
            n_computed (int):
                The number of worlds computed.
            imputer (ObservedDonorImputer):
                The imputer of s05, whose draw the pair re-draw takes.
            stories (StoriesFromHeightInferer):
                The inferer of story counts from footprint heights.

        Raises:
            ValueError:
                If an asset's source string is not one
                :func:`~brails.types.asset_inventory.read_source_string`
                reads.
        """
        view = sources.freeze.read(STAGE_S05_VIEW)
        kept = list(view.inventory)
        stages = (STAGE_S07_WITHHELD, STAGE_S07_PAIR_REDRAW, STAGE_S07_DERIVED)
        assembled = {stage: AssetInventory() for stage in stages}
        redraws = []
        for world in range(n_computed):
            started = time.perf_counter()
            frozen = view.get_world_realization(world)
            merged_world = sources.merge_world(world)
            new: dict[str, dict[str, dict[str, Any]]] = {
                stage: {} for stage in stages
            }
            # The re-draw draws from the observed donors of the view.
            donors = _select_observed_donors(frozen, _read_occupancy_sources(frozen))
            redraw = PairRedraw(imputer, donors, stories, world=world).redraw(
                merged_world
            )
            redraws.append(redraw.log)
            for row in redraw.withheld.itertuples(index=False):
                new[STAGE_S07_WITHHELD].setdefault(row.asset_id, {})[row.feature] = (
                    row.source
                )
                replaced = {
                    **redraw.updates.get(row.asset_id, {}),
                    **redraw.derived.get(row.asset_id, {}),
                }
                report.record_action(
                    row.asset_id,
                    row.feature,
                    stage='s07_pair_redraw',
                    action='withheld',
                    world=world,
                    old=row.value,
                    new=replaced.get(row.feature),
                    source=row.source,
                    reason=(
                        'a member of an implausible pair of occupancy class and '
                        'story count'
                    ),
                )
            for label, stage, action, reason, values_of in (
                (
                    'imputed',
                    STAGE_S07_PAIR_REDRAW,
                    'redrawn',
                    "a donor's value for a member of an implausible pair",
                    redraw.updates,
                ),
                (
                    'derived',
                    STAGE_S07_DERIVED,
                    'derived',
                    "the footprint height's count for an uncorroborated NSI count",
                    redraw.derived,
                ),
            ):
                for asset_id, values in values_of.items():
                    for feature, value in values.items():
                        report.record_action(
                            asset_id,
                            feature,
                            stage='s07_pair_redraw',
                            action=action,
                            world=world,
                            old=merged_world.inventory[asset_id].features.get(
                                feature
                            ),
                            new=value,
                            source=label,
                            reason=reason,
                        )
                    new[stage].setdefault(asset_id, {}).update(values)
                    _write_coded_values(
                        asset_id,
                        merged_world.inventory[asset_id].features,
                        values,
                        label,
                    )

            for stage in stages:
                assembled[stage].add_world(
                    _build_inventory(kept, centroids, new[stage], vector_features=())
                )
            del frozen, merged_world
            report.set_timing(
                's07_pair_redraw', time.perf_counter() - started, world=world
            )

        logs['pair_redraw'] = pd.concat(redraws, ignore_index=True)
        started = time.perf_counter()
        sources.add_withheld(STAGE_S07_WITHHELD, assembled[STAGE_S07_WITHHELD])
        sources.add_part(
            'imputed', STAGE_S07_PAIR_REDRAW, assembled[STAGE_S07_PAIR_REDRAW]
        )
        sources.add_part('derived', STAGE_S07_DERIVED, assembled[STAGE_S07_DERIVED])
        report.set_timing('s07_pair_redraw_freeze', time.perf_counter() - started)

    def _assign_structure_in_each_world(
        self,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
        *,
        state: str,
        seismic_zone: str,
    ) -> None:
        """
        Give each building its design level, structure type and height class.

        This runs in every world and also applies the W1 carve-out. Each
        world's draws read that world merged after s07. A building excluded
        from the world by an earlier step, which can only be s06, takes none.
        The section "The run" of the README beside this module states the
        procedure.

        Args:
            sources (_WorldSources):
                The sources, to which the parts are added.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``resolution``, with a ``world``
                column.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            report (CompilationReport):
                The report, whose exclusions are read and which receives the
                actions and exclusions of every world.
            n_computed (int):
                The number of worlds computed.
            state (str):
                The full state name the compilation applies.
            seismic_zone (str):
                The 1997 UBC seismic zone the compilation applies.

        Raises:
            ValueError:
                If an asset's source string is not one
                :func:`~brails.types.asset_inventory.read_source_string`
                reads, or, as :meth:`ResolutionLoop.resolve` raises, a
                building type has no source.
        """
        kept = list(sources.freeze.read(STAGE_S05_VIEW).inventory)
        exclusions = report.exclusions
        stages = (STAGE_S08_STRUCTURE, STAGE_S08_DERIVED)
        assembled = {stage: AssetInventory() for stage in stages}
        resolutions = []
        for world in range(n_computed):
            started = time.perf_counter()
            merged_world = sources.merge_world(world)
            new: dict[str, dict[str, dict[str, Any]]] = {
                stage: {} for stage in stages
            }
            excluded = set(exclusions.loc[exclusions['world'] == world, 'asset_id'])
            resolutions.append(
                {
                    'world': world,
                    **self._assign_structure(
                        world,
                        sources,
                        report,
                        merged_world,
                        kept,
                        excluded,
                        new[STAGE_S08_DERIVED],
                        new[STAGE_S08_STRUCTURE],
                        state=state,
                        seismic_zone=seismic_zone,
                    ),
                }
            )
            for stage in stages:
                assembled[stage].add_world(
                    _build_inventory(kept, centroids, new[stage], vector_features=())
                )
            del merged_world
            report.set_timing(
                's08_structure', time.perf_counter() - started, world=world
            )

        logs['resolution'] = pd.DataFrame(resolutions)
        started = time.perf_counter()
        sources.add_part(
            'drawn', STAGE_S08_STRUCTURE, assembled[STAGE_S08_STRUCTURE]
        )
        sources.add_part('derived', STAGE_S08_DERIVED, assembled[STAGE_S08_DERIVED])
        report.set_timing('s08_structure_freeze', time.perf_counter() - started)

    def _assign_structure(  # noqa: C901
        self,
        world: int,
        sources: _WorldSources,
        report: CompilationReport,
        merged_world: AssetInventory,
        candidates: Sequence[str],
        excluded: set[str],
        derived: dict[str, dict[str, Any]],
        drawn: dict[str, dict[str, Any]],
        *,
        state: str,
        seismic_zone: str,
        step: str = 's08',
    ) -> dict[str, int]:
        """
        Give assets their design levels, structure types and height classes.

        Design levels and structure types go to the assets with a
        construction year, and height classes to every asset with a
        structure type and no height class.

        The section "The run" of the README beside this module states the
        procedure.

        Args:
            world (int):
                The index of the world being compiled. The structure type
                inferer draws in it, and the report records it.
            sources (_WorldSources):
                The sources, whose frozen parts say which source holds a
                building type.
            report (CompilationReport):
                The report, which receives the actions and exclusions.
            merged_world (AssetInventory):
                The world's merged inventory, into which the design levels
                are written.
            candidates (Sequence[str]):
                The ids of the assets to give values; those in
                ``excluded`` are skipped.
            excluded (set[str]):
                The assets excluded in the world, to which the refused ones
                are added.
            derived (dict[str, dict[str, Any]]):
                The world's new derived values, written per asset.
            drawn (dict[str, dict[str, Any]]):
                The world's new drawn values, written per asset.
            state (str):
                The full state name the compilation applies.
            seismic_zone (str):
                The 1997 UBC seismic zone the compilation applies.
            step (str, optional):
                The label of the step the report files the rows under, the
                prefix of each report stage: ``'s08'`` for the buildings with
                a year before s09, ``'s10'`` for those s09 gives one.
                Defaults to ``'s08'``.

        Returns:
            dict[str, int]:
                The loop's counts: ``passed_first_screen``,
                ``resolved_by_withholding`` and ``excluded``.

        Raises:
            ValueError:
                If an asset's source string is not one
                :func:`~brails.types.asset_inventory.read_source_string`
                reads, or, as :meth:`ResolutionLoop.resolve` raises, a
                building type has no source.
        """
        design = HazusEQDesignLevelInferer(
            self.design_level_method, state, seismic_zone
        )
        heights = HazusEQHeightClassInferer()
        labeled = [
            (label, sources.freeze.read(stage))
            for label in PRECEDENCE
            for stage in sources.parts[label]
        ]

        def holds_value(asset_id: str, key: str) -> bool:
            """
            Say whether an asset holds a feature's value in the merged inventory.

            Args:
                asset_id (str):
                    The id of the asset to look up.
                key (str):
                    The name of the feature to look up.

            Returns:
                bool:
                    Whether the value is not missing.
            """
            return not is_missing_value(
                merged_world.inventory[asset_id].features.get(key)
            )

        # Derive a design level for each asset with a year and no design level.
        with_year = [
            a
            for a in candidates
            if a not in excluded
            and holds_value(a, YEAR_BUILT_KEY)
            and not holds_value(a, DESIGN_LEVEL_KEY)
        ]
        checked = design.check(_select_assets(merged_world, with_year))
        for row in checked[checked['diagnosis'] != DIAGNOSIS_OK].itertuples():
            excluded.add(row.Index)
            report.record_exclusion(
                row.Index,
                world=world,
                stage=f'{step}_design_level',
                category=_DIAGNOSIS_CATEGORY[row.diagnosis],
                diagnosis=row.diagnosis,
                detail=row.detail,
                resolvable_by=row.resolvable_by,
            )
        usable = list(checked.index[checked['diagnosis'] == DIAGNOSIS_OK])
        for asset_id, level in design.draw(
            _select_assets(merged_world, usable)
        ).items():
            derived.setdefault(asset_id, {})[DESIGN_LEVEL_KEY] = level
            _write_coded_values(
                asset_id,
                merged_world.inventory[asset_id].features,
                {DESIGN_LEVEL_KEY: level},
                'derived',
            )

        # The screen and the loop, then the type draw.
        structure = HazusEQStructureTypeInferer(
            state, seed=self.seed, world=world, complete_tables=True
        )
        loop = ResolutionLoop(
            ComposedScreen(design, structure),
            report_stages={
                STAGE_DESIGN_LEVEL: f'{step}_design_level',
                STAGE_STRUCTURE_TYPE: f'{step}_structure_type',
                STAGE_RESOLUTION: f'{step}_resolution',
            },
        )
        without_structure_type = [
            a
            for a in candidates
            if a not in excluded
            and holds_value(a, YEAR_BUILT_KEY)
            and not holds_value(a, STRUCTURE_TYPE_KEY)
        ]
        # The source holding each building type, in merge precedence.
        type_sources = {}
        for asset_id in without_structure_type:
            if not holds_value(asset_id, BUILDING_TYPE_KEY):
                continue
            for label, part in labeled:
                asset = part.inventory.get(asset_id)
                value = (
                    None if asset is None else asset.features.get(BUILDING_TYPE_KEY)
                )
                if isinstance(value, list):
                    value = value[world]
                if not is_missing_value(value):
                    type_sources[asset_id] = label
                    break
        resolution = loop.resolve(
            _select_assets(merged_world, without_structure_type),
            report,
            building_type_sources=type_sources,
        )
        excluded.update(resolution.excluded)
        types = structure.draw(resolution.resolved)
        for asset_id, structure_type in types.items():
            drawn.setdefault(asset_id, {})[STRUCTURE_TYPE_KEY] = structure_type

        # Derive a height class for each asset that has a structure type and no
        # height class.
        height_class_inputs = {}
        for asset_id in candidates:
            structure_type = types.get(asset_id)
            if structure_type is None and holds_value(asset_id, STRUCTURE_TYPE_KEY):
                structure_type = merged_world.inventory[asset_id].features[
                    STRUCTURE_TYPE_KEY
                ]
            if (
                asset_id in excluded
                or structure_type is None
                or holds_value(asset_id, HEIGHT_CLASS_KEY)
            ):
                continue
            height_class_inputs[asset_id] = {
                STRUCTURE_TYPE_KEY: structure_type,
                STORIES_KEY: merged_world.inventory[asset_id].features.get(
                    STORIES_KEY
                ),
            }
        with_structure_type = _build_inventory(
            height_class_inputs,
            {a: merged_world.inventory[a].coordinates for a in height_class_inputs},
            height_class_inputs,
        )
        classed = heights.check(with_structure_type)
        for row in classed[classed['diagnosis'] != DIAGNOSIS_OK].itertuples():
            excluded.add(row.Index)
            report.record_exclusion(
                row.Index,
                world=world,
                stage=f'{step}_height_class',
                category=_DIAGNOSIS_CATEGORY[row.diagnosis],
                diagnosis=row.diagnosis,
                detail=row.detail,
            )
        ok = list(classed.index[classed['diagnosis'] == DIAGNOSIS_OK])
        for asset_id, height_class in heights.draw(
            _select_assets(with_structure_type, ok)
        ).items():
            if height_class is not None:
                derived.setdefault(asset_id, {})[HEIGHT_CLASS_KEY] = height_class

        if state == CARVE_OUT_STATE and design.method == CARVE_OUT_METHOD:
            drawn_levels = [
                a
                for a in types.index
                if a in derived and DESIGN_LEVEL_KEY in derived[a]
            ]
            levels = pd.Series(
                {a: derived[a][DESIGN_LEVEL_KEY] for a in drawn_levels},
                dtype=object,
            )
            adjusted = apply_w1_carveout(levels, types.reindex(levels.index))
            for asset_id in levels.index[adjusted != levels]:
                derived[asset_id][DESIGN_LEVEL_KEY] = adjusted[asset_id]
                report.record_action(
                    asset_id,
                    DESIGN_LEVEL_KEY,
                    stage=f'{step}_carve_out',
                    action='derived',
                    world=world,
                    old=levels[asset_id],
                    new=adjusted[asset_id],
                    source='derived',
                    reason='a Pre-Code W1 building is taken as Moderate-Code',
                )

        return {
            'passed_first_screen': resolution.passed_first_screen,
            'resolved_by_withholding': resolution.resolved_by_withholding,
            'excluded': resolution.excluded_count,
        }

    def _impute_nonresidential_years(
        self,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
        imputer: ObservedDonorImputer,
        *,
        state: str,
        seismic_zone: str,
    ) -> None:
        """
        Run s09 and s10: years built, then structure, for buildings without a year.

        s09 draws a year built for each building without one, and s10 gives
        those buildings their design levels, structure types and height
        classes. Each step is timed in each world, and the freeze of its
        parts once, under its own label.

        The section "The run" of the README beside this module states the
        procedure.

        Args:
            sources (_WorldSources):
                The sources, to which the parts are added as
                :data:`STAGE_S09_NONRESIDENTIAL_YEAR`,
                :data:`STAGE_S10_NONRESIDENTIAL_STRUCTURE` and
                :data:`STAGE_S10_DERIVED`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``neighbor_year`` and
                ``nonresidential_resolution``, each with a ``world`` column.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            report (CompilationReport):
                The report, whose exclusions are read and which receives the
                actions and exclusions of every world.
            n_computed (int):
                The number of worlds computed.
            imputer (ObservedDonorImputer):
                The imputer of s05, whose ``k`` and ``cap_m`` the sampler
                takes.
            state (str):
                The full state name the compilation applies.
            seismic_zone (str):
                The 1997 UBC seismic zone the compilation applies.

        Raises:
            ValueError:
                If an asset's source string is not one
                :func:`~brails.types.asset_inventory.read_source_string`
                reads, or, as :meth:`ResolutionLoop.resolve` raises, a
                building type has no source.
        """
        view = sources.freeze.read(STAGE_S08_VIEW)
        kept = list(view.inventory)
        exclusions = report.exclusions
        stages = (
            STAGE_S09_NONRESIDENTIAL_YEAR,
            STAGE_S10_NONRESIDENTIAL_STRUCTURE,
            STAGE_S10_DERIVED,
        )
        assembled = {stage: AssetInventory() for stage in stages}
        picks, resolutions = [], []
        for world in range(n_computed):
            started = time.perf_counter()
            excluded = set(exclusions.loc[exclusions['world'] == world, 'asset_id'])
            frozen = view.get_world_realization(world)
            merged_world = view.get_world_realization(world)
            new: dict[str, dict[str, dict[str, Any]]] = {
                stage: {} for stage in stages
            }
            sampler = ResidentialNeighborYearSampler(
                seed=self.seed,
                world=world,
                k=imputer.k,
                cap_m=imputer.cap_m,
            ).select_donors(sources.freeze, STAGE_S08_VIEW)
            targets = [
                a
                for a, asset in frozen.inventory.items()
                if a not in excluded
                and asset.features.get(OCCUPANCY_KEY) not in UNIT_COUNT_CLASSES
                and is_missing_value(asset.features.get(YEAR_BUILT_KEY))
            ]
            verdicts = sampler.check(_select_assets(frozen, targets))
            for row in verdicts[verdicts['diagnosis'] != DIAGNOSIS_OK].itertuples():
                excluded.add(row.Index)
                report.record_exclusion(
                    row.Index,
                    world=world,
                    stage='s09_nonresidential_year',
                    category=_DIAGNOSIS_CATEGORY[row.diagnosis],
                    diagnosis=row.diagnosis,
                    detail=row.detail,
                    resolvable_by=row.resolvable_by,
                )
            passing = list(verdicts.index[verdicts['diagnosis'] == DIAGNOSIS_OK])
            picked = sampler.draw_details(_select_assets(frozen, passing))
            picks.append(
                picked.rename_axis('asset_id').reset_index().assign(world=world)
            )
            for asset_id, year in picked[YEAR_BUILT_KEY].items():
                new[STAGE_S09_NONRESIDENTIAL_YEAR][asset_id] = {
                    YEAR_BUILT_KEY: int(year)
                }
                _write_coded_values(
                    asset_id,
                    merged_world.inventory[asset_id].features,
                    {YEAR_BUILT_KEY: int(year)},
                    'imputed',
                )
            assembled[STAGE_S09_NONRESIDENTIAL_YEAR].add_world(
                _build_inventory(
                    kept,
                    centroids,
                    new[STAGE_S09_NONRESIDENTIAL_YEAR],
                    vector_features=(),
                )
            )
            report.set_timing(
                STAGE_S09_NONRESIDENTIAL_YEAR,
                time.perf_counter() - started,
                world=world,
            )
            started = time.perf_counter()
            resolutions.append(
                {
                    'world': world,
                    **self._assign_structure(
                        world,
                        sources,
                        report,
                        merged_world,
                        passing,
                        excluded,
                        new[STAGE_S10_DERIVED],
                        new[STAGE_S10_NONRESIDENTIAL_STRUCTURE],
                        state=state,
                        seismic_zone=seismic_zone,
                        step='s10',
                    ),
                }
            )
            for stage in stages[1:]:
                assembled[stage].add_world(
                    _build_inventory(kept, centroids, new[stage], vector_features=())
                )
            del frozen, merged_world
            report.set_timing(
                STAGE_S10_NONRESIDENTIAL_STRUCTURE,
                time.perf_counter() - started,
                world=world,
            )

        logs['neighbor_year'] = pd.concat(picks, ignore_index=True)
        logs['nonresidential_resolution'] = pd.DataFrame(resolutions)
        started = time.perf_counter()
        sources.add_part(
            'imputed',
            STAGE_S09_NONRESIDENTIAL_YEAR,
            assembled[STAGE_S09_NONRESIDENTIAL_YEAR],
        )
        report.set_timing(
            f'{STAGE_S09_NONRESIDENTIAL_YEAR}_freeze', time.perf_counter() - started
        )
        started = time.perf_counter()
        sources.add_part(
            'drawn',
            STAGE_S10_NONRESIDENTIAL_STRUCTURE,
            assembled[STAGE_S10_NONRESIDENTIAL_STRUCTURE],
        )
        sources.add_part('derived', STAGE_S10_DERIVED, assembled[STAGE_S10_DERIVED])
        report.set_timing(
            f'{STAGE_S10_NONRESIDENTIAL_STRUCTURE}_freeze',
            time.perf_counter() - started,
        )

    def _freeze_view(
        self,
        stage: str,
        sources: _WorldSources,
        report: CompilationReport,
        n_computed: int,
    ) -> None:
        """
        Freeze each world's merged inventory of every source as one view.

        The report receives the time of each world's merge under
        ``f'{stage}_merge'`` and that of the freeze under
        ``f'{stage}_freeze'``.

        Args:
            stage (str):
                The name the view is frozen under.
            sources (_WorldSources):
                The sources whose parts and records each world's merge takes.
            report (CompilationReport):
                The report, which receives the timings.
            n_computed (int):
                The number of worlds computed.
        """
        view = AssetInventory()
        for world in range(n_computed):
            started = time.perf_counter()
            view.add_world(sources.merge_world(world))
            report.set_timing(
                f'{stage}_merge', time.perf_counter() - started, world=world
            )
        started = time.perf_counter()
        sources.freeze.freeze(stage, view)
        report.set_timing(f'{stage}_freeze', time.perf_counter() - started)

    def _compute_gross_areas(
        self,
        world: int,
        sources: _WorldSources,
        centroids: Mapping[str, list[list[float]]],
        rows: list[tuple[str, int, str | None]],
        report: CompilationReport,
    ) -> AssetInventory:
        """
        Run s12 in one world: the gross area of every building.

        The area is computed from the footprint in the world's realization of
        the view :data:`STAGE_S08_VIEW`, whose story counts are final, for
        every building of the view, excluded ones included, so that a
        building the schema check excludes keeps its row in the log
        ``gross_area``. The section "The run" of the README beside this
        module states the procedure.

        Args:
            world (int):
                The index of the world whose areas are computed.
            sources (_WorldSources):
                The sources, whose freeze holds the view.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            rows (list[tuple[str, int, str | None]]):
                The rows of the log ``gross_area``, to which the step adds the
                (asset id, world, ``area_source``) that
                ``footprint_gross_area`` gives each building in the world.
            report (CompilationReport):
                The report, which receives the step's time in the world.

        Returns:
            AssetInventory:
                One asset per building of the view, holding its gross area
                in the world where the footprint gives one. The run freezes
                the areas of every world as :data:`STAGE_S12_GROSS_AREA`,
                which no merge takes.

        Raises:
            ValueError:
                As :func:`footprint_gross_area` raises, for a building
                without a readable footprint.
        """
        started = time.perf_counter()
        view = sources.freeze.read(STAGE_S08_VIEW)
        values = {}
        for asset_id, asset in view.get_world_realization(world).inventory.items():
            area, area_source = footprint_gross_area(asset_id, asset)
            rows.append((asset_id, world, area_source))
            if area is not None:
                values[asset_id] = {GROSS_AREA_KEY: area}
        report.set_timing(
            STAGE_S12_GROSS_AREA, time.perf_counter() - started, world=world
        )
        return _build_inventory(
            list(view.inventory), centroids, values, vector_features=()
        )

    def _withhold_replaced_values(  # noqa: C901
        self, sources: _WorldSources, centroids: Mapping[str, list[list[float]]]
    ) -> None:
        """
        Withhold from every later merge the values that s12, s13 and s14 replace.

        Each record is named by the step whose value takes the place of the
        values it withholds, and each names, per building, the source a value
        is withheld from: :data:`STAGE_S12_WITHHELD_LOCAL` the local
        ``GrossArea``, :data:`STAGE_S12_WITHHELD_JOIN` NSI's ``GrossArea``,
        :data:`STAGE_S12_WITHHELD_FOOTPRINT` a ``GrossArea`` the footprints
        carry, :data:`STAGE_S13_WITHHELD` the join's ``ReplacementCost`` where
        no local value outranks it, :data:`STAGE_S14_WITHHELD` NSI's
        ``BuildingType`` and :data:`STAGE_S14_WITHHELD_FOOTPRINT` a
        ``BuildingType`` the footprints carry. One building's ``GrossArea``
        of two sources needs two records, since a record withholds a feature
        from one source. The run calls this after the view
        :data:`STAGE_S08_VIEW` is frozen and before the first merge of s11, so
        that the unit budget, the structure type draw and the rates of s13
        read these values.

        Args:
            sources (_WorldSources):
                The sources, to which the six records are added.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
        """
        local = [
            (asset_id, asset.features)
            for stage in sources.parts['local']
            for asset_id, asset in sources.freeze.read(stage).inventory.items()
        ]
        kept = sources.freeze.read(STAGE_S08_VIEW).inventory.keys() - {
            asset_id
            for asset_id, features in local
            if not is_missing_value(features.get(REPLACEMENT_COST_KEY))
        }
        records: dict[str, dict[str, dict[str, str]]] = {
            STAGE_S12_WITHHELD_LOCAL: {
                asset_id: {GROSS_AREA_KEY: 'local'}
                for asset_id, features in local
                if not is_missing_value(features.get(GROSS_AREA_KEY))
            },
            STAGE_S12_WITHHELD_JOIN: {},
            STAGE_S12_WITHHELD_FOOTPRINT: {},
            STAGE_S13_WITHHELD: {},
            STAGE_S14_WITHHELD: {},
            STAGE_S14_WITHHELD_FOOTPRINT: {},
        }
        # Footprints from BRAILS's scrapers carry neither feature, but a
        # caller's footprints may.
        for stage in sources.parts[SPINE_LABEL]:
            for asset_id, asset in sources.freeze.read(stage).inventory.items():
                for key, record in (
                    (GROSS_AREA_KEY, STAGE_S12_WITHHELD_FOOTPRINT),
                    (BUILDING_TYPE_KEY, STAGE_S14_WITHHELD_FOOTPRINT),
                ):
                    if not is_missing_value(asset.features.get(key)):
                        records[record][asset_id] = {key: SPINE_LABEL}
        for label, stage in (
            ('nsi', STAGE_S02_NSI_JOIN),
            ('derived', STAGE_S02_DERIVED),
        ):
            for asset_id, asset in sources.freeze.read(stage).inventory.items():
                features = asset.features
                if not is_missing_value(features.get(GROSS_AREA_KEY)):
                    records[STAGE_S12_WITHHELD_JOIN][asset_id] = {
                        GROSS_AREA_KEY: label
                    }
                if asset_id in kept and not is_missing_value(
                    features.get(REPLACEMENT_COST_KEY)
                ):
                    records[STAGE_S13_WITHHELD][asset_id] = {
                        REPLACEMENT_COST_KEY: label
                    }
                # The join puts a BuildingType in its observed part only; the
                # derived part is read in case a join ever puts one there.
                if not is_missing_value(features.get(BUILDING_TYPE_KEY)):
                    records[STAGE_S14_WITHHELD][asset_id] = {
                        BUILDING_TYPE_KEY: label
                    }
        for stage, withheld in records.items():
            sources.add_withheld(
                stage, _build_inventory(withheld, centroids, withheld)
            )

    def _check_schema(
        self,
        world: int,
        merged: AssetInventory,
        excluded: set[str],
        report: CompilationReport,
    ) -> pd.DataFrame:
        """
        Run s11 in a world: check buildings not excluded against the schema.

        The schema reads none of ``GrossArea``, ``ReplacementCost`` and
        ``BuildingType``, which s12 to s14 give after the check.

        Args:
            world (int):
                The index of the world the merged inventory holds, which
                the report records.
            merged (AssetInventory):
                The world's merged inventory.
            excluded (set[str]):
                The buildings excluded from the world, to which those the
                schema refuses are added.
            report (CompilationReport):
                The report, which receives an exclusion for each building the
                schema refuses and the step's time in the world.

        Returns:
            pandas.DataFrame:
                The schema's violations in the world, with a ``world``
                column.
        """
        started = time.perf_counter()
        checked = _select_assets(
            merged, (a for a in merged.inventory if a not in excluded)
        )
        verdict = self._validator.validate(checked)
        for asset_id, messages in verdict.by_asset().items():
            report.record_exclusion(
                asset_id,
                world=world,
                stage='s11_schema_check',
                category=FAILS_SCHEMA_CATEGORY,
                diagnosis=DIAGNOSIS_INVALID,
                detail='; '.join(messages),
            )
            excluded.add(asset_id)
        report.set_timing(
            's11_schema_check', time.perf_counter() - started, world=world
        )
        return verdict.to_dataframe().drop(columns='world_index').assign(world=world)

    def _impute_replacement_costs(
        self,
        world: int,
        sources: _WorldSources,
        centroids: Mapping[str, list[list[float]]],
        imputer: ObservedDonorImputer,
        excluded: set[str],
        areas: AssetInventory,
        report: CompilationReport,
    ) -> tuple[AssetInventory, pd.DataFrame]:
        """
        Run s13 in one world: the value of every building.

        Every building of the view :data:`STAGE_S08_VIEW` is valued, excluded
        ones included, at its gross area of s12 in the world. The section
        "The run" of the README beside this module states the procedure.

        Args:
            world (int):
                The index of the world valued.
            sources (_WorldSources):
                The sources, whose freeze holds the view and the donor stage.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            imputer (ObservedDonorImputer):
                The imputer of s05, whose ``k``, ``cap_m`` and distance
                weights the value imputer takes.
            excluded (set[str]):
                The buildings excluded from the world, to which those left
                without a value are added.
            areas (AssetInventory):
                The gross areas of s12 in the world.
            report (CompilationReport):
                The report, which receives an exclusion for each building
                left without a value that is not yet excluded, and the step's
                time in the world.

        Returns:
            tuple[AssetInventory, pandas.DataFrame]:
                One asset per building of the view, holding its value and the
                value's source label where the value is not a kept local one,
                and the value imputer's log of the world.

        Raises:
            ValueError:
                As :meth:`ReplacementCostImputer.impute_stage` raises.
        """
        started = time.perf_counter()
        valuer = ReplacementCostImputer(
            seed=self.seed,
            k=imputer.k,
            cap_m=imputer.cap_m,
            location_scale_m=imputer.location_scale_m,
            area_weight=imputer.area_weight,
            stories_weight=imputer.stories_weight,
            # A rate divides a value, NSI's, the join's sum of NSI's or a local
            # one, by NSI's sqft, never by a local GrossArea.
            gross_area_key=NSI_GROSS_AREA_KEY,
        )
        result = valuer.impute_stage(
            sources.freeze,
            STAGE_S08_VIEW,
            STAGE_S05_DONORS,
            world=world,
            target_area_key=GROSS_AREA_KEY,
            areas=areas,
        )
        values: dict[str, dict[str, Any]] = {}
        for asset_id, outcome in zip(result.log['asset_id'], result.log['outcome']):
            code = OUTCOME_VALUE_CODES.get(outcome)
            if code is None and asset_id not in excluded:
                # The three outcomes the value imputer gives without a value.
                # A building without an occupancy class fails the schema check
                # first, and every footprint encloses an area, so only
                # no_donor is expected here.
                category, diagnosis = {
                    OUTCOME_NO_CLASS: (MISSING_INPUT_CATEGORY, DIAGNOSIS_INVALID),
                    OUTCOME_NO_AREA: (MISSING_INPUT_CATEGORY, DIAGNOSIS_INVALID),
                    OUTCOME_NO_DONOR: (
                        NO_PLAUSIBLE_VALUE_CATEGORY,
                        DIAGNOSIS_NO_DONOR,
                    ),
                }.get(
                    outcome,
                    # The imputer gives no other outcome without a value. One
                    # would say neither that an input is missing nor that the
                    # donor search found none, only that the imputer, with its
                    # inputs, gave no value, which infeasible states.
                    (NO_PLAUSIBLE_VALUE_CATEGORY, DIAGNOSIS_INFEASIBLE),
                )
                report.record_exclusion(
                    asset_id,
                    world=world,
                    stage='s13_replacement_cost',
                    category=category,
                    diagnosis=diagnosis,
                    detail=f'the value step gives no {REPLACEMENT_COST_KEY}: '
                    f'{outcome}',
                )
                excluded.add(asset_id)
            # A local value is already in the merged inventory and is left alone.
            elif code is not None and code != SOURCE_CODES['local']:
                values[asset_id] = {
                    REPLACEMENT_COST_KEY: result.costs.inventory[asset_id].features[
                        REPLACEMENT_COST_KEY
                    ],
                    _VALUE_SOURCE_KEY: SOURCE_LABELS[code],
                }
        kept = list(sources.freeze.read(STAGE_S08_VIEW).inventory)
        report.set_timing(
            STAGE_S13_REPLACEMENT_COST, time.perf_counter() - started, world=world
        )
        return (
            _build_inventory(kept, centroids, values, vector_features=()),
            result.log,
        )

    def _derive_building_types(
        self,
        world: int,
        merged: AssetInventory,
        sources: _WorldSources,
        centroids: Mapping[str, list[list[float]]],
        excluded: set[str],
        report: CompilationReport,
    ) -> AssetInventory:
        """
        Run s14 in one world: the letter of each building's structure type.

        Every building without a local ``BuildingType`` takes the letter of
        its structure type in the world, excluded ones included, by the
        pairing of ``get_hazus_general_type_membership`` that the structure
        type inferer draws with. A local letter is kept. A building not yet
        excluded whose local letter differs from the letter of its structure
        type is excluded from the world as a ``conflict``, since its local
        inventory contradicts itself.

        Args:
            world (int):
                The index of the world the merged inventory holds.
            merged (AssetInventory):
                The world's merged inventory.
            sources (_WorldSources):
                The sources, whose local parts say which building has a local
                letter.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid, which the assets built here take as
                their coordinates.
            excluded (set[str]):
                The buildings excluded from the world, to which those whose
                letters contradict each other are added.
            report (CompilationReport):
                The report, which receives an exclusion for each such
                building and the step's time in the world.

        Returns:
            AssetInventory:
                One asset per building of the merged inventory, holding the
                letter where the step gives one.
        """
        started = time.perf_counter()
        local_letters = {
            asset_id
            for stage in sources.parts['local']
            for asset_id, asset in sources.freeze.read(stage).inventory.items()
            if not is_missing_value(asset.features.get(BUILDING_TYPE_KEY))
        }
        letters: dict[str, dict[str, Any]] = {}
        for asset_id, asset in merged.inventory.items():
            structure_type = asset.features.get(STRUCTURE_TYPE_KEY)
            letter = _LETTER_OF_TYPE.get(structure_type)
            if asset_id not in local_letters:
                if letter is not None:
                    letters[asset_id] = {BUILDING_TYPE_KEY: letter}
            elif (
                asset_id not in excluded
                and letter is not None
                and asset.features[BUILDING_TYPE_KEY] != letter
            ):
                report.record_exclusion(
                    asset_id,
                    world=world,
                    stage='s14_building_type',
                    category=CONFLICT_CATEGORY,
                    diagnosis=DIAGNOSIS_INFEASIBLE,
                    detail=f'the local {BUILDING_TYPE_KEY} '
                    f'{asset.features[BUILDING_TYPE_KEY]!r} is not the letter '
                    f'{letter!r} of the {STRUCTURE_TYPE_KEY} {structure_type!r}',
                    # Without the local letter, s14 writes the type's letter.
                    resolvable_by=(BUILDING_TYPE_KEY,),
                )
                excluded.add(asset_id)
        report.set_timing(
            STAGE_S14_BUILDING_TYPE, time.perf_counter() - started, world=world
        )
        return _build_inventory(
            list(merged.inventory), centroids, letters, vector_features=()
        )

    def _assemble_output(
        self,
        world: int,
        merged: AssetInventory,
        excluded: set[str],
        areas: AssetInventory,
        costs: AssetInventory,
        letters: AssetInventory,
        report: CompilationReport,
    ) -> list[tuple[str, int, str]]:
        """
        Run s15 in one world: write Excluded and the results of s12 to s14.

        Every building receives ``Excluded`` and the letter of s14. A
        building not excluded from the world also receives its value of s13,
        coded by its source label, and its gross area of s12 as
        ``GrossArea``, and is checked for the properties of
        :data:`DEFAULT_PROPERTIES` that apply to it.

        Args:
            world (int):
                The index of the world the merged inventory holds.
            merged (AssetInventory):
                The world's merged inventory, changed in place.
            excluded (set[str]):
                The buildings excluded from the world.
            areas (AssetInventory):
                The gross areas of s12 in the world.
            costs (AssetInventory):
                The values of s13 in the world.
            letters (AssetInventory):
                The letters of s14 in the world.
            report (CompilationReport):
                The report, which receives the step's time in the world.

        Returns:
            list[tuple[str, int, str]]:
                The (asset id, world, property) of each default property a
                compiled building lacks where it applies.

        Raises:
            ValueError:
                If a source string is not one that
                :func:`~brails.types.asset_inventory.read_source_string`
                reads.
        """
        started = time.perf_counter()
        lacking = []
        for asset_id, asset in merged.inventory.items():
            features = asset.features
            features[EXCLUDED_KEY] = asset_id in excluded
            features.update(letters.inventory[asset_id].features)
            if asset_id in excluded:
                continue
            value = costs.inventory[asset_id].features
            if not is_missing_value(value.get(REPLACEMENT_COST_KEY)):
                _write_coded_values(
                    asset_id,
                    features,
                    {REPLACEMENT_COST_KEY: value[REPLACEMENT_COST_KEY]},
                    value[_VALUE_SOURCE_KEY],
                )
            area = areas.inventory[asset_id].features.get(GROSS_AREA_KEY)
            if area is not None:
                features[GROSS_AREA_KEY] = area
            lacking.extend(
                (asset_id, world, key)
                for key in DEFAULT_PROPERTIES
                if is_missing_value(features.get(key))
                and not (
                    key == UNITS_KEY
                    and features.get(OCCUPANCY_KEY) not in UNIT_COUNT_CLASSES
                )
                and not (
                    key == HEIGHT_CLASS_KEY
                    and features.get(STRUCTURE_TYPE_KEY) in _WITHOUT_HEIGHT_CLASS
                )
            )
        report.set_timing(
            STAGE_S15_OUTPUT, time.perf_counter() - started, world=world
        )
        return lacking
