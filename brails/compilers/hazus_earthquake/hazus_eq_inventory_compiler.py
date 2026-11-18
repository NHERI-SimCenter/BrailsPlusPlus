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
import operator
import secrets
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd

from brails.aggregators.points_to_polygons.nsi_footprint_join import NSIFootprintJoin
from brails.aggregators.stacked_footprints.stacked_footprints import (
    RULE_COPY,
    RULE_STACKED,
    StackedFootprintMerge,
)
from brails.compilers.compilation_report import DROPPED_CATEGORY, CompilationReport
from brails.compilers.hazus_earthquake.pair_redraw import PairRedraw
from brails.compilers.hazus_earthquake.resolution_loop import (
    ComposedScreen,
    ResolutionLoop,
)
from brails.compilers.world_assembly import (
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
    UNITS_KEY,
    YEAR_BUILT_KEY,
)
from brails.imputers.observed_donor_imputer.donor_records import OUTCOME_NO_DONOR
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    ObservedDonorImputer,
    imputation_donors_from_log,
)
from brails.imputers.replacement_cost.replacement_cost_imputer import (
    DONOR_STAGE,
    OUTCOME_VALUE_CODES,
    ReplacementCostImputer,
    footprint_gross_area,
)
from brails.imputers.residential_neighbor_year_sampler.residential_neighbor_year_sampler import (
    ResidentialNeighborYearSampler,
)
from brails.inferers.acs_unit_budget.acs_unit_budget import (
    DROP_PROBABILITY,
    GAP_NO_ACS,
    ACSUnitBudget,
    UnitBudgetResult,
)
from brails.inferers.acs_year_built_sampler.acs_year_built_sampler import (
    ACSYearBuiltSampler,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
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
    get_hazus_general_type_membership,
    get_hazus_height_classes_by_type,
)
from brails.inferers.hazus_inferer_earthquake.hazus_table_completion import (
    is_implausible,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
)
from brails.inferers.stories_from_height_inferer.stories_from_height_inferer import (
    StoriesFromHeightInferer,
    parse_sources,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    OVERTURE_PROVIDER,
    SOURCES_KEY,
)
from brails.scrapers.us_census_scrapers.census_tract_scraper import (
    LEVEL_BLOCK_GROUP,
    CensusTractScraper,
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
from brails.utils.stage_freeze import StageFreeze
from brails.utils.unit_converter import UNIT_TO_METER
from brails.utils.unit_count_rules import UNIT_COUNT_CLASSES
from brails.validators.schema_validator import SchemaValidator

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    import geopandas as gpd

# The number of possible worlds compiled by default.
DEFAULT_N_WORLDS = 10

# The source labels of a world's merge, highest precedence first.
PRECEDENCE = ('local', 'footprint', 'nsi', 'derived', 'imputed', 'drawn')

# The label of the footprints, the spine of every merge.
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

# The frozen stages, in the order they are frozen.
STAGE_FOOTPRINTS = 'footprints'
STAGE_LOCAL = 'local'
STAGE_JOIN_NSI = 'join_nsi'
STAGE_JOIN_DERIVED = 'join_derived'
STAGE_S03_STORY_COUNTS = 's03_story_counts'
STAGE_S03_WITHHELD = 's03_withheld'
STAGE_S04_UNIT_BUDGET = 's04_unit_budget'
STAGE_S04_DROPPED = 's04_dropped'
STAGE_S04_WITHHELD = 's04_withheld'
STAGE_S05_IMPUTATION = 's05_imputation'
STAGE_STEP4 = 'step4'
STAGE_S06_RESIDENTIAL_YEAR = 's06_residential_year'
STAGE_S07_WITHHELD = 's07_withheld'
STAGE_S07_PAIR_REDRAW = 's07_pair_redraw'
STAGE_S07_DERIVED = 's07_derived'
STAGE_STEP6_DRAWN = 'step6_drawn'
STAGE_STEP6_DERIVED = 'step6_derived'
STAGE_STEP6 = 'step6'
STAGE_STEP7_IMPUTED = 'step7_imputed'
STAGE_STEP7_DRAWN = 'step7_drawn'
STAGE_STEP7_DERIVED = 'step7_derived'
STAGE_S11_GROSS_AREA = 's11_gross_area'
STAGE_S11_WITHHELD_LOCAL = 's11_withheld_local'
STAGE_S11_WITHHELD_JOIN = 's11_withheld_join'
STAGE_S11_WITHHELD_FOOTPRINT = 's11_withheld_footprint'
STAGE_S13_WITHHELD = 's13_withheld'
STAGE_S14_WITHHELD = 's14_withheld'
STAGE_S14_WITHHELD_FOOTPRINT = 's14_withheld_footprint'
STAGE_S13_REPLACEMENT_COST = 's13_replacement_cost'
STAGE_S14_BUILDING_TYPE = 's14_building_type'
STAGE_S15_OUTPUT = 's15_output'

# The stages the report records actions and exclusions under.
REPORT_STAGES = (
    'footprints',
    'join',
    's03_story_counts',
    's04_unit_budget',
    's05_imputation',
    's06_residential_year',
    'design_level',
    's07_pair_redraw',
    'screen',
    'resolution',
    'structure_type',
    'height_class',
    'carve_out',
    'nonresidential_year',
    'value',
    'certificate',
)

# The categories the run files an exclusion under, besides the report's
# DROPPED_CATEGORY. The section "Exclusion categories" of the compilers'
# README defines each.
EXCLUSION_CATEGORIES = (
    'missing_input',
    'unsupported',
    'conflict',
    'no_plausible_value',
    'undetermined',
    'fails_schema',
)

# The Hazus input schema every compiled realization is checked against.
DLML_DATASET = 'seismic/building/portfolio/Hazus v5.1'

# The state and design level method whose table assumes Moderate-Code for a
# Pre-Code W1 building (Table C-10 of the Hazus 5.1 technical manual).
CARVE_OUT_STATE = 'California'
CARVE_OUT_METHOD = '5.1'

# The exclusion category of each diagnosis by which a year sampler, the
# design level inferer or the height class inferer refuses an asset. An asset
# ACSYearBuiltSampler finds infeasible lies in a block group the census tables
# do not cover, and one ResidentialNeighborYearSampler finds infeasible has no
# residential neighbor with a construction year.
_DIAGNOSIS_CATEGORY = {
    DIAGNOSIS_INVALID: 'missing_input',
    DIAGNOSIS_UNSUPPORTED: 'unsupported',
    DIAGNOSIS_INFEASIBLE: 'missing_input',
}

# The feature of the record of s13's values holding the source label of each
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

# Position of each feature in the source string.
_CODE_POSITION = {
    feature: position for position, feature in enumerate(HAZUS_EQ_SOURCE_FEATURES)
}

# The sources whose occupancy class makes a footprint reached, and those of
# them whose occupancy class the unit budget's mobile-home rule may relabel.
_REACHED_SOURCES = ('local', 'nsi', 'derived')
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
        block_groups (geopandas.GeoDataFrame):
            One polygon per census block group, with a ``GEOID`` column, as
            a TIGER/Line file holds them. Every footprint must lie in one.
        b25024_block_groups (pandas.DataFrame):
            Table B25024 per block group, as
            :class:`~brails.inferers.acs_unit_budget.acs_unit_budget.ACSUnitBudget`
            takes it.
        b25034_block_groups (pandas.DataFrame):
            Table B25034 per block group, as
            :class:`~brails.inferers.acs_year_built_sampler.acs_year_built_sampler.ACSYearBuiltSampler`
            takes it.
        b25034_tracts (pandas.DataFrame):
            Table B25034 per tract, in the same form.
        b25127_tracts (pandas.DataFrame):
            Table B25127 per tract, in the same form.
        mh_park_points (object):
            The mobile-home park points, in any form
            :class:`~brails.inferers.acs_unit_budget.acs_unit_budget.ACSUnitBudget`
            accepts for ``mh_park_points``.
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
    """

    footprints: AssetInventory
    nsi_points: AssetInventory
    block_groups: gpd.GeoDataFrame
    b25024_block_groups: pd.DataFrame
    b25034_block_groups: pd.DataFrame
    b25034_tracts: pd.DataFrame
    b25127_tracts: pd.DataFrame
    mh_park_points: object
    footprint_release: str | None = None
    local: AssetInventory | None = None
    study_area: str = ''
    height_unit: str = DEFAULT_UNITS['length']


class CompilationResult(NamedTuple):
    """
    The compiled inventory, with the record of how it was compiled.

    Attributes:
        inventory (AssetInventory):
            Every footprint the run kept, over the worlds 0 to
            ``n_worlds - 1``, with every feature the run gave it. A feature
            equal in every world is held as a scalar and otherwise as a list
            in world order. Each asset holds
            :data:`~brails.constants.EXCLUDED_KEY`, true in a world it was
            excluded from. A run that keeps no footprint gives an inventory
            without assets, which holds one world. :meth:`write` still
            writes the report rows of every world the run computed.
        report (CompilationReport):
            The ledger, the exclusions, the timings and the summary.
        freeze (StageFreeze):
            Every frozen stage, in freeze order.
        manifest (dict):
            The freeze's manifest at the end of the run.
        logs (Mapping[str, pandas.DataFrame]):
            The components' logs, by name.
        seed (int):
            The seed every component drew from.
    """

    inventory: AssetInventory
    report: CompilationReport
    freeze: StageFreeze
    manifest: dict
    logs: Mapping[str, pd.DataFrame]
    seed: int

    def write(  # noqa: C901
        self,
        path: str | os.PathLike[str],
        *,
        world: int | None = None,
        integer_ids: bool = False,
        keep_partly_excluded: bool = False,
        extra_properties: str | Sequence[str] = (),
    ) -> None:
        """
        Write every world, or one, as GeoJSON, with the report, manifest and logs.

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
        numbers the features, and its asset id as ``unique_id``. Beside it,
        under the stem of ``path``: the report through
        :meth:`CompilationReport.save` as ``<stem>_report.parquet``, the
        manifest through :meth:`StageFreeze.save_manifest` as
        ``<stem>_manifest.json``, and the logs through
        :meth:`CompilationReport.save_logs` as ``<stem>_<name>.parquet``,
        the report and the logs keeping the rows of the written worlds. The
        summary's notices, timings and certificate cover every world the run
        computed.

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
                Features of :attr:`inventory` to write beside the default
                properties, or :data:`ALL_PROPERTIES` for every feature it
                holds. They change no value of the default properties.

        Raises:
            TypeError:
                If ``world`` is a bool or not an integer, or
                ``extra_properties`` is a string other than
                :data:`ALL_PROPERTIES`; nothing is then written.
            ValueError:
                If ``world`` is not one of the worlds the run computed, or
                ``extra_properties`` names a feature no building of
                :attr:`inventory` holds or names
                :data:`~brails.constants.EXCLUDED_KEY`; nothing is then
                written.
            RuntimeError:
                If the log :data:`MISSING_DEFAULTS_LOG` names a default
                property that a compiled building lacks in a written world,
                once the report, the logs and the manifest are written and
                before the GeoJSON is; or if a frozen stage changed after it
                was frozen, in which case every file but the manifest has been
                written.
        """
        # The report holds the run's worlds and checks world numbers against
        # them. An inventory without assets holds one world, whatever the run
        # computed, so the worlds are not read from the inventory.
        worlds = (
            tuple(
                self.report._selected_worlds(None)  # noqa: SLF001
                or range(self.inventory.n_pw)
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
            key
            for asset in self.inventory.inventory.values()
            for key in asset.features
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
        if not lacking.empty:
            self.report.save(
                path.with_name(f'{path.stem}_report.parquet'), worlds=worlds
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
        groups = self.report.exclusion_groups(worlds=worlds)
        groups = groups[groups['group'] != 'dropped']
        left_out = set(groups.loc[groups['group'] == 'every', 'asset_id'])
        if not keep_partly_excluded:
            left_out |= set(groups.loc[groups['group'] == 'some', 'asset_id'])
        excluded_in = {
            w: set(rows) for w, rows in groups.groupby('world')['asset_id']
        }
        # Every world is written from the inventory as it is, unless a building
        # to be written has Excluded marks that differ from the report's
        # exclusion rows, as after a caller adds an exclusion to the report.
        # One world, and every world after such a change, is assembled from
        # the written worlds' realizations, each building marked from the
        # report.
        written = {
            asset_id: asset
            for asset_id, asset in self.inventory.inventory.items()
            if asset_id not in left_out
        }
        as_is = world is None
        for asset_id, asset in written.items() if as_is else ():
            marks = asset.features[EXCLUDED_KEY]
            listed = [asset_id in excluded_in.get(w, ()) for w in worlds]
            if (
                marks if isinstance(marks, list) else [marks] * len(listed)
            ) != listed:
                as_is = False
                break
        if not as_is:
            assembled = AssetInventory()
            for w in worlds:
                realization = self.inventory.get_world_realization(w)
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
                    # In one world, None marks a feature of another world.
                    and (world is None or value is not None)
                }
                for asset_id, asset in written.items()
            },
            n_pw=len(worlds),
            vector_features=self.inventory.vector_features,
        )
        inventory.write_to_geojson(str(path), integer_ids=integer_ids)
        self.report.save(
            path.with_name(f'{path.stem}_report.parquet'), worlds=worlds
        )
        self.report.save_logs(path.parent, path.stem, self.logs, worlds=worlds)
        self.freeze.save_manifest(path.with_name(f'{path.stem}_manifest.json'))


class _ImputationOutcome(NamedTuple):
    """
    What the imputation from observed donors hands to the steps after it.

    Attributes:
        imputer (ObservedDonorImputer):
            The imputer that ran, configured by the compiler's options. The
            pair re-draw draws with it, and the year draw of nonresidential
            buildings and the value step take its settings.
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

    A part is registered under the label of the source it belongs to, a
    withheld record names, per asset and feature, the source a value is
    withheld from, and the dropped record, named by :attr:`dropped`, holds
    the footprints removed from every source. :meth:`merge_world` builds one
    world's merge from them.
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
                The merge, with one possible world, its source string
                written. A feature whose value is ``None`` is left out, so
                the merge of a world holds the same features whatever the
                number of worlds.
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
    :data:`_REACHED_SOURCES` and that holds a story count.

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
            if occupancy_source[asset_id] in _REACHED_SOURCES
            and not is_missing_value(asset.features.get(STORIES_KEY))
        ],
        vector_features=vector_features,
    )


def _plausible(
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


def _overlay(
    asset_id: str, features: dict[str, Any], values: Mapping[str, Any], label: str
) -> None:
    """
    Write values from one source into a working merge, with their codes.

    Args:
        asset_id (str):
            The asset's id, which an error message names.
        features (dict[str, Any]):
            The asset's features in a merge, its source string included,
            changed in place.
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


class HazusEQInventoryCompiler:
    """
    Compile a Hazus earthquake inventory of one study area over several worlds.

    The section "Hazus earthquake inventory compiler" of the README beside
    this module describes the run, its frozen stages, its worlds and the
    written file.

    Every constructor argument is stored as an attribute of the same name,
    read-only by convention; ``seed`` holds the seed used, never ``None``,
    and the options hold the mappings as resolved.
    """

    def __init__(  # noqa: C901
        self,
        *,
        seed: int | None = None,
        n_worlds: int = DEFAULT_N_WORLDS,
        state: str = 'California',
        seismic_zone: str = '4',
        design_level_method: str = '5.1',
        join_cap_m: float = 15.0,
        projected_crs: object = None,
        imputer_options: Mapping[str, object] | None = None,
        budget_options: Mapping[str, object] | None = None,
        sampler_options: Mapping[str, object] | None = None,
        merge_stacked: bool = False,
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
                The full state name, as in ``'California'``, which every
                inferer that takes a state receives. Structure types are
                drawn from the completed West Coast tables, so the state
                must be one of that region with a RES1 distribution.
            seismic_zone (str, optional):
                The 1997 UBC seismic zone, which every inferer that takes
                one receives.
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
                Whether step 1 first merges the stacked footprints of one
                building with :class:`StackedFootprintMerge`, in the
                footprints' height unit, which re-keys them by geohash. The
                footprints must then all be Overture footprints with an
                ``overture_id``, and no local inventory is accepted.

        Raises:
            TypeError:
                If ``seed`` or ``n_worlds`` is not an integer,
                ``merge_stacked`` is not a boolean, an options argument is
                not a mapping, or an options mapping names an argument its
                class does not take or that the compiler sets.
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
        if not isinstance(merge_stacked, bool):
            msg = f'merge_stacked must be a boolean, not {merge_stacked!r}.'
            raise TypeError(msg)
        self.merge_stacked = merge_stacked

        try:
            # Built only to check the state. Each world builds its own.
            HazusEQStructureTypeInferer(state, seed=self.seed, complete_tables=True)
        except ValueError as error:
            msg = f'The compiler cannot draw structure types for {state}: {error}'
            raise ValueError(msg) from error
        # Built only to check the method, state and zone. Each world builds its own.
        HazusEQDesignLevelInferer(design_level_method, state, seismic_zone)
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
        self.sampler_options = MappingProxyType(resolved['sampler_options'])

    def compile(self, inputs: CompilerInputs) -> CompilationResult:
        """
        Compile the study area's building inventory over :attr:`n_worlds` worlds.

        The inputs are not modified.

        Args:
            inputs (CompilerInputs):
                The study area's data.

        Returns:
            CompilationResult:
                The compiled inventory and the record of the run.

        Raises:
            TypeError:
                If a footprint's asset id is not a string.
            ValueError:
                If the local inventory holds an asset that the footprints
                lack, a feature of :data:`RESERVED_LOCAL_FEATURES` or more
                than one possible world, or is given with ``merge_stacked``;
                or as the components raise, among them a footprint outside
                every block group and, with ``merge_stacked``, a footprint
                that is not Overture's.
            RuntimeError:
                If a frozen stage changed during the run.
        """
        not_strings = [
            a for a in inputs.footprints.inventory if not isinstance(a, str)
        ]
        if not_strings:
            msg = f'Footprint asset ids must be strings, not {not_strings[:10]!r}.'
            raise TypeError(msg)
        if inputs.local is not None:
            if self.merge_stacked:
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

        n_computed = self.n_worlds
        settings = {
            'study_area': inputs.study_area,
            'footprint_release': inputs.footprint_release,
            'height_unit': inputs.height_unit,
            'n_worlds': self.n_worlds,
            'worlds': list(range(n_computed)),
            'state': self.state,
            'seismic_zone': self.seismic_zone,
            'design_level_method': self.design_level_method,
            'join_cap_m': self.join_cap_m,
            'projected_crs': None
            if self.projected_crs is None
            else str(self.projected_crs),
            'imputer_options': dict(self.imputer_options),
            'budget_options': dict(self.budget_options),
            'sampler_options': dict(self.sampler_options),
            'merge_stacked': self.merge_stacked,
        }
        report = CompilationReport(
            self.seed,
            stages=REPORT_STAGES,
            exclusion_categories=EXCLUSION_CATEGORIES,
            settings=settings,
        )
        report.set_world_count(n_computed)
        freeze = StageFreeze()
        sources = _WorldSources(freeze)
        logs: dict[str, pd.DataFrame] = {}

        started = time.perf_counter()
        centroids = self._footprints(inputs, sources, logs, report)
        report.set_timing(STAGE_FOOTPRINTS, time.perf_counter() - started)

        started = time.perf_counter()
        self._join(inputs, sources, logs, centroids)
        report.set_timing('join', time.perf_counter() - started)

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
        self._view(STAGE_STEP4, sources, report, n_computed)
        sources.freeze.freeze(
            DONOR_STAGE,
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
        self._step6(sources, logs, centroids, report, n_computed)
        self._view(STAGE_STEP6, sources, report, n_computed)
        self._step7(sources, logs, centroids, report, n_computed, imputer)

        started = time.perf_counter()
        self._compute_gross_areas(sources, logs, centroids, n_computed)
        report.set_timing(STAGE_S11_GROSS_AREA, time.perf_counter() - started)
        self._withhold_replaced_values(sources, centroids)

        # Steps s12 to s15 run world by world on one merge of the world, which
        # s15 completes, and each world's results are added to those of the
        # earlier worlds.
        excluded_in = {
            world: set(ids)
            for world, ids in report.exclusions.groupby('world')['asset_id']
        }
        costs, letters, inventory = (
            AssetInventory(),
            AssetInventory(),
            AssetInventory(),
        )
        violations, value_logs, lacking = [], [], []
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
            world_costs, value_log = self._impute_replacement_costs(
                world, sources, centroids, imputer, excluded, report
            )
            world_letters = self._derive_building_types(
                world, merged, sources, centroids, excluded, report
            )
            lacking.extend(
                self._assemble_output(
                    world,
                    merged,
                    excluded,
                    world_costs,
                    world_letters,
                    sources,
                    report,
                )
            )
            costs.add_world(world_costs)
            letters.add_world(world_letters)
            inventory.add_world(merged)
            value_logs.append(value_log)
            del merged

        # s12: the schema's violations and the count of each world's refusals.
        logs['certificate'] = pd.concat(violations, ignore_index=True)
        report.set_certificate(
            {
                'dataset': DLML_DATASET,
                'excluded_by_world': {
                    world: int(frame['asset_id'].nunique())
                    for world, frame in enumerate(violations)
                },
            }
        )
        # s13: the value imputer's log of every world.
        logs['value'] = pd.concat(value_logs, ignore_index=True)
        # s15: the default properties compiled buildings lack.
        logs[MISSING_DEFAULTS_LOG] = pd.DataFrame(
            lacking, columns=['asset_id', 'world', 'property'], dtype=object
        )
        if lacking:
            report.note(
                f'{len(lacking)} default value(s) that apply to compiled '
                f'buildings are missing; the log {MISSING_DEFAULTS_LOG} names '
                'them, and write refuses the file.'
            )
        # The stages of s13, s14 and s15, over the worlds.
        for stage, assembled in (
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
        return CompilationResult(
            inventory=inventory,
            report=report,
            freeze=freeze,
            manifest=manifest,
            logs=MappingProxyType(logs),
            seed=self.seed,
        )

    def _footprints(
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        report: CompilationReport,
    ) -> dict[str, list[list[float]]]:
        """
        Run step 1: block groups and footprint features, frozen as the spine.

        With :attr:`merge_stacked`, the stacked footprints are merged first,
        and the rest of the step reads the merged set.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            sources (_WorldSources):
                The sources, to which the footprints and a local inventory
                are added as :data:`STAGE_FOOTPRINTS` and
                :data:`STAGE_LOCAL`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``footprint_ties`` and, with
                :attr:`merge_stacked`, the merge's parts table as
                ``stacked_parts`` and its id log as ``stacked_asset_ids``.
            report (CompilationReport):
                The report, whose notices count the merges.

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
        if self.merge_stacked:
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
            # A copy set's keeper is a head too, so the rule tells them apart.
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
        sources.add_part(SPINE_LABEL, STAGE_FOOTPRINTS, spine)
        if inputs.local is not None:
            local = deepcopy(inputs.local)
            for asset_id, asset in local.inventory.items():
                asset.coordinates = centroids[asset_id]
                # The local GrossArea also under a name of its own, which the
                # output keeps where the GrossArea of s11 takes its place.
                if not is_missing_value(asset.features.get(GROSS_AREA_KEY)):
                    asset.features[LOCAL_GROSS_AREA_KEY] = asset.features[
                        GROSS_AREA_KEY
                    ]
            sources.add_part('local', STAGE_LOCAL, local)
        return centroids

    def _join(
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
    ) -> None:
        """
        Run steps 2 and 3: join NSI points to the frozen footprints.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            sources (_WorldSources):
                The sources, to which the join's observed and derived values
                are added as :data:`STAGE_JOIN_NSI` and
                :data:`STAGE_JOIN_DERIVED`, with centroids as their
                coordinates.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive the join's links, unlinked points
                and branch counts.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid.
        """
        result = NSIFootprintJoin(
            cap_m=self.join_cap_m, seed=self.seed, projected_crs=self.projected_crs
        ).join(sources.freeze.read(STAGE_FOOTPRINTS), inputs.nsi_points)
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
        sources.add_part('nsi', STAGE_JOIN_NSI, result.observed)
        sources.add_part('derived', STAGE_JOIN_DERIVED, result.derived)

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
                Each footprint's centroid.
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
        Run s04: the unit budget, which also drops footprints left without units.

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
                Each footprint's centroid.
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
        # A footprint is reached where a source holds its occupancy class.
        reached = _select_assets(
            merged,
            (
                a
                for a, source in occupancy_source.items()
                if source in _REACHED_SOURCES
            ),
            vector_features=DEFAULT_VECTOR_FEATURES,
        )
        unreached = _select_assets(
            merged,
            (
                a
                for a, source in occupancy_source.items()
                if source not in _REACHED_SOURCES
            ),
            vector_features=DEFAULT_VECTOR_FEATURES,
        )
        relabel = {
            a: source
            for a, source in occupancy_source.items()
            if source in _RELABELED_SOURCES
        }
        budget = ACSUnitBudget(
            inputs.b25024_block_groups,
            inputs.mh_park_points,
            imputer=imputer,
            seed=self.seed,
            **self.budget_options,
        )
        allocation = budget.allocate(
            unreached, reached, n_worlds=n_computed, relabel_sources=relabel
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
                Each footprint's centroid.
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
                The allocation of s04.

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
            validate=_plausible,
        )
        imputer_log = imputation.log.copy()
        imputer_log['values'] = [
            json.dumps(dict(values), default=str, sort_keys=True)
            for values in imputer_log['values']
        ]
        logs['imputer'] = imputer_log
        # A footprint with an occupancy class from a source and outside the
        # residential classes is a target for its story count only, since the
        # join gives units to housing records alone.
        for asset_id, asset in imputation.imputed.inventory.items():
            class_of = merged.inventory[asset_id].features.get(OCCUPANCY_KEY)
            if (
                occupancy_source[asset_id] in _REACHED_SOURCES
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
                Each footprint's centroid.
            report (CompilationReport):
                The report, which receives the exclusions of every world.
            n_computed (int):
                The number of worlds computed.
        """
        view = sources.freeze.read(STAGE_STEP4)
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
        view :data:`STAGE_STEP4`. The section "The run" of the README beside
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
                Each footprint's centroid.
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
        view = sources.freeze.read(STAGE_STEP4)
        kept = list(view.inventory)
        stages = (STAGE_S07_WITHHELD, STAGE_S07_PAIR_REDRAW, STAGE_S07_DERIVED)
        assembled = {stage: AssetInventory() for stage in stages}
        redraws = []
        for world in range(n_computed):
            started = time.perf_counter()
            frozen = view.get_world_realization(world)
            working = sources.merge_world(world)
            new: dict[str, dict[str, dict[str, Any]]] = {
                stage: {} for stage in stages
            }
            # The re-draw draws from the observed donors of the frozen view.
            donors = _select_observed_donors(frozen, _read_occupancy_sources(frozen))
            redraw = PairRedraw(imputer, donors, stories, world=world).redraw(
                working
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
                            old=working.inventory[asset_id].features.get(feature),
                            new=value,
                            source=label,
                            reason=reason,
                        )
                    new[stage].setdefault(asset_id, {}).update(values)
                    _overlay(
                        asset_id, working.inventory[asset_id].features, values, label
                    )

            for stage in stages:
                assembled[stage].add_world(
                    _build_inventory(kept, centroids, new[stage], vector_features=())
                )
            del frozen, working
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

    def _step6(
        self,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
    ) -> None:
        """
        Give each building its design level, structure type and height class.

        This runs in every world and also applies the W1 carve-out. Each
        world's draws read that world merged after s07, and a building that
        an earlier step excluded from the world, today only s06, takes none. The section "The run"
        of the README beside this module states the procedure.

        Args:
            sources (_WorldSources):
                The sources, to which the parts are added.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``resolution``, with a ``world``
                column.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid.
            report (CompilationReport):
                The report, whose exclusions are read and which receives the
                actions and exclusions of every world.
            n_computed (int):
                The number of worlds computed.

        Raises:
            ValueError:
                If an asset's source string is not one
                :func:`~brails.types.asset_inventory.read_source_string`
                reads, or, as :meth:`ResolutionLoop.resolve` raises, a
                building type has no source.
        """
        kept = list(sources.freeze.read(STAGE_STEP4).inventory)
        exclusions = report.exclusions
        stages = (STAGE_STEP6_DRAWN, STAGE_STEP6_DERIVED)
        assembled = {stage: AssetInventory() for stage in stages}
        resolutions = []
        for world in range(n_computed):
            started = time.perf_counter()
            working = sources.merge_world(world)
            new: dict[str, dict[str, dict[str, Any]]] = {
                stage: {} for stage in stages
            }
            excluded = set(exclusions.loc[exclusions['world'] == world, 'asset_id'])
            resolutions.append(
                {
                    'world': world,
                    **self._draw_types(
                        world,
                        sources,
                        report,
                        working,
                        kept,
                        excluded,
                        new[STAGE_STEP6_DERIVED],
                        new[STAGE_STEP6_DRAWN],
                    ),
                }
            )
            for stage in stages:
                assembled[stage].add_world(
                    _build_inventory(kept, centroids, new[stage], vector_features=())
                )
            del working
            report.set_timing('step6', time.perf_counter() - started, world=world)

        logs['resolution'] = pd.DataFrame(resolutions)
        started = time.perf_counter()
        sources.add_part('drawn', STAGE_STEP6_DRAWN, assembled[STAGE_STEP6_DRAWN])
        sources.add_part(
            'derived', STAGE_STEP6_DERIVED, assembled[STAGE_STEP6_DERIVED]
        )
        report.set_timing('step6_parts_freeze', time.perf_counter() - started)

    def _draw_types(  # noqa: C901
        self,
        world: int,
        sources: _WorldSources,
        report: CompilationReport,
        working: AssetInventory,
        candidates: Sequence[str],
        excluded: set[str],
        derived: dict[str, dict[str, Any]],
        drawn: dict[str, dict[str, Any]],
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
            working (AssetInventory):
                The world's working merge, whose design levels are written.
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
            self.design_level_method, self.state, self.seismic_zone
        )
        heights = HazusEQHeightClassInferer()
        labeled = [
            (label, sources.freeze.read(stage))
            for label in PRECEDENCE
            for stage in sources.parts[label]
        ]

        def held(asset_id: str, key: str) -> bool:
            """
            Say whether an asset holds a value of a feature in the working merge.

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
                working.inventory[asset_id].features.get(key)
            )

        # Derive a design level for each asset with a year and no design level.
        dated = [
            a
            for a in candidates
            if a not in excluded
            and held(a, YEAR_BUILT_KEY)
            and not held(a, DESIGN_LEVEL_KEY)
        ]
        checked = design.check(_select_assets(working, dated))
        for row in checked[checked['diagnosis'] != DIAGNOSIS_OK].itertuples():
            excluded.add(row.Index)
            report.record_exclusion(
                row.Index,
                world=world,
                stage='design_level',
                category=_DIAGNOSIS_CATEGORY[row.diagnosis],
                diagnosis=row.diagnosis,
                detail=row.detail,
                resolvable_by=row.resolvable_by,
            )
        usable = list(checked.index[checked['diagnosis'] == DIAGNOSIS_OK])
        for asset_id, level in design.draw(_select_assets(working, usable)).items():
            derived.setdefault(asset_id, {})[DESIGN_LEVEL_KEY] = level
            _overlay(
                asset_id,
                working.inventory[asset_id].features,
                {DESIGN_LEVEL_KEY: level},
                'derived',
            )

        # The screen and the loop, then the type draw.
        structure = HazusEQStructureTypeInferer(
            self.state, seed=self.seed, world=world, complete_tables=True
        )
        loop = ResolutionLoop(ComposedScreen(design, structure))
        untyped = [
            a
            for a in candidates
            if a not in excluded
            and held(a, YEAR_BUILT_KEY)
            and not held(a, STRUCTURE_TYPE_KEY)
        ]
        # The source holding each building type, in merge precedence.
        type_sources = {}
        for asset_id in untyped:
            if not held(asset_id, BUILDING_TYPE_KEY):
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
            _select_assets(working, untyped),
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
            if structure_type is None and held(asset_id, STRUCTURE_TYPE_KEY):
                structure_type = working.inventory[asset_id].features[
                    STRUCTURE_TYPE_KEY
                ]
            if (
                asset_id in excluded
                or structure_type is None
                or held(asset_id, HEIGHT_CLASS_KEY)
            ):
                continue
            height_class_inputs[asset_id] = {
                STRUCTURE_TYPE_KEY: structure_type,
                STORIES_KEY: working.inventory[asset_id].features.get(STORIES_KEY),
            }
        typed = _build_inventory(
            height_class_inputs,
            {a: working.inventory[a].coordinates for a in height_class_inputs},
            height_class_inputs,
        )
        classed = heights.check(typed)
        for row in classed[classed['diagnosis'] != DIAGNOSIS_OK].itertuples():
            excluded.add(row.Index)
            report.record_exclusion(
                row.Index,
                world=world,
                stage='height_class',
                category=_DIAGNOSIS_CATEGORY[row.diagnosis],
                diagnosis=row.diagnosis,
                detail=row.detail,
            )
        ok = list(classed.index[classed['diagnosis'] == DIAGNOSIS_OK])
        for asset_id, height_class in heights.draw(
            _select_assets(typed, ok)
        ).items():
            if height_class is not None:
                derived.setdefault(asset_id, {})[HEIGHT_CLASS_KEY] = height_class

        if self.state == CARVE_OUT_STATE and design.method == CARVE_OUT_METHOD:
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
                    stage='carve_out',
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

    def _step7(
        self,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
        imputer: ObservedDonorImputer,
    ) -> None:
        """
        Run step 7 in every world: the year of each building left without one.

        The section "The run" of the README beside this module states the
        procedure.

        Args:
            sources (_WorldSources):
                The sources, to which the parts are added as
                :data:`STAGE_STEP7_IMPUTED`, :data:`STAGE_STEP7_DRAWN` and
                :data:`STAGE_STEP7_DERIVED`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``neighbor_year`` and
                ``nonresidential_resolution``, each with a ``world`` column.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid.
            report (CompilationReport):
                The report, whose exclusions are read and which receives the
                actions and exclusions of every world.
            n_computed (int):
                The number of worlds computed.
            imputer (ObservedDonorImputer):
                The imputer of s05, whose ``k`` and ``cap_m`` the sampler
                takes.

        Raises:
            ValueError:
                If an asset's source string is not one
                :func:`~brails.types.asset_inventory.read_source_string`
                reads, or, as :meth:`ResolutionLoop.resolve` raises, a
                building type has no source.
        """
        view = sources.freeze.read(STAGE_STEP6)
        kept = list(view.inventory)
        exclusions = report.exclusions
        stages = (STAGE_STEP7_IMPUTED, STAGE_STEP7_DRAWN, STAGE_STEP7_DERIVED)
        assembled = {stage: AssetInventory() for stage in stages}
        picks, resolutions = [], []
        for world in range(n_computed):
            started = time.perf_counter()
            excluded = set(exclusions.loc[exclusions['world'] == world, 'asset_id'])
            frozen = view.get_world_realization(world)
            working = view.get_world_realization(world)
            new: dict[str, dict[str, dict[str, Any]]] = {
                stage: {} for stage in stages
            }
            sampler = ResidentialNeighborYearSampler(
                seed=self.seed,
                world=world,
                k=imputer.k,
                cap_m=imputer.cap_m,
            ).select_donors(sources.freeze, STAGE_STEP6)
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
                    stage='nonresidential_year',
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
                new[STAGE_STEP7_IMPUTED][asset_id] = {YEAR_BUILT_KEY: int(year)}
                _overlay(
                    asset_id,
                    working.inventory[asset_id].features,
                    {YEAR_BUILT_KEY: int(year)},
                    'imputed',
                )
            resolutions.append(
                {
                    'world': world,
                    **self._draw_types(
                        world,
                        sources,
                        report,
                        working,
                        passing,
                        excluded,
                        new[STAGE_STEP7_DERIVED],
                        new[STAGE_STEP7_DRAWN],
                    ),
                }
            )
            for stage in stages:
                assembled[stage].add_world(
                    _build_inventory(kept, centroids, new[stage], vector_features=())
                )
            del frozen, working
            report.set_timing('step7', time.perf_counter() - started, world=world)

        logs['neighbor_year'] = pd.concat(picks, ignore_index=True)
        logs['nonresidential_resolution'] = pd.DataFrame(resolutions)
        started = time.perf_counter()
        for label, stage in (
            ('imputed', STAGE_STEP7_IMPUTED),
            ('drawn', STAGE_STEP7_DRAWN),
            ('derived', STAGE_STEP7_DERIVED),
        ):
            sources.add_part(label, stage, assembled[stage])
        report.set_timing('step7_parts_freeze', time.perf_counter() - started)

    def _view(
        self,
        stage: str,
        sources: _WorldSources,
        report: CompilationReport,
        n_computed: int,
    ) -> None:
        """
        Freeze the merge of every registered source over the worlds.

        The report receives the time of each world's merge under
        ``f'{stage}_view'`` and that of the freeze under
        ``f'{stage}_freeze'``.

        Args:
            stage (str):
                The view's stage name.
            sources (_WorldSources):
                The sources.
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
                f'{stage}_view', time.perf_counter() - started, world=world
            )
        started = time.perf_counter()
        sources.freeze.freeze(stage, view)
        report.set_timing(f'{stage}_freeze', time.perf_counter() - started)

    def _compute_gross_areas(
        self,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        n_computed: int,
    ) -> None:
        """
        Run s11 in every world: the gross area of every building.

        The area is computed from the footprint in the view
        :data:`STAGE_STEP6`, whose story counts are final, for every building
        of the view, excluded ones included. The section "The run" of the
        README beside this module states the procedure.

        Args:
            sources (_WorldSources):
                The sources, in whose freeze the areas are frozen as
                :data:`STAGE_S11_GROSS_AREA`, a record that s13 and s15 read
                and no merge takes.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``gross_area``: per building and
                world, the ``area_source`` that ``footprint_gross_area``
                gives.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid.
            n_computed (int):
                The number of worlds computed.

        Raises:
            ValueError:
                As :func:`footprint_gross_area` raises, for a building
                without a readable footprint.
        """
        view = sources.freeze.read(STAGE_STEP6)
        kept = list(view.inventory)
        areas = AssetInventory()
        rows = []
        for world in range(n_computed):
            realization = view.get_world_realization(world)
            values = {}
            for asset_id, asset in realization.inventory.items():
                area, area_source = footprint_gross_area(asset_id, asset)
                rows.append((asset_id, world, area_source))
                if area is not None:
                    values[asset_id] = {GROSS_AREA_KEY: area}
            areas.add_world(
                _build_inventory(kept, centroids, values, vector_features=())
            )
            del realization
        logs['gross_area'] = pd.DataFrame(
            rows, columns=['asset_id', 'world', 'area_source']
        ).astype({'asset_id': object, 'area_source': 'category'})
        sources.freeze.freeze(STAGE_S11_GROSS_AREA, areas)

    def _withhold_replaced_values(  # noqa: C901
        self, sources: _WorldSources, centroids: Mapping[str, list[list[float]]]
    ) -> None:
        """
        Withhold from every later merge the values that s11, s13 and s14 replace.

        Each record is named by the step whose value takes the place of the
        values it withholds, and each names, per building, the source a value
        is withheld from: :data:`STAGE_S11_WITHHELD_LOCAL` the local
        ``GrossArea``, :data:`STAGE_S11_WITHHELD_JOIN` NSI's ``GrossArea``,
        :data:`STAGE_S11_WITHHELD_FOOTPRINT` a ``GrossArea`` the footprints
        carry, :data:`STAGE_S13_WITHHELD` the join's ``ReplacementCost`` where
        no local value outranks it, :data:`STAGE_S14_WITHHELD` NSI's
        ``BuildingType`` and :data:`STAGE_S14_WITHHELD_FOOTPRINT` a
        ``BuildingType`` the footprints carry. One building's ``GrossArea``
        of two sources needs two records, since a record withholds a feature
        from one source. The run calls this after the view
        :data:`STAGE_STEP6` is frozen and before the first merge of s12, so
        that the unit budget, the structure type draw and the rates of s13
        read these values.

        Args:
            sources (_WorldSources):
                The sources, to which the six records are added.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid.
        """
        local = [
            (asset_id, asset.features)
            for stage in sources.parts['local']
            for asset_id, asset in sources.freeze.read(stage).inventory.items()
        ]
        kept = sources.freeze.read(STAGE_STEP6).inventory.keys() - {
            asset_id
            for asset_id, features in local
            if not is_missing_value(features.get(REPLACEMENT_COST_KEY))
        }
        records: dict[str, dict[str, dict[str, str]]] = {
            STAGE_S11_WITHHELD_LOCAL: {
                asset_id: {GROSS_AREA_KEY: 'local'}
                for asset_id, features in local
                if not is_missing_value(features.get(GROSS_AREA_KEY))
            },
            STAGE_S11_WITHHELD_JOIN: {},
            STAGE_S11_WITHHELD_FOOTPRINT: {},
            STAGE_S13_WITHHELD: {},
            STAGE_S14_WITHHELD: {},
            STAGE_S14_WITHHELD_FOOTPRINT: {},
        }
        # Footprints from BRAILS's scrapers carry neither feature, but a
        # caller's footprints may.
        for stage in sources.parts[SPINE_LABEL]:
            for asset_id, asset in sources.freeze.read(stage).inventory.items():
                for key, record in (
                    (GROSS_AREA_KEY, STAGE_S11_WITHHELD_FOOTPRINT),
                    (BUILDING_TYPE_KEY, STAGE_S14_WITHHELD_FOOTPRINT),
                ):
                    if not is_missing_value(asset.features.get(key)):
                        records[record][asset_id] = {key: SPINE_LABEL}
        for label, stage in (
            ('nsi', STAGE_JOIN_NSI),
            ('derived', STAGE_JOIN_DERIVED),
        ):
            for asset_id, asset in sources.freeze.read(stage).inventory.items():
                features = asset.features
                if not is_missing_value(features.get(GROSS_AREA_KEY)):
                    records[STAGE_S11_WITHHELD_JOIN][asset_id] = {
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
        Run s12 in one world: check every building not yet excluded against the schema.

        The schema reads none of ``ReplacementCost``, ``GrossArea`` and
        ``BuildingType``, which s15 writes after the check.

        Args:
            world (int):
                The index of the world the merge holds, which the report
                records.
            merged (AssetInventory):
                The world's merge of every source.
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
                stage='certificate',
                category='fails_schema',
                detail='; '.join(messages),
            )
            excluded.add(asset_id)
        report.set_timing(
            's12_schema_check', time.perf_counter() - started, world=world
        )
        return verdict.to_dataframe().drop(columns='world_index').assign(world=world)

    def _impute_replacement_costs(
        self,
        world: int,
        sources: _WorldSources,
        centroids: Mapping[str, list[list[float]]],
        imputer: ObservedDonorImputer,
        excluded: set[str],
        report: CompilationReport,
    ) -> tuple[AssetInventory, pd.DataFrame]:
        """
        Run s13 in one world: the value of every building.

        Every building of the view :data:`STAGE_STEP6` is valued, excluded
        ones included, at the gross area of :data:`STAGE_S11_GROSS_AREA`.
        The section "The run" of the README beside this module states the
        procedure.

        Args:
            world (int):
                The index of the world valued.
            sources (_WorldSources):
                The sources, whose freeze holds the view, the donor stage and
                the areas.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid.
            imputer (ObservedDonorImputer):
                The imputer of s05, whose ``k``, ``cap_m`` and distance
                weights the value imputer takes.
            excluded (set[str]):
                The buildings excluded from the world, to which those left
                without a value are added.
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
            STAGE_STEP6,
            DONOR_STAGE,
            world=world,
            target_area_key=GROSS_AREA_KEY,
            area_stage=STAGE_S11_GROSS_AREA,
        )
        values: dict[str, dict[str, Any]] = {}
        for asset_id, outcome in zip(result.log['asset_id'], result.log['outcome']):
            code = OUTCOME_VALUE_CODES.get(outcome)
            if code is None and asset_id not in excluded:
                # A building without a donor has no plausible value. A
                # building without an occupancy class fails the schema check
                # first, and every footprint encloses an area, so no other
                # outcome without a value is expected here. One that occurs
                # is filed as a missing input rather than passed over.
                report.record_exclusion(
                    asset_id,
                    world=world,
                    stage='value',
                    category='no_plausible_value'
                    if outcome == OUTCOME_NO_DONOR
                    else 'missing_input',
                    detail=f'the value step gives no {REPLACEMENT_COST_KEY}: '
                    f'{outcome}',
                )
                excluded.add(asset_id)
            # A local value is the merge's own and is left alone.
            elif code is not None and code != SOURCE_CODES['local']:
                values[asset_id] = {
                    REPLACEMENT_COST_KEY: result.costs.inventory[asset_id].features[
                        REPLACEMENT_COST_KEY
                    ],
                    _VALUE_SOURCE_KEY: SOURCE_LABELS[code],
                }
        kept = list(sources.freeze.read(STAGE_STEP6).inventory)
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
        type inferer draws with. A local letter is kept, and a notice counts
        the buildings not excluded whose local letter differs from the
        letter of their structure type.

        Args:
            world (int):
                The index of the world the merge holds.
            merged (AssetInventory):
                The world's merge of every source.
            sources (_WorldSources):
                The sources, whose local parts say which building has a local
                letter.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid.
            excluded (set[str]):
                The buildings excluded from the world.
            report (CompilationReport):
                The report, which receives the notice and the step's time in
                the world.

        Returns:
            AssetInventory:
                One asset per building of the merge, holding the letter where
                the step gives one.
        """
        started = time.perf_counter()
        local_letters = {
            asset_id
            for stage in sources.parts['local']
            for asset_id, asset in sources.freeze.read(stage).inventory.items()
            if not is_missing_value(asset.features.get(BUILDING_TYPE_KEY))
        }
        letters: dict[str, dict[str, Any]] = {}
        contradicted = 0
        for asset_id, asset in merged.inventory.items():
            letter = _LETTER_OF_TYPE.get(asset.features.get(STRUCTURE_TYPE_KEY))
            if asset_id not in local_letters:
                if letter is not None:
                    letters[asset_id] = {BUILDING_TYPE_KEY: letter}
            elif (
                asset_id not in excluded
                and asset.features[BUILDING_TYPE_KEY] != letter
            ):
                contradicted += 1
        if contradicted:
            report.note(
                f'World {world}: {contradicted} compiled building(s) keep a '
                f'local {BUILDING_TYPE_KEY} other than the letter of their '
                f'{STRUCTURE_TYPE_KEY}.'
            )
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
        costs: AssetInventory,
        letters: AssetInventory,
        sources: _WorldSources,
        report: CompilationReport,
    ) -> list[tuple[str, int, str]]:
        """
        Run s15 in one world: write the last steps' results into the merge.

        Every building receives ``Excluded`` and the letter of s14. A
        building not excluded from the world also receives its value of s13,
        coded by its source label, and its gross area of s11 as
        ``GrossArea``, and is checked for the properties of
        :data:`DEFAULT_PROPERTIES` that apply to it.

        Args:
            world (int):
                The index of the world the merge holds.
            merged (AssetInventory):
                The world's merge of every source, changed in place.
            excluded (set[str]):
                The buildings excluded from the world.
            costs (AssetInventory):
                The values of s13 in the world.
            letters (AssetInventory):
                The letters of s14 in the world.
            sources (_WorldSources):
                The sources, whose freeze holds :data:`STAGE_S11_GROSS_AREA`.
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
        areas = (
            sources.freeze.read(STAGE_S11_GROSS_AREA)
            .get_world_realization(world)
            .inventory
        )
        lacking = []
        for asset_id, asset in merged.inventory.items():
            features = asset.features
            features[EXCLUDED_KEY] = asset_id in excluded
            features.update(letters.inventory[asset_id].features)
            if asset_id in excluded:
                continue
            value = costs.inventory[asset_id].features
            if not is_missing_value(value.get(REPLACEMENT_COST_KEY)):
                _overlay(
                    asset_id,
                    features,
                    {REPLACEMENT_COST_KEY: value[REPLACEMENT_COST_KEY]},
                    value[_VALUE_SOURCE_KEY],
                )
            area = areas[asset_id].features.get(GROSS_AREA_KEY)
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
