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
from brails.compilers.compilation_report import CompilationReport
from brails.compilers.world_assembly import WorldAssembler, compose_source
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
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    ObservedDonorImputer,
)
from brails.inferers.acs_unit_budget.acs_unit_budget import (
    DROP_PROBABILITY,
    GAP_NO_ACS,
    ACSUnitBudget,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
    HazusEQDesignLevelInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_structure_type_inferer import (
    HazusEQStructureTypeInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_table_completion import (
    is_implausible,
)
from brails.inferers.inference_diagnosis import DIAGNOSIS_OK
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
    HAZUS_EQ_SOURCE_FEATURES,
    SOURCE_CODES,
    SOURCE_LABELS,
    Asset,
    AssetInventory,
    is_missing_value,
    read_source_string,
)
from brails.utils.inventory_union import union_disjoint
from brails.utils.stage_freeze import StageFreeze
from brails.utils.unit_converter import UNIT_TO_METER
from brails.utils.unit_count_rules import UNIT_COUNT_CLASSES

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

# The frozen stages, in the order they are frozen.
STAGE_FOOTPRINTS = 'footprints'
STAGE_LOCAL = 'local'
STAGE_JOIN_NSI = 'join_nsi'
STAGE_JOIN_DERIVED = 'join_derived'
STAGE_STEP4_DERIVED = 'step4_derived'
STAGE_STEP4_DROPPED = 'step4_dropped'
STAGE_STEP4_WITHHELD = 'step4_withheld'
STAGE_STEP4_IMPUTED = 'step4_imputed'
STAGE_STEP4 = 'step4'
STAGE_OUTPUT = 'output'

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
}

# Position of each feature in the source string.
_CODE_POSITION = {
    feature: position for position, feature in enumerate(HAZUS_EQ_SOURCE_FEATURES)
}

# The sources whose occupancy class makes a footprint reached, and those of
# them whose occupancy class the unit budget's mobile-home rule may relabel.
_REACHED_SOURCES = ('local', 'nsi', 'derived')
_RELABELED_SOURCES = ('nsi', 'derived')


class CompilerInputs(NamedTuple):
    """
    The data of one study area.

    Attributes:
        footprints (AssetInventory):
            The building footprints, polygons in EPSG:4326 keyed by string
            asset ids, with one possible world. Each holds
            ``footprint_provider``, for an Overture footprint ``sources``,
            and ``type`` where the file is written with ``integer_ids``. The
            ids are taken as given.
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
            :data:`RESERVED_LOCAL_FEATURES`.
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
            ``max(worlds)``, with every feature the run gave it. A feature
            equal in every world is held as a scalar and otherwise as a list
            in world order. Each asset holds
            :data:`~brails.constants.EXCLUDED_KEY`, true in a world it was
            excluded from.
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
        worlds (tuple[int, ...]):
            The worlds :meth:`write` writes, in ascending order.
    """

    inventory: AssetInventory
    report: CompilationReport
    freeze: StageFreeze
    manifest: dict
    logs: Mapping[str, pd.DataFrame]
    seed: int
    worlds: tuple[int, ...]

    def write(  # noqa: C901
        self,
        path: str | os.PathLike[str],
        *,
        integer_ids: bool = False,
        keep_partly_excluded: bool = False,
        extra_properties: str | Sequence[str] = (),
    ) -> None:
        """
        Write the listed worlds as GeoJSON, with the report, manifest and logs.

        The GeoJSON at ``path`` holds the realizations of :attr:`worlds`,
        so list position ``i`` of a property is world ``worlds[i]``. A
        building excluded in every listed world, by the report's exclusion
        rows of those worlds, is not written, and one excluded in some is
        written only with ``keep_partly_excluded``. Then, and only then,
        each building also holds :data:`~brails.constants.EXCLUDED_KEY`.
        Each building holds the properties of :data:`DEFAULT_PROPERTIES` it
        has and those of ``extra_properties``. With ``integer_ids`` the file
        takes the form the R2D tools read: each building also holds
        ``type``, within which :meth:`AssetInventory.write_to_geojson`
        numbers the features, and its asset id as ``unique_id``. Beside it,
        under the stem of ``path``: the report through
        :meth:`CompilationReport.save` as ``<stem>_report.parquet``, the
        manifest through :meth:`StageFreeze.save_manifest` as
        ``<stem>_manifest.json``, and the logs through
        :meth:`CompilationReport.save_logs` as ``<stem>_<name>.parquet``,
        the report and the logs keeping the rows of the listed worlds.

        Args:
            path (str or os.PathLike):
                The GeoJSON file to write.
            integer_ids (bool, optional):
                Whether to number the features, as
                :meth:`AssetInventory.write_to_geojson` does with it.
            keep_partly_excluded (bool, optional):
                Whether to write the buildings excluded in some, but not
                all, of the listed worlds.
            extra_properties (str or Sequence[str], optional):
                Features of :attr:`inventory` to write beside the default
                properties, or :data:`ALL_PROPERTIES` for every feature it
                holds. They change no value of the default properties.

        Raises:
            TypeError:
                If ``extra_properties`` is a string other than
                :data:`ALL_PROPERTIES`.
            ValueError:
                If ``extra_properties`` names a feature no building of
                :attr:`inventory` holds, or names
                :data:`~brails.constants.EXCLUDED_KEY`, in which case nothing
                is written.
            RuntimeError:
                If a frozen stage changed after it was frozen, in which case
                every file but the manifest has been written.
        """
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
        groups = self.report.exclusion_groups(worlds=self.worlds)
        groups = groups[groups['group'] != 'dropped']
        left_out = set(groups.loc[groups['group'] == 'every', 'asset_id'])
        if not keep_partly_excluded:
            left_out |= set(groups.loc[groups['group'] == 'some', 'asset_id'])
        excluded_in = {
            world: set(rows) for world, rows in groups.groupby('world')['asset_id']
        }
        # Where the listed worlds are every world and the marks of the assets
        # written agree with the report, the inventory's own assets are
        # written, without a re-assembly.
        written = {
            asset_id: asset
            for asset_id, asset in self.inventory.inventory.items()
            if asset_id not in left_out
        }
        as_is = self.worlds == tuple(range(self.inventory.n_pw))
        for asset_id, asset in written.items() if as_is else ():
            marks = asset.features[EXCLUDED_KEY]
            listed = [asset_id in excluded_in.get(w, ()) for w in self.worlds]
            if (
                marks if isinstance(marks, list) else [marks] * len(listed)
            ) != listed:
                as_is = False
                break
        if not as_is:
            assembler = WorldAssembler(len(self.worlds))
            for position, world in enumerate(self.worlds):
                realization = self.inventory.get_world_realization(world)
                for asset_id in left_out:
                    realization.inventory.pop(asset_id, None)
                excluded = excluded_in.get(world, set())
                for asset_id, asset in realization.inventory.items():
                    asset.features[EXCLUDED_KEY] = asset_id in excluded
                assembler.add(position, realization)
                del realization
            written = assembler.result().inventory
        inventory = AssetInventory(
            vector_features=list(self.inventory.vector_features)
        )
        for asset_id, asset in written.items():
            inventory.inventory[asset_id] = Asset(
                asset_id,
                asset.coordinates,
                {
                    key: value
                    for key, value in asset.features.items()
                    if key in selected or (everything and key != EXCLUDED_KEY)
                },
            )
        inventory.n_pw = len(self.worlds)
        inventory.write_to_geojson(str(path), integer_ids=integer_ids)
        self.report.save(
            path.with_name(f'{path.stem}_report.parquet'), worlds=self.worlds
        )
        self.report.save_logs(path.parent, path.stem, self.logs, worlds=self.worlds)
        self.freeze.save_manifest(path.with_name(f'{path.stem}_manifest.json'))


class _Step4(NamedTuple):
    """
    What step 4 hands to the steps after it.

    Attributes:
        imputer (ObservedDonorImputer):
            The imputer step 4 ran, configured by the compiler's options.
        stories (StoriesFromHeightInferer):
            The inferer of story counts from footprint heights.
        targets (AssetInventory):
            The imputer's targets, carrying the budget's values.
        log (pandas.DataFrame):
            The imputer's log, as ``impute_stage`` returned it.
    """

    imputer: ObservedDonorImputer
    stories: StoriesFromHeightInferer
    targets: AssetInventory
    log: pd.DataFrame


class _WorldSources:
    """
    The frozen parts of each source, and the records that act on them.

    A part is registered under the label of the source it belongs to, a
    withheld record names, per asset and feature, the source a value is
    withheld from, and the dropped record, named by :attr:`dropped`, holds
    the footprints removed from every source. :meth:`merge` builds one
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
                The stage name.
            inventory (AssetInventory):
                The part.
        """
        self.freeze.freeze(stage, inventory)
        self.parts[label].append(stage)

    def add_withheld(self, stage: str, inventory: AssetInventory) -> None:
        """
        Freeze a record of withheld values and register it.

        Args:
            stage (str):
                The stage name.
            inventory (AssetInventory):
                One asset per asset with a withheld value, one feature per
                withheld feature, holding the label of its source.
        """
        self.freeze.freeze(stage, inventory)
        self.withheld.append(stage)

    def merge(self, world: int) -> AssetInventory:
        """
        Compose every registered source in one world and merge them.

        Args:
            world (int):
                The world.

        Returns:
            AssetInventory:
                The merge, with one possible world, its source string
                written. A feature whose value is ``None`` is left out, so
                the merge of a world holds the same features whatever the
                number of worlds.
        """
        sources = {
            label: compose_source(
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


def _padded(
    kept: Sequence[str],
    centroids: Mapping[str, list[list[float]]],
    values: Mapping[str, dict[str, Any]],
    *,
    n_pw: int = 1,
    vector_features: Sequence[str] = (),
) -> AssetInventory:
    """
    Build a part or record that holds every footprint the run kept.

    Args:
        kept (Sequence[str]):
            The footprints the run kept, in the order the part holds them.
        centroids (Mapping[str, list[list[float]]]):
            Each footprint's centroid, the coordinates of its asset.
        values (Mapping[str, dict[str, Any]]):
            The features of the assets that hold any, shared with the part.
        n_pw (int, optional):
            The part's number of possible worlds.
        vector_features (Sequence[str], optional):
            The part's vector features.

    Returns:
        AssetInventory:
            One asset per kept footprint, without features where ``values``
            holds none.
    """
    part = AssetInventory(vector_features=list(vector_features))
    for asset_id in kept:
        part.inventory[asset_id] = Asset(
            asset_id, centroids[asset_id], values.get(asset_id, {})
        )
    part.n_pw = n_pw
    return part


def _plausible(
    asset_id: object,  # noqa: ARG001
    world: int,  # noqa: ARG001
    features: Mapping[str, Any],
) -> bool:
    """
    Say whether an imputed occupancy class and story count may stand together.

    Args:
        asset_id (object):
            The asset, unused.
        world (int):
            The world, unused.
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

    def __init__(
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
    ) -> None:
        """
        Configure the compiler and check what can be checked before a run.

        Args:
            seed (int, optional):
                The seed of every component. ``None`` draws one from system
                entropy.
            n_worlds (int, optional):
                The number of possible worlds compiled when
                :meth:`compile` is given no worlds. Defaults to
                :data:`DEFAULT_N_WORLDS`.
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

        Raises:
            TypeError:
                If ``seed`` or ``n_worlds`` is not an integer, an options
                argument is not a mapping, or an options mapping names an
                argument its class does not take or that the compiler sets.
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

        try:
            HazusEQStructureTypeInferer(state, seed=self.seed, complete_tables=True)
        except ValueError as error:
            msg = f'The compiler cannot draw structure types for {state}: {error}'
            raise ValueError(msg) from error
        HazusEQDesignLevelInferer(design_level_method, state, seismic_zone)
        self.state = state
        self.seismic_zone = seismic_zone
        self.design_level_method = design_level_method

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
        ObservedDonorImputer(**resolved['imputer_options'], seed=self.seed)
        self.imputer_options = MappingProxyType(resolved['imputer_options'])
        self.budget_options = MappingProxyType(resolved['budget_options'])

    def compile(
        self, inputs: CompilerInputs, *, worlds: Sequence[int] | None = None
    ) -> CompilationResult:
        """
        Compile the study area.

        The inputs are not modified.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            worlds (Sequence[int], optional):
                The worlds the result writes, distinct non-negative
                integers. Every world from 0 to the largest is computed.
                ``None`` takes every world of :attr:`n_worlds`.

        Returns:
            CompilationResult:
                The compiled inventory and the record of the run.

        Raises:
            TypeError:
                If ``worlds`` holds a value that is not an integer, or a
                footprint's asset id is not a string.
            ValueError:
                If ``worlds`` is empty or repeats a world or holds a
                negative one; if the local inventory holds an asset that the
                footprints lack, a feature of :data:`RESERVED_LOCAL_FEATURES`
                or more than one possible world; or as the
                components raise, among them a footprint outside every block
                group.
            RuntimeError:
                If a frozen stage changed during the run.
        """
        if worlds is None:
            written = tuple(range(self.n_worlds))
        else:
            if any(isinstance(world, bool) for world in worlds):
                msg = f'worlds must hold integers, not {worlds!r}.'
                raise TypeError(msg)
            written = tuple(sorted(operator.index(world) for world in worlds))
            if not written or len(set(written)) != len(written) or written[0] < 0:
                msg = (
                    'worlds must hold distinct non-negative integers, at least '
                    f'one, not {list(worlds)!r}.'
                )
                raise ValueError(msg)
        not_strings = [
            a for a in inputs.footprints.inventory if not isinstance(a, str)
        ]
        if not_strings:
            msg = f'Footprint asset ids must be strings, not {not_strings[:10]!r}.'
            raise TypeError(msg)
        if inputs.local is not None:
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

        n_computed = written[-1] + 1
        settings = {
            'study_area': inputs.study_area,
            'footprint_release': inputs.footprint_release,
            'height_unit': inputs.height_unit,
            'n_worlds': self.n_worlds,
            'worlds': list(written),
            'state': self.state,
            'seismic_zone': self.seismic_zone,
            'design_level_method': self.design_level_method,
            'join_cap_m': self.join_cap_m,
            'projected_crs': None
            if self.projected_crs is None
            else str(self.projected_crs),
            'imputer_options': dict(self.imputer_options),
            'budget_options': dict(self.budget_options),
        }
        report = CompilationReport(self.seed, settings=settings)
        report.set_world_count(n_computed)
        freeze = StageFreeze()
        sources = _WorldSources(freeze)
        logs: dict[str, pd.DataFrame] = {}

        started = time.perf_counter()
        centroids = self._footprints(inputs, sources, logs)
        report.set_timing(STAGE_FOOTPRINTS, time.perf_counter() - started)

        started = time.perf_counter()
        self._join(inputs, sources, logs, centroids)
        report.set_timing('join', time.perf_counter() - started)

        started = time.perf_counter()
        self._step4(inputs, sources, logs, centroids, report, n_computed)
        report.set_timing(STAGE_STEP4, time.perf_counter() - started)
        self._view(STAGE_STEP4, sources, report, n_computed)

        inventory = self._output(sources, report, n_computed)

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
            worlds=written,
        )

    def _footprints(
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
    ) -> dict[str, list[list[float]]]:
        """
        Run step 1: block groups and footprint features, frozen as the spine.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            sources (_WorldSources):
                The sources, to which the footprints and a local inventory
                are added as :data:`STAGE_FOOTPRINTS` and
                :data:`STAGE_LOCAL`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive ``footprint_ties``.

        Returns:
            dict[str, list[list[float]]]:
                Each footprint's centroid as coordinates, the exterior
                rings' centroid of
                :meth:`AssetInventory.centroid_and_plan_area`.

        Raises:
            ValueError:
                If a footprint lies outside every block group or encloses no
                area.
        """
        spine, ties = CensusTractScraper().get_census_geographies(
            inputs.footprints,
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

    def _step4(  # noqa: C901
        self,
        inputs: CompilerInputs,
        sources: _WorldSources,
        logs: dict[str, pd.DataFrame],
        centroids: Mapping[str, list[list[float]]],
        report: CompilationReport,
        n_computed: int,
    ) -> _Step4:
        """
        Run step 4: story counts, the unit budget and the donor imputer.

        The section "The run" of the README beside this module states the
        procedure.

        Args:
            inputs (CompilerInputs):
                The study area's data.
            sources (_WorldSources):
                The sources, to which the parts and records are added as
                :data:`STAGE_STEP4_DERIVED`, :data:`STAGE_STEP4_DROPPED`,
                :data:`STAGE_STEP4_WITHHELD` and :data:`STAGE_STEP4_IMPUTED`.
            logs (dict[str, pandas.DataFrame]):
                The logs, which receive the stories, budget and imputer
                logs.
            centroids (Mapping[str, list[list[float]]]):
                Each footprint's centroid.
            report (CompilationReport):
                The report, which receives the withheld values and the
                dropped footprints of every world.
            n_computed (int):
                The number of worlds computed.

        Returns:
            _Step4:
                The imputer, the stories inferer, the imputer's targets and
                its log.

        Raises:
            ValueError:
                If an asset's source string is not one
                :func:`~brails.types.asset_inventory.read_source_string`
                reads.
        """
        merged = sources.merge(0)

        inferer = StoriesFromHeightInferer(length_unit=inputs.height_unit)
        verdicts = inferer.check(merged)
        logs['stories'] = verdicts.rename_axis('asset_id').reset_index()
        passing = AssetInventory()
        for asset_id in verdicts.index[verdicts['diagnosis'] == DIAGNOSIS_OK]:
            passing.inventory[asset_id] = merged.inventory[asset_id]
        counts = inferer.draw(passing)[STORIES_FROM_HEIGHT_KEY]

        derived: dict[str, dict[str, Any]] = {}
        withheld: dict[str, dict[str, str]] = {}
        # (asset id, feature, old, new, source label, stage, reason)
        actions: list[tuple] = []
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
                withheld.setdefault(asset_id, {})[STORIES_KEY] = 'nsi'
                actions.append(
                    (
                        asset_id,
                        STORIES_KEY,
                        features[STORIES_KEY],
                        count,
                        'nsi',
                        'stories',
                        'the footprint height gives the story count of an '
                        f'occupancy class other than {NSI_STORIES_CLASS}',
                    )
                )
            derived[asset_id] = values
            features.update(values)

        # A footprint is reached where a source holds its occupancy class.
        # Donors are the reached footprints with a story count.
        reached, unreached, donors = (
            AssetInventory(),
            AssetInventory(),
            AssetInventory(),
        )
        relabel = {}
        for asset_id, asset in merged.inventory.items():
            source = SOURCE_LABELS.get(
                read_source_string(asset_id, asset.features)[
                    _CODE_POSITION[OCCUPANCY_KEY]
                ]
            )
            if source not in _REACHED_SOURCES:
                unreached.inventory[asset_id] = asset
                continue
            reached.inventory[asset_id] = asset
            if not is_missing_value(asset.features.get(STORIES_KEY)):
                donors.inventory[asset_id] = asset
            if source in _RELABELED_SOURCES:
                relabel[asset_id] = source

        imputer = ObservedDonorImputer(**self.imputer_options, seed=self.seed)
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
        for row in allocation.withheld.itertuples(index=False):
            withheld.setdefault(row.asset_id, {})[row.feature] = row.source
            actions.append(
                (
                    row.asset_id,
                    row.feature,
                    row.value,
                    allocation.imputed.inventory[row.asset_id].features.get(
                        row.feature
                    ),
                    row.source,
                    'units',
                    'a footprint near a mobile-home park point made a mobile home',
                )
            )
        dropped_ids = set(allocation.dropped)
        no_acs = set(
            allocation.gaps.loc[allocation.gaps['kind'] == GAP_NO_ACS, 'block_group']
        )

        targets = AssetInventory()
        for asset_id, asset in merged.inventory.items():
            if asset_id in dropped_ids:
                continue
            if asset_id in reached.inventory and not is_missing_value(
                asset.features.get(STORIES_KEY)
            ):
                continue
            features = dict(asset.features)
            budgeted = allocation.imputed.inventory.get(asset_id)
            if budgeted is not None:
                features.update(budgeted.features)
            targets.inventory[asset_id] = Asset(
                asset_id, asset.coordinates, features
            )
        targets.n_pw = n_computed
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
        # A reached non-residential footprint is a target for its story
        # count only, since the join gives units to housing records alone.
        for asset_id, asset in imputation.imputed.inventory.items():
            class_of = merged.inventory[asset_id].features.get(OCCUPANCY_KEY)
            if asset_id in reached.inventory and class_of not in UNIT_COUNT_CLASSES:
                asset.features.pop(UNITS_KEY, None)

        kept = sorted((a for a in merged.inventory if a not in dropped_ids), key=str)
        imputed = union_disjoint(allocation.imputed, imputation.imputed)
        padded = _padded(
            kept,
            centroids,
            {
                asset_id: asset.features
                for asset_id, asset in imputed.inventory.items()
            },
            n_pw=n_computed,
            vector_features=imputed.vector_features,
        )

        derived_part = AssetInventory()
        for asset_id in sorted(derived, key=str):
            derived_part.inventory[asset_id] = Asset(
                asset_id, centroids[asset_id], derived[asset_id]
            )
        dropped_record = AssetInventory()
        for asset_id in sorted(dropped_ids, key=str):
            dropped_record.inventory[asset_id] = Asset(
                asset_id, centroids[asset_id], {DROPPED_KEY: True}
            )
        withheld_record = AssetInventory()
        for asset_id in sorted(withheld, key=str):
            withheld_record.inventory[asset_id] = Asset(
                asset_id, centroids[asset_id], withheld[asset_id]
            )
        sources.add_part('derived', STAGE_STEP4_DERIVED, derived_part)
        sources.freeze.freeze(STAGE_STEP4_DROPPED, dropped_record)
        sources.dropped = STAGE_STEP4_DROPPED
        sources.add_withheld(STAGE_STEP4_WITHHELD, withheld_record)
        sources.add_part('imputed', STAGE_STEP4_IMPUTED, padded)

        for world in range(n_computed):
            for asset_id, feature, old, new, label, stage, reason in actions:
                report.record_action(
                    asset_id,
                    feature,
                    stage=stage,
                    action='withheld',
                    world=world,
                    old=old,
                    new=new[world] if isinstance(new, list) else new,
                    source=label,
                    reason=reason,
                    block_group=merged.inventory[asset_id].features.get(
                        BLOCK_GROUP_KEY
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
                    stage='units',
                    category='dropped',
                    detail=detail,
                )
        return _Step4(imputer, inferer, targets, imputation.log)

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
        assembler = WorldAssembler(n_computed)
        for world in range(n_computed):
            started = time.perf_counter()
            assembler.add(world, sources.merge(world))
            report.set_timing(
                f'{stage}_view', time.perf_counter() - started, world=world
            )
        started = time.perf_counter()
        sources.freeze.freeze(stage, assembler.result())
        report.set_timing(f'{stage}_freeze', time.perf_counter() - started)

    def _output(
        self,
        sources: _WorldSources,
        report: CompilationReport,
        n_computed: int,
    ) -> AssetInventory:
        """
        Run the output step: the final merge of each world, marked and assembled.

        Args:
            sources (_WorldSources):
                The sources.
            report (CompilationReport):
                The report, whose exclusion rows mark the buildings and which
                receives the timings.
            n_computed (int):
                The number of worlds computed.

        Returns:
            AssetInventory:
                The output, frozen as :data:`STAGE_OUTPUT`.
        """
        # The GrossArea a local inventory gives, by building.
        local_areas = {
            asset_id: asset.features[GROSS_AREA_KEY]
            for stage in sources.parts['local']
            for asset_id, asset in sources.freeze.read(stage).inventory.items()
            if not is_missing_value(asset.features.get(GROSS_AREA_KEY))
        }
        exclusions = report.exclusions
        excluded_in = {
            world: set(rows)
            for world, rows in exclusions.groupby('world')['asset_id']
        }
        assembler = WorldAssembler(n_computed)
        for world in range(n_computed):
            started = time.perf_counter()
            merged = sources.merge(world)
            excluded = excluded_in.get(world, set())
            for asset_id, asset in merged.inventory.items():
                features = asset.features
                features.pop(GROSS_AREA_KEY, None)
                if asset_id in local_areas:
                    features[LOCAL_GROSS_AREA_KEY] = local_areas[asset_id]
                features[EXCLUDED_KEY] = asset_id in excluded
            assembler.add(world, merged)
            del merged
            report.set_timing(
                STAGE_OUTPUT, time.perf_counter() - started, world=world
            )
        output = assembler.result()
        started = time.perf_counter()
        sources.freeze.freeze(STAGE_OUTPUT, output)
        report.set_timing(f'{STAGE_OUTPUT}_freeze', time.perf_counter() - started)
        return output
