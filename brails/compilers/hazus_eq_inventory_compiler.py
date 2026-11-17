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

import operator
import secrets
import time
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from brails.aggregators.points_to_polygons.nsi_footprint_join import (
    NSIFootprintJoin,
)
from brails.compilers.compilation_report import CompilationReport
from brails.compilers.world_assembly import WorldAssembler, compose_source
from brails.constants import (
    BLOCK_GROUP_KEY,
    DEFAULT_UNITS,
    EXCLUDED_KEY,
    FOOTPRINT_DATASETS_KEY,
    FOOTPRINT_PROVIDER_KEY,
    FOOTPRINT_RELEASE_KEY,
    PLAN_AREA_KEY,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_design_level_inferer import (
    HazusEQDesignLevelInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_eq_structure_type_inferer import (
    HazusEQStructureTypeInferer,
)
from brails.inferers.stories_from_height_inferer.stories_from_height_inferer import (
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
from brails.types.asset_inventory import AssetInventory
from brails.utils.stage_freeze import StageFreeze
from brails.utils.unit_converter import UNIT_TO_METER

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import geopandas as gpd
    import pandas as pd

# The number of possible worlds compiled by default.
DEFAULT_N_WORLDS = 10

# The source labels of a world's merge, highest precedence first.
PRECEDENCE = ('local', 'footprint', 'nsi', 'derived', 'imputed', 'drawn')

# The label of the footprints, the spine of every merge.
SPINE_LABEL = 'footprint'

# The features the compiler writes for itself and reads back, which a local
# inventory may not hold: its precedence would put them in place of the
# compiler's.
RESERVED_LOCAL_FEATURES = (BLOCK_GROUP_KEY,)

# The frozen stages, in the order they are frozen.
STAGE_FOOTPRINTS = 'footprints'
STAGE_LOCAL = 'local'
STAGE_JOIN_NSI = 'join_nsi'
STAGE_JOIN_DERIVED = 'join_derived'
STAGE_OUTPUT = 'output'


class CompilerInputs(NamedTuple):
    """
    The data of one study area.

    Attributes:
        footprints (AssetInventory):
            The building footprints, polygons in EPSG:4326 keyed by string
            asset ids, with one possible world. Each holds
            ``footprint_provider`` and, for an Overture footprint,
            ``sources``. The ids are taken as given.
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


class HazusEQInventoryCompiler:
    """
    Compile a Hazus earthquake inventory of one study area over several worlds.

    The section "Hazus earthquake inventory compiler" of the README beside
    this module describes the run, its frozen stages and its worlds.

    Every constructor argument is stored as an attribute of the same name,
    read-only by convention; ``seed`` holds the seed used, never ``None``.
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

        Raises:
            TypeError:
                If ``seed`` or ``n_worlds`` is not an integer.
            ValueError:
                If ``n_worlds`` is below 1; if
                :class:`HazusEQStructureTypeInferer` refuses the state with
                its completed tables (a state outside the West Coast region,
                or one without a RES1 distribution); if the zone or the method
                is refused by :class:`HazusEQDesignLevelInferer`; or if the
                join refuses its settings.
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
        sources.add_part('nsi', STAGE_JOIN_NSI, result.observed)
        sources.add_part('derived', STAGE_JOIN_DERIVED, result.derived)

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
                asset.features[EXCLUDED_KEY] = asset_id in excluded
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
