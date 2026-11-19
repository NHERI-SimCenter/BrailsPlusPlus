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

"""Tests of the Hazus earthquake inventory compiler."""

from __future__ import annotations

import contextlib
import functools
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import brails.compilers
from brails.aggregators.stacked_footprints.stacked_footprints import (
    StackedFootprintMerge,
)
from brails.compilers.hazus_earthquake import hazus_eq_inventory_compiler as hc
from brails.compilers.hazus_earthquake.hazus_eq_inventory_compiler import (
    ALL_PROPERTIES,
    DEFAULT_IMPUTER_OPTIONS,
    DEFAULT_PROPERTIES,
    MISSING_DEFAULTS_LOG,
    CompilationResult,
    CompilerInputs,
    HazusEQInventoryCompiler,
)
from brails.constants import (
    AREA_FROM_FEATURE,
    BLOCK_GROUP_KEY,
    BUILDING_TYPE_KEY,
    DESIGN_LEVEL_KEY,
    DROPPED_KEY,
    EXCLUDED_KEY,
    FOOTPRINT_DATASETS_KEY,
    FOOTPRINT_PROVENANCE_FEATURES,
    FOOTPRINT_PROVIDER_KEY,
    FOOTPRINT_RELEASE_KEY,
    GROSS_AREA_KEY,
    HEIGHT_CLASS_KEY,
    IMPUTATION_DONOR_KEY,
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
from brails.imputers.observed_donor_imputer.donor_records import (
    OUTCOME_DONOR,
    OUTCOME_NO_DONOR,
)
from brails.imputers.replacement_cost.replacement_cost_imputer import (
    OUTCOME_IMPUTATION_DONOR,
    OUTCOME_LOCAL_VALUE,
    OUTCOME_OWN_RATE,
    OUTCOME_VALUE_CODES,
    ReplacementCostImputer,
    footprint_gross_area,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_height_classes_by_type,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    HEIGHT_KEY,
    OVERTURE_ID_KEY,
    OVERTURE_PROVIDER,
)
from brails.types.asset_inventory import (
    HAZUS_EQ_SOURCE_FEATURES,
    Asset,
    AssetInventory,
)
from brails.utils import Importer
from brails.utils.unit_converter import UNIT_TO_METER
from brails.utils.unit_count_rules import UNIT_COUNT_CLASSES
from brails.validators.schema_validator import SchemaValidator
from tests.compilers.hazus_earthquake.compiler_fixtures import (
    FOOTPRINTS,
    LOCAL,
    METRIC_CRS,
    MH_PARK,
    NSI_RECORDS,
    RELEASE,
    SEED,
    build_inputs,
    square,
)

if TYPE_CHECKING:
    from brails.compilers.compilation_report import CompilationReport

STORIES_CODE = HAZUS_EQ_SOURCE_FEATURES.index(STORIES_KEY)
# Values carried on some footprints, as a caller's footprints may carry them:
# f01 and u08 are excluded in every world, h05 and u03 are not.
FOOTPRINT_VALUES = {
    'f01': {GROSS_AREA_KEY: 789.0, BUILDING_TYPE_KEY: 'C'},
    'u08': {GROSS_AREA_KEY: 456.0, BUILDING_TYPE_KEY: 'S'},
    'u03': {BUILDING_TYPE_KEY: 'S'},
    'h05': {GROSS_AREA_KEY: 123.0},
}
OCCUPANCY_CODE = HAZUS_EQ_SOURCE_FEATURES.index(OCCUPANCY_KEY)
YEAR_CODE = HAZUS_EQ_SOURCE_FEATURES.index(YEAR_BUILT_KEY)
# The footprints no NSI point is joined to.
UNREACHED = set(FOOTPRINTS) - {record[0] for record in NSI_RECORDS.values()}
FREEZE_ORDER = (
    's01_footprints',
    's01_local',
    's02_nsi_join',
    's02_derived',
    's03_story_counts',
    's03_withheld',
    's04_dropped',
    's04_withheld',
    's04_unit_budget',
    's05_imputation',
    's05_view',
    's05_donors',
    's06_residential_year',
    's07_withheld',
    's07_pair_redraw',
    's07_derived',
    's08_structure',
    's08_derived',
    's08_view',
    's09_nonresidential_year',
    's10_nonresidential_structure',
    's10_derived',
    's12_withheld_local',
    's12_withheld_join',
    's12_withheld_footprint',
    's13_withheld',
    's14_withheld',
    's14_withheld_footprint',
    's12_gross_area',
    's13_replacement_cost',
    's14_building_type',
    's15_output',
)
# The timed units of a run, once and per world.
TIMED_ONCE = (
    's01_footprints',
    's02_nsi_join',
    's03_story_counts',
    's04_unit_budget',
    's05_imputation',
    's05_view_freeze',
    's06_residential_year_freeze',
    's07_pair_redraw_freeze',
    's08_view_freeze',
    's09_nonresidential_year_freeze',
    's10_nonresidential_structure_freeze',
    's12_gross_area_freeze',
    's13_replacement_cost_freeze',
    's14_building_type_freeze',
    's15_output_freeze',
    'manifest',
)
TIMED_PER_WORLD = (
    's05_view_merge',
    's06_residential_year',
    's07_pair_redraw',
    's08_structure',
    's08_view_merge',
    's09_nonresidential_year',
    's10_nonresidential_structure',
    's15_output_merge',
    's11_schema_check',
    's12_gross_area',
    's13_replacement_cost',
    's14_building_type',
    's15_output',
)


@pytest.fixture(scope='session')
def study_area() -> CompilerInputs:
    """
    Return the study area, shared by every test, which must not change it.

    Returns:
        CompilerInputs:
            The study area's data.
    """
    return build_inputs()


@pytest.fixture(scope='session')
def compiled(study_area: CompilerInputs) -> Callable[..., CompilationResult]:
    """
    Return a function that compiles the study area, each setting once.

    The results are shared by every test, which must not change them.

    Args:
        study_area (CompilerInputs):
            The study area's data, from the ``study_area`` fixture.

    Returns:
        Callable[..., CompilationResult]:
            ``compiled(n_worlds=3)`` gives the result of the compiler at
            the fixture seed and CRS.
    """

    @functools.cache
    def run(n_worlds: int = 3) -> object:
        """
        Compile the study area with the given number of worlds.

        Args:
            n_worlds (int, optional):
                The number of worlds computed.

        Returns:
            CompilationResult:
                The result, shared between calls with the same settings.
        """
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=n_worlds, projected_crs=METRIC_CRS
        )
        return compiler.compile(study_area)

    return run


def written(path: object) -> dict[str, dict]:
    """
    Read a written GeoJSON's properties by feature id.

    Args:
        path (object):
            The GeoJSON file to read.

    Returns:
        dict[str, dict]:
            Each feature's properties, by its id.
    """
    with Path(path).open(encoding='utf-8') as file:
        collection = json.load(file)
    return {
        feature['id']: feature['properties'] for feature in collection['features']
    }


def features_in(inventory: AssetInventory, world: int) -> dict[str, dict]:
    """
    Return every asset's features in one world, without the empty ones.

    An assembled inventory holds a feature as a list over its worlds, with
    ``None`` in a world where it has no value, so a realization of a longer
    run holds ``None`` where a shorter run holds no key. The merged
    inventories the run builds hold no ``None``
    (``test_merge_keys_do_not_depend_on_n_worlds``), and the comparison of
    realizations leaves ``None`` out.

    Args:
        inventory (AssetInventory):
            An inventory of several worlds.
        world (int):
            The index of the world to read.

    Returns:
        dict[str, dict]:
            The features of each asset in that world that are not ``None``.
    """
    realization = inventory.get_world_realization(world)
    return {
        a: {key: value for key, value in asset.features.items() if value is not None}
        for a, asset in realization.inventory.items()
    }


def rows_of(frame: pd.DataFrame, world: int) -> list[tuple]:
    """
    Return the rows of a report frame in one world, without the world.

    Args:
        frame (pandas.DataFrame):
            The action rows or the exclusion rows of a report.
        world (int):
            The index of the world whose rows are returned.

    Returns:
        list[tuple]:
            The rows, in order, as tuples of text.
    """
    kept = frame[frame['world'] == world].drop(columns='world')
    return [tuple(map(str, row)) for row in kept.itertuples(index=False)]


class TestConstructor:
    """The constructor checks its settings and resolves the imputer's."""

    @pytest.mark.parametrize(
        ('arguments', 'error', 'match'),
        [
            ({'n_worlds': 0}, ValueError, 'at least 1'),
            ({'n_worlds': True}, TypeError, 'n_worlds'),
            ({'seed': 1.5}, TypeError, None),
            ({'state': 'CA'}, ValueError, 'not a state in the Hazus region'),
            ({'state': 'Texas'}, ValueError, 'complete_tables applies to the West'),
            ({'state': 'Guam'}, ValueError, 'no RES1 distribution for Guam'),
            ({'seismic_zone': '9'}, ValueError, 'zone'),
            ({'join_cap_m': -1.0}, ValueError, 'cap_m'),
            ({'imputer_options': {'k': 0}}, ValueError, 'at least 1'),
            ({'imputer_options': {'n_batches': 2}}, ValueError, 'donors directly'),
            ({'imputer_options': {'seed': 1}}, TypeError, 'seed'),
            ({'imputer_options': {'kk': 1}}, TypeError, 'kk'),
            ({'imputer_options': [('k', 3)]}, TypeError, 'mapping'),
            ({'imputer_options': {'status_key': 'x'}}, TypeError, 'status_key'),
            ({'budget_options': {'imputer': None}}, TypeError, 'imputer'),
            ({'budget_options': {'radius': 1}}, TypeError, 'radius'),
            ({'sampler_options': {'world': 1}}, TypeError, 'world'),
            (
                {'sampler_options': {'block_group_key': 'x'}},
                TypeError,
                'block_group_key',
            ),
            ({'merge_stacked': 1}, TypeError, 'merge_stacked'),
        ],
    )
    def test_refusals(self, arguments: dict, error: type, match: str | None) -> None:
        """A setting no component accepts is refused before any run."""
        with pytest.raises(error, match=match):
            HazusEQInventoryCompiler(**arguments)

    def test_options_and_seed(self) -> None:
        """The imputer defaults are the compiler's, overridden key by key."""
        compiler = HazusEQInventoryCompiler(
            imputer_options={'k': 7},
            budget_options={'threshold_sqft': 900.0},
            sampler_options={'smoothing': 5.0},
        )
        assert isinstance(compiler.seed, int)
        assert dict(compiler.imputer_options) == {**DEFAULT_IMPUTER_OPTIONS, 'k': 7}
        assert dict(compiler.budget_options) == {'threshold_sqft': 900.0}
        assert dict(compiler.sampler_options) == {'smoothing': 5.0}
        assert dict(HazusEQInventoryCompiler().imputer_options) == dict(
            DEFAULT_IMPUTER_OPTIONS
        )


class TestCompileArguments:
    """`compile` checks its inputs and leaves them alone."""

    def test_inputs_refused(self, study_area: object) -> None:
        """Footprint ids must be strings, and the local values one world of them."""
        footprints = build_inputs().footprints
        footprints.inventory[7] = footprints.inventory.pop('h01')
        with pytest.raises(TypeError, match='must be strings'):
            HazusEQInventoryCompiler(seed=SEED).compile(
                study_area._replace(footprints=footprints)
            )
        local = build_inputs().local
        local.add_asset_coordinates('zz', [[-122.4, 37.7]])
        with pytest.raises(ValueError, match='footprints lack'):
            HazusEQInventoryCompiler(seed=SEED).compile(
                study_area._replace(local=local)
            )
        local = build_inputs().local
        local.n_pw = 2
        with pytest.raises(ValueError, match='one possible world'):
            HazusEQInventoryCompiler(seed=SEED).compile(
                study_area._replace(local=local)
            )
        with pytest.raises(ValueError, match='merge_stacked'):
            HazusEQInventoryCompiler(seed=SEED, merge_stacked=True).compile(
                study_area
            )

    @pytest.mark.parametrize('key', hc.RESERVED_LOCAL_FEATURES)
    def test_reserved_local_feature_refused(self, key: str) -> None:
        """A local inventory holding a feature the compiler reads back is refused."""
        # Were a local NSIGrossArea of 7.0 on h02 accepted, s13 would divide
        # NSI's ReplacementCost by 7 sq ft to get the rate of h02 and of every
        # building that takes its rate from h02.
        inputs = build_inputs()
        coordinates = inputs.footprints.inventory['h02'].coordinates
        inputs.local.add_asset('h02', Asset('h02', coordinates, {key: 7.0}))
        with pytest.raises(ValueError, match=key):
            HazusEQInventoryCompiler(seed=SEED).compile(inputs)

    def test_inputs_not_modified(self, study_area: object, compiled: object) -> None:
        """A run leaves every input as it was."""
        compiled()
        fresh = build_inputs()
        for name in ('footprints', 'nsi_points', 'local'):
            given = getattr(study_area, name).get_geojson()['features']
            assert given == getattr(fresh, name).get_geojson()['features']
        for name in (
            'block_groups',
            'b25024_block_groups',
            'b25034_block_groups',
            'b25034_tracts',
            'b25127_tracts',
        ):
            pd.testing.assert_frame_equal(
                getattr(study_area, name), getattr(fresh, name)
            )
        np.testing.assert_array_equal(
            study_area.mh_park_points, fresh.mh_park_points
        )


class TestFootprintsAndJoin:
    """Steps s01 and s02: the footprints stage, its features and the join's parts."""

    def test_footprint_source_and_source_string(self, compiled: object) -> None:
        """Every footprint names its source and carries a source string."""
        result = compiled(n_worlds=3)
        for asset_id, asset in result.inventory.inventory.items():
            values = asset.features
            codes = values[SOURCE_STRING_KEY]
            for code in codes if isinstance(codes, list) else [codes]:
                assert len(code) == len(HAZUS_EQ_SOURCE_FEATURES)
            side = FOOTPRINTS[asset_id][2]
            assert math.isclose(
                values[PLAN_AREA_KEY],
                side**2 / UNIT_TO_METER['ft'] ** 2,
                rel_tol=1e-3,
            )
            if FOOTPRINTS[asset_id][4] is None:
                assert values[FOOTPRINT_PROVIDER_KEY] == 'osm'
                assert FOOTPRINT_RELEASE_KEY not in values
                assert FOOTPRINT_DATASETS_KEY not in values
            else:
                assert values[FOOTPRINT_PROVIDER_KEY] == OVERTURE_PROVIDER
                assert values[FOOTPRINT_RELEASE_KEY] == f'overture_{RELEASE}'
                assert values[FOOTPRINT_DATASETS_KEY]
        held = result.inventory.inventory['h06'].features
        assert held[FOOTPRINT_DATASETS_KEY] == 'Microsoft ML Buildings;OpenStreetMap'

    def test_block_groups_queried_without_polygons(
        self, study_area: object, compiled: object
    ) -> None:
        """Inputs without block-group polygons pass None to the scraper and compile as with them."""
        real = hc.CensusTractScraper.get_census_geographies
        asked = []

        def query(
            scraper: object, inventory: AssetInventory, **settings: object
        ) -> tuple:
            # Stands in for the query with the fixture's own polygons.
            asked.append(settings.pop('polygons'))
            return real(
                scraper, inventory, polygons=study_area.block_groups, **settings
            )

        given = study_area._asdict()
        del given['block_groups']
        with mock.patch.object(
            hc.CensusTractScraper,
            'get_census_geographies',
            autospec=True,
            side_effect=query,
        ):
            result = HazusEQInventoryCompiler(
                seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
            ).compile(CompilerInputs(**given))

        assert asked == [None]
        expected = compiled(n_worlds=3).inventory
        for world in range(3):
            assert features_in(result.inventory, world) == features_in(
                expected, world
            )

    @pytest.mark.parametrize(('height_unit', 'merged'), [('ft', 1), ('m', 2)])
    def test_stacked_footprints_merge_into_the_spine(
        self, study_area: object, height_unit: str, merged: int
    ) -> None:
        """With merge_stacked the footprints stage holds the merged footprints, under their new keys."""
        footprints = build_inputs().footprints
        for asset_id, asset in footprints.inventory.items():
            asset.features.update(
                {
                    FOOTPRINT_PROVIDER_KEY: OVERTURE_PROVIDER,
                    OVERTURE_ID_KEY: asset_id,
                }
            )
        # A tower on p01, half a meter of it outside, 68 ft taller; a
        # footprint in c04 5 ft taller, which merges only if read in meters;
        # and a second drawing of c01, half a meter east, which is dropped.
        x, y, side, height = FOOTPRINTS['p01'][:4]
        added = {
            't01': (square(x + side - 9.5, y, 10), height + 68.0),
            'k01': (square(162, 142, 5), FOOTPRINTS['c04'][3] + 5.0),
            'c01b': (square(50.5, 100, 20), FOOTPRINTS['c01'][3]),
        }
        for asset_id, (ring, part_height) in added.items():
            features = {
                'type': 'building',
                FOOTPRINT_PROVIDER_KEY: OVERTURE_PROVIDER,
                OVERTURE_ID_KEY: asset_id,
                HEIGHT_KEY: part_height,
            }
            footprints.add_asset(asset_id, Asset(asset_id, ring, features))
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS, merge_stacked=True
        ).compile(
            study_area._replace(
                footprints=footprints, local=None, height_unit=height_unit
            )
        )

        spine = result.freeze.read('s01_footprints').inventory
        merge = StackedFootprintMerge(length_unit=height_unit)
        assert set(spine) == set(merge.merge(footprints).inventory.inventory)
        assert len(spine) == len(FOOTPRINTS) + 2 - merged
        (head,) = [a for a in spine.values() if a.features[OVERTURE_ID_KEY] == 'p01']
        assert math.isclose(
            head.features[PLAN_AREA_KEY],
            (side**2 + 5.0) / UNIT_TO_METER['ft'] ** 2,
            rel_tol=1e-3,
        )
        parts = sorted(result.logs['stacked_parts'][OVERTURE_ID_KEY])
        stacked = ['c04', 'k01', 'p01', 't01'] if merged == 2 else ['p01', 't01']
        assert parts == sorted(['c01', 'c01b', *stacked])
        assert list(result.logs['stacked_asset_ids'].columns) == [
            'position',
            'geohash',
            'area_m2',
            'rank',
            'id',
            'reason',
        ]
        summary = result.report.summary()
        assert summary['settings']['merge_stacked'] is True
        notice = (
            f'Stacked footprints: {merged} footprint(s) merged into {merged} '
            'asset(s); 1 duplicate drawing(s) dropped.'
        )
        assert notice in summary['notices']

    def test_join_parts(self, compiled: object) -> None:
        """The footprints stage holds every footprint with its outline, and the other parts of s01 and s02 place each asset at its centroid."""
        result = compiled(n_worlds=3)
        read = result.freeze.read
        assert result.freeze.order == FREEZE_ORDER
        assert [e['stage'] for e in result.manifest['stages']] == list(FREEZE_ORDER)
        assert set(read('s01_footprints').inventory) == set(FOOTPRINTS)
        assert all(
            len(asset.coordinates) > 1
            for asset in read('s01_footprints').inventory.values()
        )
        assert set(read('s01_local').inventory) == set(LOCAL)
        for stage in ('s01_local', 's02_nsi_join', 's02_derived'):
            for asset in read(stage).inventory.values():
                assert len(asset.coordinates) == 1, stage
        assert read('s15_output') is result.inventory

    def test_changed_part_raises_at_the_manifest(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A component that changes a frozen part it read stops the run."""
        real = hc.compose_source_parts

        def changing(
            freeze: object, parts: list, **kwargs: object
        ) -> AssetInventory:
            """
            Compose a source, then change an asset it shares with a part.

            Args:
                freeze (object):
                    The freeze the stages are read from.
                parts (list):
                    The names of the stages composed into the source.
                **kwargs (object):
                    The other arguments of ``compose_source_parts``, passed on.

            Returns:
                AssetInventory:
                    The composed source, with h01's story count set to 9
                    where ``parts`` holds ``'s02_nsi_join'``.
            """
            composed = real(freeze, parts, **kwargs)
            if 's02_nsi_join' in parts:
                composed.inventory['h01'].features[STORIES_KEY] = 9
            return composed

        monkeypatch.setattr(hc, 'compose_source_parts', changing)
        with pytest.raises(RuntimeError, match='s02_nsi_join'):
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
            ).compile(study_area)


class TestOutput:
    """The last steps' output and what `write` makes of it."""

    def test_output_holds_the_excluded_values(self, study_area: object) -> None:
        """An exclusion recorded before s11 sets Excluded in its world."""
        real = hc.CompilationReport.set_world_count

        def counted(report: CompilationReport, count: int) -> None:
            """
            Set the world count, then exclude two buildings from world 1.

            Args:
                report (CompilationReport):
                    The run's report.
                count (int):
                    The number of worlds computed.
            """
            real(report, count)
            for asset_id in ('h01', 'u02'):
                report.record_exclusion(
                    asset_id,
                    world=1,
                    stage='s08_resolution',
                    category='conflict',
                    detail='x',
                )

        with mock.patch.object(
            hc.CompilationReport,
            'set_world_count',
            autospec=True,
            side_effect=counted,
        ):
            result = HazusEQInventoryCompiler(
                seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
            ).compile(study_area)
        for asset_id in ('h01', 'u02'):
            excluded_values = result.inventory.inventory[asset_id].features[
                EXCLUDED_KEY
            ]
            assert excluded_values == [False, True, False]
        # An excluded building takes no value from s13 in that world.
        values = result.inventory.inventory['u02'].features[REPLACEMENT_COST_KEY]
        assert values[1] is None
        assert values[0] is not None
        assert values[2] is not None

    def test_excluded_assets(self, study_area: object, tmp_path: object) -> None:
        """Excluded in every written world: left out; in some: on request."""
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        ).compile(study_area)
        report = result.report
        assert result.inventory.inventory['h01'].features[EXCLUDED_KEY] is False
        report.record_exclusion(
            'h01',
            world=1,
            stage='s06_residential_year',
            category='missing_input',
            detail='x',
        )
        for world in range(3):
            report.record_exclusion(
                'h02',
                world=world,
                stage='s08_resolution',
                category='conflict',
                detail='x',
            )
        result.write(tmp_path / 'default.geojson')
        default = written(tmp_path / 'default.geojson')
        assert 'h01' not in default
        assert 'h02' not in default
        assert all(EXCLUDED_KEY not in values for values in default.values())

        result.write(tmp_path / 'kept.geojson', keep_partly_excluded=True)
        kept = written(tmp_path / 'kept.geojson')
        assert kept['h01'][EXCLUDED_KEY] == [False, True, False]
        assert 'h02' not in kept

        result.write(tmp_path / 'world0.geojson', world=0)
        assert EXCLUDED_KEY not in written(tmp_path / 'world0.geojson')['h01']

        groups = report.exclusion_groups(worlds=(0, 1, 2))
        summary = json.loads(
            (tmp_path / 'default_report_summary.json').read_text(encoding='utf-8')
        )
        for group in ('every', 'some', 'dropped'):
            assets = groups.loc[groups['group'] == group, 'asset_id']
            assert summary['assets_excluded'][group] == assets.nunique()
        assert 'h02' in set(groups.loc[groups['group'] == 'every', 'asset_id'])

    def test_full_write_takes_the_inventory(
        self, compiled: object, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write of every world assembles nothing new, and a write of one world assembles."""
        result = compiled(n_worlds=3)
        adder = mock.create_autospec(
            AssetInventory.add_world, side_effect=AssetInventory.add_world
        )
        monkeypatch.setattr(AssetInventory, 'add_world', adder)
        result.write(tmp_path / 'all.geojson', keep_partly_excluded=True)
        result.write(tmp_path / 'default.geojson')
        assert not adder.called
        assert result.freeze.manifest() == result.manifest
        result.write(tmp_path / 'world1.geojson', world=1)
        assert adder.called

    @pytest.mark.parametrize(
        ('world', 'error', 'message'),
        [
            (-1, ValueError, 'outside the 3 worlds of the run, 0 to 2'),
            (3, ValueError, 'outside the 3 worlds of the run, 0 to 2'),
            (True, TypeError, 'must be an integer'),
            (1.5, TypeError, 'must be an integer'),
        ],
    )
    def test_world_refused(
        self,
        compiled: object,
        tmp_path: object,
        world: object,
        error: type,
        message: str,
    ) -> None:
        """A world outside the run, a bool or a value that is not an integer writes nothing."""
        with pytest.raises(error, match=message):
            compiled(n_worlds=3).write(tmp_path / 'run.geojson', world=world)
        assert not list(tmp_path.iterdir())

    def test_one_world_file_ignores_the_world_count(
        self, compiled: object, tmp_path: object
    ) -> None:
        """World 3 is written the same from four worlds and from ten, with no null property."""
        files = {}
        for n_worlds in (4, 10):
            path = tmp_path / f'w3_of_{n_worlds}.geojson'
            compiled(n_worlds=n_worlds).write(
                path,
                world=3,
                keep_partly_excluded=True,
                extra_properties=ALL_PROPERTIES,
            )
            files[n_worlds] = {
                feature['id']: (feature['geometry'], feature['properties'])
                for feature in json.loads(path.read_text(encoding='utf-8'))[
                    'features'
                ]
            }
        assert files[4] == files[10]
        assert all(
            value is not None
            for _, properties in files[10].values()
            for value in properties.values()
        )

    def test_files(self, compiled: object, tmp_path: object) -> None:
        """The manifest and the logs are written beside the file, readable."""
        result = compiled(n_worlds=3)
        result.write(tmp_path / 'run.geojson')
        manifest = json.loads(
            (tmp_path / 'run_manifest.json').read_text(encoding='utf-8')
        )
        assert manifest == result.manifest
        for name in result.logs:
            assert (tmp_path / f'run_{name}.parquet').exists()
        counts = pd.read_parquet(tmp_path / 'run_join_branch_counts.parquet')
        assert 'inside' in set(counts['branch'])


class TestWrittenFile:
    """What a written building holds: the default properties and those asked for."""

    def test_default_properties(self, compiled: object, tmp_path: object) -> None:
        """A default file holds each default property a building has, and no other."""
        result = compiled(n_worlds=3)
        result.write(tmp_path / 'run.geojson')
        properties = written(tmp_path / 'run.geojson')
        assert properties
        for asset_id, values in properties.items():
            features = result.inventory.inventory[asset_id].features
            assert set(values) - {'id'} == set(DEFAULT_PROPERTIES) & set(features)
        # NSI's and a local GrossArea are kept under names of their own.
        inventory = result.inventory.inventory
        assert inventory['h01'].features[NSI_GROSS_AREA_KEY] == 1400.0
        assert inventory['h01'].features[GROSS_AREA_KEY] != 1400.0
        assert inventory['l01'].features[LOCAL_GROSS_AREA_KEY] == 777.0
        assert inventory['l01'].features[NSI_GROSS_AREA_KEY] == 1400.0
        # l01's one local story: its GrossArea is its plan area.
        assert math.isclose(
            inventory['l01'].features[GROSS_AREA_KEY],
            inventory['l01'].features[PLAN_AREA_KEY],
        )
        # The join's sum over l02's two NSI points, a derived value.
        assert inventory['l02'].features[NSI_GROSS_AREA_KEY] == 7000.0

    def test_coverage(self, compiled: object) -> None:
        """No compiled building lacks a default property that applies to it."""
        result = compiled(n_worlds=3)
        assert result.logs[MISSING_DEFAULTS_LOG].empty
        # Both exemptions occur: a building outside UNIT_COUNT_CLASSES without
        # units, and one of a type without height classes.
        bands = get_hazus_height_classes_by_type()
        exempt = set()
        for world in range(3):
            for values in features_in(result.inventory, world).values():
                if values[EXCLUDED_KEY]:
                    continue
                if values[OCCUPANCY_KEY] not in UNIT_COUNT_CLASSES:
                    exempt |= {UNITS_KEY} - set(values)
                if bands[values[STRUCTURE_TYPE_KEY]] is None:
                    exempt |= {HEIGHT_CLASS_KEY} - set(values)
        assert exempt == {UNITS_KEY, HEIGHT_CLASS_KEY}

    @pytest.mark.parametrize(
        ('building', 'key'),
        [
            ('c01', OCCUPANCY_KEY),
            ('h01', UNITS_KEY),
            ('c01', STORIES_KEY),
            ('c01', YEAR_BUILT_KEY),
            ('c01', DESIGN_LEVEL_KEY),
            ('c01', STRUCTURE_TYPE_KEY),
            ('c01', HEIGHT_CLASS_KEY),
            ('c02', REPLACEMENT_COST_KEY),
            ('c01', BLOCK_GROUP_KEY),
            ('c01', PLAN_AREA_KEY),
            ('c02', SOURCE_STRING_KEY),
        ],
    )
    def test_coverage_defect(
        self,
        study_area: object,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: object,
        building: str,
        key: str,
    ) -> None:
        """A missing default value is logged, and write keeps the rest of the run."""
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        )
        real = compiler._validator.validate

        def validate(inventory: AssetInventory) -> object:
            """
            Check an inventory, then remove one value of one building.

            Args:
                inventory (AssetInventory):
                    The buildings checked, which share their asset objects
                    with the merged inventory s11 to s15 work on.

            Returns:
                object:
                    The check's verdict.
            """
            verdict = real(inventory)
            inventory.inventory[building].features.pop(key)
            return verdict

        monkeypatch.setattr(compiler._validator, 'validate', validate)
        result = compiler.compile(study_area)
        lacking = set(
            result.logs[MISSING_DEFAULTS_LOG].itertuples(index=False, name=None)
        )
        assert (building, 0, key) in lacking
        with pytest.raises(RuntimeError, match=key):
            result.write(tmp_path / 'run.geojson')
        assert not (tmp_path / 'run.geojson').exists()
        assert (tmp_path / 'run_report.parquet').exists()
        assert (tmp_path / 'run_manifest.json').exists()
        saved = pd.read_parquet(tmp_path / f'run_{MISSING_DEFAULTS_LOG}.parquet')
        assert (building, 0, key) in set(saved.itertuples(index=False, name=None))

    def test_coverage_defect_of_one_world(
        self,
        study_area: object,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: object,
    ) -> None:
        """A file of the world without the defect is written, and a file of the other world or of both is refused."""
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=2, projected_crs=METRIC_CRS
        )
        real = compiler._validator.validate
        calls = []

        def validate(inventory: AssetInventory) -> object:
            """
            Check an inventory, then remove c01's year in the second world.

            Args:
                inventory (AssetInventory):
                    The buildings checked, which share their asset objects
                    with the merged inventory s11 to s15 work on.

            Returns:
                object:
                    The check's verdict.
            """
            verdict = real(inventory)
            calls.append(None)
            if len(calls) == 2:
                inventory.inventory['c01'].features.pop(YEAR_BUILT_KEY)
            return verdict

        monkeypatch.setattr(compiler._validator, 'validate', validate)
        result = compiler.compile(study_area)
        result.write(tmp_path / 'world0.geojson', world=0)
        assert (tmp_path / 'world0.geojson').exists()
        with pytest.raises(RuntimeError, match=YEAR_BUILT_KEY):
            result.write(tmp_path / 'both.geojson')
        with pytest.raises(RuntimeError, match=YEAR_BUILT_KEY):
            result.write(tmp_path / 'world1.geojson', world=1)
        # A refused file still leaves the report of the worlds asked for.
        for stem, worlds in (('world0', {0}), ('both', {0, 1}), ('world1', {1})):
            rows = pd.read_parquet(tmp_path / f'{stem}_report_exclusions.parquet')
            assert set(rows['world']) == worlds

    @pytest.mark.parametrize(
        ('name', 'replacement', 'building', 'key'),
        [
            # Without an area every other building gets no value and is
            # excluded, and c02 keeps its local value.
            (
                'footprint_gross_area',
                lambda *_args, **_kwargs: (None, None),
                'c02',
                GROSS_AREA_KEY,
            ),
            ('_LETTER_OF_TYPE', {'C1': 'C'}, 'h01', BUILDING_TYPE_KEY),
        ],
    )
    def test_coverage_of_written_values(
        self,
        study_area: object,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        replacement: object,
        building: str,
        key: str,
    ) -> None:
        """A value s15 cannot write is logged as missing."""
        monkeypatch.setattr(hc, name, replacement)
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        ).compile(study_area)
        lacking = set(
            result.logs[MISSING_DEFAULTS_LOG].itertuples(index=False, name=None)
        )
        assert (building, 0, key) in lacking

    def test_schema_reads_no_value_written_after_it(self) -> None:
        """The input schema reads none of the values s15 writes after it."""
        schema = json.dumps(SchemaValidator(hc.DLML_DATASET)._schema, default=str)
        for key in (REPLACEMENT_COST_KEY, GROSS_AREA_KEY, BUILDING_TYPE_KEY):
            assert f'"{key}"' not in schema

    def test_extra_properties(self, compiled: object, tmp_path: object) -> None:
        """Extra properties are written beside the default ones and change none."""
        result = compiled(n_worlds=3)
        extra = (
            *FOOTPRINT_PROVENANCE_FEATURES,
            NSI_GROSS_AREA_KEY,
            LOCAL_GROSS_AREA_KEY,
        )
        result.write(tmp_path / 'default.geojson')
        result.write(tmp_path / 'extra.geojson', extra_properties=extra)
        result.write(tmp_path / 'all.geojson', extra_properties=ALL_PROPERTIES)
        default = written(tmp_path / 'default.geojson')
        more = written(tmp_path / 'extra.geojson')
        everything = written(tmp_path / 'all.geojson')
        assert set(default) == set(more) == set(everything)
        for asset_id, values in default.items():
            assert {key: more[asset_id][key] for key in values} == values
            assert {key: everything[asset_id][key] for key in values} == values
            assert set(more[asset_id]) - set(values) <= set(extra)
            features = result.inventory.inventory[asset_id].features
            assert set(everything[asset_id]) - {'id'} == set(features) - {
                EXCLUDED_KEY
            }
        assert more['h01'][NSI_GROSS_AREA_KEY] == 1400.0

    def test_extra_properties_refused(
        self, compiled: object, tmp_path: object
    ) -> None:
        """An unknown name, Excluded, or one name as a string writes nothing."""
        result = compiled(n_worlds=3)
        with pytest.raises(ValueError, match='no_such_feature'):
            result.write(
                tmp_path / 'x.geojson', extra_properties=('no_such_feature',)
            )
        with pytest.raises(ValueError, match='keep_partly_excluded'):
            result.write(tmp_path / 'x.geojson', extra_properties=(EXCLUDED_KEY,))
        with pytest.raises(TypeError, match='sequence'):
            result.write(
                tmp_path / 'x.geojson', extra_properties=FOOTPRINT_RELEASE_KEY
            )
        assert not list(Path(tmp_path).iterdir())

    def test_integer_ids(self, compiled: object, tmp_path: object) -> None:
        """The integer-id form also writes the type and the asset id."""
        result = compiled(n_worlds=3)
        result.write(tmp_path / 'default.geojson')
        result.write(tmp_path / 'numbered.geojson', integer_ids=True)
        collection = json.loads(
            (tmp_path / 'numbered.geojson').read_text(encoding='utf-8')
        )
        features = collection['features']
        assert all(isinstance(feature['id'], int) for feature in features)
        assert {feature['properties']['type'] for feature in features} == {
            'building'
        }
        assert {feature['properties']['unique_id'] for feature in features} == set(
            written(tmp_path / 'default.geojson')
        )


class TestRandomness:
    """The run depends on the seed alone."""

    def test_repeated_and_reordered(self, tmp_path: object) -> None:
        """One seed gives one file, in any input order; another seed differs."""
        files = []
        for name, inputs, seed in (
            ('first', build_inputs(), SEED),
            ('second', build_inputs(), SEED),
            ('reversed', build_inputs(reverse=True), SEED),
            ('other', build_inputs(), SEED + 1),
        ):
            result = HazusEQInventoryCompiler(
                seed=seed, n_worlds=3, projected_crs=METRIC_CRS
            ).compile(inputs)
            result.write(
                tmp_path / f'{name}.geojson', extra_properties=ALL_PROPERTIES
            )
            collection = json.loads(
                (tmp_path / f'{name}.geojson').read_text(encoding='utf-8')
            )
            collection.pop('generated')
            files.append(collection)
        first, second, reversed_order, other = files
        assert first == second == reversed_order
        assert first != other


class TestReport:
    """The report times the run and records its settings."""

    def test_s09_and_s10_are_timed_apart(self, study_area: object) -> None:
        """In a world, s09's time and s10's are each counted under its own key."""
        pause = 0.2
        structure = hc.HazusEQInventoryCompiler._assign_structure
        draw = hc.ResidentialNeighborYearSampler.draw_details

        def slow_structure(
            compiler: object, *args: object, **kwargs: object
        ) -> object:
            """
            Run the structure operations, pausing first when s10 runs them.

            Args:
                compiler (object):
                    The compiler whose method is wrapped.
                *args (object):
                    The method's arguments.
                **kwargs (object):
                    The method's keyword arguments, ``step`` among them.

            Returns:
                object:
                    What the method returns.
            """
            if kwargs.get('step') == 's10':
                time.sleep(pause)
            return structure(compiler, *args, **kwargs)

        def slow_draw(sampler: object, *args: object) -> object:
            """
            Draw s09's years after a pause.

            Args:
                sampler (object):
                    The sampler whose method is wrapped.
                *args (object):
                    The method's arguments.

            Returns:
                object:
                    What the method returns.
            """
            time.sleep(pause)
            return draw(sampler, *args)

        with (
            mock.patch.object(
                hc.HazusEQInventoryCompiler,
                '_assign_structure',
                autospec=True,
                side_effect=slow_structure,
            ),
            mock.patch.object(
                hc.ResidentialNeighborYearSampler,
                'draw_details',
                autospec=True,
                side_effect=slow_draw,
            ),
        ):
            result = HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
            ).compile(study_area)
        seconds = {
            entry['stage']: entry['seconds']
            for entry in result.report.summary()['timings']
            if entry['world'] == 0
        }
        for key in ('s09_nonresidential_year', 's10_nonresidential_structure'):
            assert pause <= seconds[key] < 2 * pause, key

    def test_timings_and_settings(self, compiled: object) -> None:
        """The summary holds seconds per stage, per world and of the manifest."""
        result = compiled(n_worlds=3)
        summary = result.report.summary()
        timed = {(entry['stage'], entry['world']) for entry in summary['timings']}
        assert {(stage, None) for stage in TIMED_ONCE} <= timed
        assert {
            (stage, world) for stage in TIMED_PER_WORLD for world in range(3)
        } <= timed
        assert summary['manifest_stages'] == list(FREEZE_ORDER)
        assert summary['worlds']['computed'] == 3
        assert summary['settings']['imputer_options'] == dict(
            DEFAULT_IMPUTER_OPTIONS
        )
        assert summary['settings']['study_area'] == '06075'
        assert summary['seed'] == SEED


def test_discovered_and_exported() -> None:
    """The Importer finds the compiler, and the package exports it and its inputs."""
    assert (
        Importer().get_class('HazusEQInventoryCompiler') is HazusEQInventoryCompiler
    )
    for name in ('HazusEQInventoryCompiler', 'CompilerInputs'):
        assert name in brails.compilers.__all__
    assert brails.compilers.HazusEQInventoryCompiler is HazusEQInventoryCompiler
    assert brails.compilers.CompilerInputs is CompilerInputs


class TestStoriesUnitsAndImputation:
    """Steps s03 to s05: story counts, the unit budget and the imputation."""

    def test_components_receive_settings(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run gives the imputer and the budget their options and the seed."""
        compiler = HazusEQInventoryCompiler(
            seed=5,
            n_worlds=1,
            projected_crs=METRIC_CRS,
            imputer_options={'k': 7},
            budget_options={'threshold_sqft': 900.0},
        )
        imputer = mock.Mock(wraps=hc.ObservedDonorImputer)
        budget = mock.Mock(wraps=hc.ACSUnitBudget)
        monkeypatch.setattr(hc, 'ObservedDonorImputer', imputer)
        monkeypatch.setattr(hc, 'ACSUnitBudget', budget)
        compiler.compile(study_area)
        assert imputer.call_args.kwargs == {
            **DEFAULT_IMPUTER_OPTIONS,
            'k': 7,
            'seed': 5,
        }
        assert budget.call_args.kwargs['seed'] == 5
        assert budget.call_args.kwargs['threshold_sqft'] == 900.0

    def test_stories(self, compiled: object) -> None:
        """The height count fills a missing count and replaces NSI's off RES1."""
        result = compiled(n_worlds=1)
        features = features_in(result.inventory, 0)
        records = result.report.records
        stories = records[records['stage'] == 's03_story_counts']
        # RES3C with NSI's 3 stories and a 52 ft height: the height's 4.
        assert features['p01'][STORIES_KEY] == 4
        assert features['p01'][SOURCE_STRING_KEY][STORIES_CODE] == 'R'
        row = stories[stories['asset_id'] == 'p01'].iloc[0]
        assert (row['feature'], row['action'], row['old'], row['new']) == (
            STORIES_KEY,
            'withheld',
            '3',
            '4',
        )
        assert row['source'] == 'nsi'
        # COM1 whose NSI count and 13 ft height agree on 1: the height's 1.
        assert features['c01'][SOURCE_STRING_KEY][STORIES_CODE] == 'R'
        row = stories[stories['asset_id'] == 'c01'].iloc[0]
        assert (row['old'], row['new']) == ('1', '1')
        # RES1 with NSI's 1 story and a 26 ft height: NSI's 1.
        assert features['h02'][STORIES_KEY] == 1
        assert features['h02'][STORIES_FROM_HEIGHT_KEY] == 2
        assert features['h02'][SOURCE_STRING_KEY][STORIES_CODE] == 'N'
        # COM1 with a local count of 2 and a 52 ft height: the local 2.
        assert features['l02'][STORIES_KEY] == 2
        assert features['l02'][STORIES_FROM_HEIGHT_KEY] == 4
        assert features['l02'][SOURCE_STRING_KEY][STORIES_CODE] == 'L'
        # RES1 without an NSI count and a 13 ft height: the height's 1.
        assert features['w02'][STORIES_KEY] == 1
        assert features['w02'][SOURCE_STRING_KEY][STORIES_CODE] == 'R'
        withheld = result.freeze.read('s03_withheld').inventory
        assert withheld['p01'].features == {STORIES_KEY: 'nsi'}
        assert 'h02' not in withheld
        assert 'l02' not in withheld

    def test_imputed_targets(self, compiled: object) -> None:
        """A house and a store without a story count each get an imputed count, and the store gets no units."""
        result = compiled(n_worlds=3)
        nonresidential = set(result.logs['budget_nonresidential']['asset_id'])
        assert nonresidential
        for world in range(3):
            features = features_in(result.inventory, world)
            # RES1 without an NSI count or a height.
            assert features['w03'][SOURCE_STRING_KEY][STORIES_CODE] == 'I'
            # COM1 without an NSI count, whose donor may hold units.
            assert features['c05'][SOURCE_STRING_KEY][STORIES_CODE] == 'I'
            assert UNITS_KEY not in features['c05']
            for asset_id in nonresidential:
                assert features[asset_id][OCCUPANCY_KEY] not in UNIT_COUNT_CLASSES

    def test_local_class_is_a_source_class(self, study_area: object) -> None:
        """A local occupancy class is observed for the budget and the donors."""
        allocate = hc.ACSUnitBudget.allocate
        impute = hc.ObservedDonorImputer.impute_stage
        with (
            mock.patch.object(
                hc.ACSUnitBudget, 'allocate', autospec=True, side_effect=allocate
            ) as allocated,
            mock.patch.object(
                hc.ObservedDonorImputer,
                'impute_stage',
                autospec=True,
                side_effect=impute,
            ) as imputed,
        ):
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
            ).compile(study_area)
        # l01 is a local RES1 over the COM1 of its NSI point.
        assert 'l01' in allocated.call_args.args[2].inventory
        assert 'l01' not in allocated.call_args.args[1].inventory
        assert 'l01' in imputed.call_args.args[2].inventory

    def test_validation(self, study_area: object) -> None:
        """The values the imputer copies from a donor are checked for implausible pairs."""
        assert not hc._is_plausible_pair(
            'a', 0, {OCCUPANCY_KEY: 'RES2', STORIES_KEY: 4}
        )
        assert hc._is_plausible_pair('a', 0, {OCCUPANCY_KEY: 'RES2', STORIES_KEY: 1})
        assert hc._is_plausible_pair('a', 0, {OCCUPANCY_KEY: 'RES2'})
        real = hc.ObservedDonorImputer.impute_stage
        with mock.patch.object(
            hc.ObservedDonorImputer, 'impute_stage', autospec=True, side_effect=real
        ) as impute:
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
            ).compile(study_area)
        assert impute.call_args.kwargs['validate'] is hc._is_plausible_pair

    def test_every_footprint_dropped(
        self, study_area: CompilerInputs, tmp_path: Path
    ) -> None:
        """A run that drops every footprint writes the report rows of every world."""
        # d01 alone: a small footprint far from the mobile-home park, which the
        # budget drops.
        footprints = AssetInventory()
        footprints.add_asset('d01', study_area.footprints.inventory['d01'])
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        ).compile(study_area._replace(footprints=footprints, local=None))
        assert not result.inventory.inventory
        result.write(tmp_path / 'empty.geojson')
        summary = json.loads(
            (tmp_path / 'empty_report_summary.json').read_text(encoding='utf-8')
        )
        assert summary['worlds'] == {'computed': 3, 'written': [0, 1, 2]}
        exclusions = pd.read_parquet(tmp_path / 'empty_report_exclusions.parquet')
        assert sorted(exclusions['world']) == [0, 1, 2]
        assert set(exclusions['category']) == {'dropped'}
        assert written(tmp_path / 'empty.geojson') == {}

    def test_dropped(self, compiled: object) -> None:
        """A small footprint far from the mobile-home park, with no occupancy class, is dropped."""
        result = compiled(n_worlds=3)
        assert 'd01' not in result.inventory.inventory
        rows = result.report.exclusions
        rows = rows[rows['asset_id'] == 'd01']
        # It is dropped in every world.
        assert sorted(rows['world']) == [0, 1, 2]
        assert set(rows['category']) == {'dropped'}
        assert set(rows['stage']) == {'s04_unit_budget'}

    def test_local_values_unchanged(self, compiled: object) -> None:
        """No local value is withheld or replaced, the mobile-home rule's included."""
        result = compiled(n_worlds=3)
        assert math.dist(MH_PARK, FOOTPRINTS['l01'][:2]) < 300.0
        records = result.report.records
        assert 'r05' in set(
            records.loc[records['stage'] == 's04_unit_budget', 'asset_id']
        )
        for world in range(3):
            features = features_in(result.inventory, world)
            for asset_id, local in LOCAL.items():
                for key, value in local.items():
                    # The file keeps a local gross area under a name of its own.
                    kept = LOCAL_GROSS_AREA_KEY if key == GROSS_AREA_KEY else key
                    assert features[asset_id][kept] == value
                    if key in HAZUS_EQ_SOURCE_FEATURES:
                        code = features[asset_id][SOURCE_STRING_KEY]
                        assert code[HAZUS_EQ_SOURCE_FEATURES.index(key)] == 'L'
        touched = set(zip(records['asset_id'], records['feature']))
        assert not touched & {(a, k) for a, local in LOCAL.items() for k in local}

    def test_every_asset_compiled_or_listed(self, compiled: object) -> None:
        """In each world every footprint is compiled or has an exclusion row."""
        result = compiled(n_worlds=3)
        exclusions = result.report.exclusions
        for world in range(3):
            features = features_in(result.inventory, world)
            compiled_ids = {a for a, f in features.items() if not f[EXCLUDED_KEY]}
            listed = set(exclusions.loc[exclusions['world'] == world, 'asset_id'])
            assert not compiled_ids & listed
            assert compiled_ids | listed == set(FOOTPRINTS)
        for column in ('stage', 'category', 'detail'):
            assert exclusions[column].map(bool).all()

    def test_parts_of_s03_to_s05(self, compiled: object) -> None:
        """Each part and record of steps s03 to s05 holds only what its kind says."""
        result = compiled(n_worlds=3)
        read = result.freeze.read
        kept = set(result.inventory.inventory)
        assert set(read('s04_dropped').inventory) == {'d01'}
        assert all(
            a.features == {DROPPED_KEY: True}
            for a in read('s04_dropped').inventory.values()
        )
        assert {
            (key, value)
            for asset in read('s03_withheld').inventory.values()
            for key, value in asset.features.items()
        } == {(STORIES_KEY, 'nsi')}
        # The mobile-home rule's relabel of an occupancy class the join derived.
        assert read('s04_withheld').inventory['r05'].features == {
            OCCUPANCY_KEY: 'derived'
        }
        assert {
            (key, value)
            for asset in read('s04_withheld').inventory.values()
            for key, value in asset.features.items()
        } == {(OCCUPANCY_KEY, 'derived')}
        assert {
            key
            for a in read('s03_story_counts').inventory.values()
            for key in a.features
        } == {STORIES_FROM_HEIGHT_KEY, STORIES_KEY}
        for stage in ('s04_unit_budget', 's05_imputation'):
            imputed = read(stage)
            assert imputed.n_pw == 3
            assert set(imputed.inventory) == kept
            assert {k for a in imputed.inventory.values() for k in a.features} <= {
                OCCUPANCY_KEY,
                UNITS_KEY,
                STORIES_KEY,
            }
        # The budget gives no story count, and the two parts share no value.
        budget = read('s04_unit_budget').inventory
        imputation = read('s05_imputation').inventory
        assert all(STORIES_KEY not in a.features for a in budget.values())
        assert all(
            not set(budget[a].features) & set(imputation[a].features) for a in kept
        )
        for stage in (
            's03_story_counts',
            's03_withheld',
            's04_dropped',
            's04_withheld',
            's04_unit_budget',
            's05_imputation',
        ):
            for asset in read(stage).inventory.values():
                assert len(asset.coordinates) == 1, stage
        assert set(read('s05_view').inventory) == kept


class TestWorlds:
    """A world does not depend on the number of worlds compiled."""

    def test_merge_keys_do_not_depend_on_n_worlds(self, study_area: object) -> None:
        """World 0's last merged inventory holds the same features at one world and seven."""
        real = hc._WorldSources.merge_world
        keys = {}
        for n_worlds in (1, 7):
            merges: list = []
            with mock.patch.object(
                hc._WorldSources,
                'merge_world',
                autospec=True,
                side_effect=lambda sources, world, _merges=merges, **kwargs: (
                    _merges.append((world, real(sources, world, **kwargs)))
                    or _merges[-1][1]
                ),
            ):
                HazusEQInventoryCompiler(
                    seed=SEED, n_worlds=n_worlds, projected_crs=METRIC_CRS
                ).compile(study_area)
            last = [merged for world, merged in merges if world == 0][-1]
            keys[n_worlds] = {
                asset_id: set(asset.features)
                for asset_id, asset in last.inventory.items()
            }
        assert keys[1] == keys[7]

    def test_world_unchanged_when_n_worlds_changes(self, compiled: object) -> None:
        """World 2 of three worlds and of five are the same, rows included."""
        three, five = compiled(n_worlds=3), compiled(n_worlds=5)
        assert features_in(three.inventory, 2) == features_in(five.inventory, 2)
        for frame in ('records', 'exclusions'):
            rows = rows_of(getattr(three.report, frame), 2)
            assert rows == rows_of(getattr(five.report, frame), 2)
        assert rows_of(three.report.records, 2)
        assert rows_of(three.report.exclusions, 2)

    def test_worlds_differ(self, compiled: object) -> None:
        """The study area's imputed values differ between worlds somewhere."""
        result = compiled(n_worlds=3)
        assert features_in(result.inventory, 0) != features_in(result.inventory, 1)

    @pytest.mark.parametrize('world', [3, 9])
    def test_one_world(self, compiled: object, tmp_path: object, world: int) -> None:
        """A file of one world holds that world's values and the rows of that world."""
        result = compiled(n_worlds=10)
        result.write(
            tmp_path / 'one.geojson',
            world=world,
            keep_partly_excluded=True,
            extra_properties=ALL_PROPERTIES,
        )
        properties = written(tmp_path / 'one.geojson')
        expected = features_in(result.inventory, world)
        assert set(properties) == {
            asset_id
            for asset_id, values in expected.items()
            if not values[EXCLUDED_KEY]
        }
        for asset_id, values in properties.items():
            for key, value in expected[asset_id].items():
                held = values.get(key)
                assert (
                    not isinstance(held, list)
                    or key in result.inventory.vector_features
                )
                assert held == value or (
                    isinstance(value, float) and math.isclose(held, value)
                ), (asset_id, key)
        records = pd.read_parquet(tmp_path / 'one_report.parquet')
        exclusions = pd.read_parquet(tmp_path / 'one_report_exclusions.parquet')
        assert set(records['world']) == {world}
        assert set(exclusions['world']) == {world}
        summary = json.loads(
            (tmp_path / 'one_report_summary.json').read_text(encoding='utf-8')
        )
        assert summary['worlds'] == {'computed': 10, 'written': [world]}
        imputer_log = pd.read_parquet(tmp_path / 'one_imputer.parquet')
        assert set(imputer_log['world']) == {world}

    def test_per_world_lists(self, compiled: object, tmp_path: object) -> None:
        """The file holds a list where worlds differ and a scalar where they agree."""
        result = compiled(n_worlds=3)
        result.write(tmp_path / 'run.geojson')
        properties = written(tmp_path / 'run.geojson')
        assert properties['h01'][OCCUPANCY_KEY] == 'RES1'
        lists = [
            value
            for values in properties.values()
            for key, value in values.items()
            if key in HAZUS_EQ_SOURCE_FEATURES and isinstance(value, list)
        ]
        assert lists
        assert all(len(value) == 3 for value in lists)


class TestYearsAndTypes:
    """Steps s06 to s08: years, pairs, design levels, types and height classes."""

    def test_components_receive_worlds(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each per-world component is told the number of the world it computes."""
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=6, projected_crs=METRIC_CRS
        )
        spies = {}
        for name in (
            'ACSYearBuiltSampler',
            'HazusEQStructureTypeInferer',
            'PairRedraw',
        ):
            spies[name] = mock.Mock(wraps=getattr(hc, name))
            monkeypatch.setattr(hc, name, spies[name])
        compiler.compile(study_area)
        for spy in spies.values():
            worlds = Counter(c.kwargs['world'] for c in spy.call_args_list)
            assert set(worlds) == set(range(6))
            assert len(set(worlds.values())) == 1

    def test_frozen_inputs_cross_the_boundary(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In each world, the year fit and the re-draw's donors read that world's realization of s05_view."""
        # The mocks record the inventories they are called with, not copies,
        # so the comparison after the run also sees a change made to them
        # after the call.
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        )
        redraw = mock.Mock(wraps=hc.PairRedraw)
        monkeypatch.setattr(hc, 'PairRedraw', redraw)
        real = hc.ACSYearBuiltSampler.fit
        with mock.patch.object(
            hc.ACSYearBuiltSampler, 'fit', autospec=True, side_effect=real
        ) as fit:
            result = compiler.compile(study_area)
        step4 = result.freeze.read('s05_view')
        occupancy = HAZUS_EQ_SOURCE_FEATURES.index(OCCUPANCY_KEY)
        for world in range(3):
            frozen = features_in(step4, world)
            assert features_in(fit.call_args_list[world].args[1], 0) == frozen
            with_occupancy_source = {
                a: values
                for a, values in frozen.items()
                if values[SOURCE_STRING_KEY][occupancy] in 'LNR'
                and STORIES_KEY in values
            }
            donors = redraw.call_args_list[world].args[1]
            assert features_in(donors, 0) == with_occupancy_source

    def test_changed_view_raises_at_the_manifest(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A component that changes s05_view after reading it stops the run."""
        real = hc.StageFreeze.read

        def changing(self: object, stage: str) -> AssetInventory:
            """
            Return a frozen stage, changing s05_view.

            Args:
                self (object):
                    The freeze the stage is read from.
                stage (str):
                    The name of the stage to read.

            Returns:
                AssetInventory:
                    The frozen stage, with h01's story count set to 9 where
                    ``stage`` is ``'s05_view'``.
            """
            held = real(self, stage)
            if stage == 's05_view':
                held.inventory['h01'].features[STORIES_KEY] = 9
            return held

        monkeypatch.setattr(hc.StageFreeze, 'read', changing)
        with pytest.raises(RuntimeError, match='s05_view'):
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
            ).compile(study_area)

    def test_parts_of_s06_to_s08(self, compiled: object) -> None:
        """Each per-world part and record of steps s06 to s08 holds every footprint the run kept."""
        result = compiled(n_worlds=3)
        kept = set(result.inventory.inventory)
        for stage in (
            's06_residential_year',
            's07_withheld',
            's07_pair_redraw',
            's07_derived',
            's08_structure',
            's08_derived',
        ):
            part = result.freeze.read(stage)
            assert part.n_pw == 3
            assert set(part.inventory) == kept
            for asset in part.inventory.values():
                assert len(asset.coordinates) == 1, stage
        for stage, features in (
            ('s06_residential_year', {YEAR_BUILT_KEY}),
            ('s08_structure', {STRUCTURE_TYPE_KEY}),
        ):
            assert {
                key
                for asset in result.freeze.read(stage).inventory.values()
                for key in asset.features
            } == features
        for stage, features in (
            ('s07_pair_redraw', {OCCUPANCY_KEY, STORIES_KEY}),
            ('s07_derived', {STORIES_KEY}),
            ('s08_derived', {DESIGN_LEVEL_KEY, HEIGHT_CLASS_KEY}),
        ):
            assert {
                key
                for asset in result.freeze.read(stage).inventory.values()
                for key in asset.features
            } <= features

    def test_the_pair_redraw_reads_the_years(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In each world, the pair re-draw reads the world merged with s06's years, coded drawn."""
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        )
        # The re-draw is watched through its method, which receives the
        # world's merged inventory.
        monkeypatch.setattr(
            hc.PairRedraw,
            'redraw',
            mock.create_autospec(
                hc.PairRedraw.redraw, side_effect=hc.PairRedraw.redraw
            ),
        )
        result = compiler.compile(study_area)
        years = result.freeze.read('s06_residential_year')
        calls = hc.PairRedraw.redraw.call_args_list
        assert len(calls) == 3
        drawn_somewhere = False
        for world, call in enumerate(calls):
            merged_world = call.args[1]
            for asset_id, values in features_in(years, world).items():
                if YEAR_BUILT_KEY in values:
                    drawn_somewhere = True
                    held = merged_world.inventory[asset_id].features
                    assert held[YEAR_BUILT_KEY] == values[YEAR_BUILT_KEY]
                    assert held[SOURCE_STRING_KEY][YEAR_CODE] == 'D'
        assert drawn_somewhere

    def test_the_structure_draws_read_the_redraw(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In each world, the structure draws read the world merged with s07's re-drawn values, coded imputed."""
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        )
        # The draws are watched through _assign_structure, which receives the
        # world's merged inventory; its first calls in a run are s08's.
        monkeypatch.setattr(
            hc.HazusEQInventoryCompiler,
            '_assign_structure',
            mock.create_autospec(
                hc.HazusEQInventoryCompiler._assign_structure,
                side_effect=hc.HazusEQInventoryCompiler._assign_structure,
            ),
        )
        result = compiler.compile(study_area)
        redrawn = result.freeze.read('s07_pair_redraw')
        calls = hc.HazusEQInventoryCompiler._assign_structure.call_args_list[:3]
        seen = False
        for world, call in enumerate(calls):
            merged_world = call.args[4]
            for asset_id, values in features_in(redrawn, world).items():
                if OCCUPANCY_KEY in values:
                    seen = True
                    held = merged_world.inventory[asset_id].features
                    assert held[OCCUPANCY_KEY] == values[OCCUPANCY_KEY]
                    assert held[SOURCE_STRING_KEY][OCCUPANCY_CODE] == 'I'
        assert seen

    def test_loop_in_the_run(self, compiled: object) -> None:
        """An NSI building type that no structure type of the asset's cell matches is withheld, and the asset compiles."""
        result = compiled(n_worlds=3)
        records = result.report.records
        rows = records[
            (records['asset_id'] == 'p01') & (records['stage'] == 's08_resolution')
        ]
        assert sorted(rows['world']) == [0, 1, 2]
        assert set(rows['feature']) == {BUILDING_TYPE_KEY}
        assert set(rows['source']) == {'nsi'}
        for world in range(3):
            features = features_in(result.inventory, world)['p01']
            assert features[EXCLUDED_KEY] is False
            assert features[STRUCTURE_TYPE_KEY] not in ('W1', 'W2')
            assert (
                features[BUILDING_TYPE_KEY]
                == hc._LETTER_OF_TYPE[features[STRUCTURE_TYPE_KEY]]
            )

    def test_building_type(self, compiled: object) -> None:
        """A local letter is kept, every other follows the structure type, and a contradicting local letter excludes the building."""
        result = compiled(n_worlds=3)
        local = {
            asset_id: values[BUILDING_TYPE_KEY]
            for asset_id, values in LOCAL.items()
            if BUILDING_TYPE_KEY in values
        }
        derived = set()
        for world in range(3):
            for asset_id, values in features_in(result.inventory, world).items():
                letter = values.get(BUILDING_TYPE_KEY)
                if asset_id in local:
                    assert letter == local[asset_id]
                elif not values[EXCLUDED_KEY] or letter is not None:
                    assert letter == hc._LETTER_OF_TYPE[values[STRUCTURE_TYPE_KEY]]
                    derived.add(asset_id)
        # q03's local W stands beside its local C1, NSI's C is not written,
        # and q03 is excluded as a conflict in each world, without its value
        # and gross area, and with no notice.
        assert local['q03'] != hc._LETTER_OF_TYPE[LOCAL['q03'][STRUCTURE_TYPE_KEY]]
        assert derived & UNREACHED
        rows = result.report.exclusions
        conflicts = rows[rows['stage'] == 's14_building_type']
        assert set(zip(conflicts['asset_id'], conflicts['world'])) == {
            ('q03', world) for world in range(3)
        }
        assert set(conflicts['category']) == {'conflict'}
        for world in range(3):
            q03 = features_in(result.inventory, world)['q03']
            assert q03[EXCLUDED_KEY] is True
            assert REPLACEMENT_COST_KEY not in q03
            assert GROSS_AREA_KEY not in q03
        # q03 remains a donor of s05.
        assert 'q03' in set(result.logs['imputer']['donor_id'])
        assert not any(
            BUILDING_TYPE_KEY in notice
            for notice in result.report.summary()['notices']
        )

    def test_letters_disagreeing_are_excluded(self, study_area: object) -> None:
        """A building whose local letter is not its type's is excluded where it is not yet."""
        # u06 gets a local C beside a local W1, as q03 holds a local W beside
        # a local C1, and is excluded from world 1; h04's local W agrees.
        local = build_inputs().local
        x, y, side = FOOTPRINTS['u06'][:3]
        local.add_asset(
            'u06',
            Asset(
                'u06',
                square(x, y, side),
                {
                    STRUCTURE_TYPE_KEY: 'W1',
                    BUILDING_TYPE_KEY: 'C',
                    YEAR_BUILT_KEY: 1980,
                },
            ),
        )
        real = hc.CompilationReport.set_world_count

        def counted(report: CompilationReport, count: int) -> None:
            """
            Set the world count, then exclude u06 from world 1.

            Args:
                report (CompilationReport):
                    The run's report.
                count (int):
                    The number of worlds computed.
            """
            real(report, count)
            report.record_exclusion(
                'u06',
                world=1,
                stage='s08_resolution',
                category='conflict',
                detail='x',
            )

        with mock.patch.object(
            hc.CompilationReport,
            'set_world_count',
            autospec=True,
            side_effect=counted,
        ):
            result = HazusEQInventoryCompiler(
                seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
            ).compile(study_area._replace(local=local))
        rows = result.report.exclusions
        conflicts = rows[rows['stage'] == 's14_building_type']
        assert set(zip(conflicts['asset_id'], conflicts['world'])) == {
            ('q03', 0),
            ('q03', 1),
            ('q03', 2),
            ('u06', 0),
            ('u06', 2),
        }
        assert 'h04' not in set(rows['asset_id'])

    def test_pairing_covers_every_type(self) -> None:
        """Every Hazus structure type has a general type letter."""
        assert set(hc._LETTER_OF_TYPE) == set(get_hazus_height_classes_by_type())

    def test_completed_tables(self, compiled: object) -> None:
        """An NSI EDU1 at five stories, for which Hazus prints no structure types, compiles."""
        result = compiled(n_worlds=3)
        printed = hc.HazusEQStructureTypeInferer('California', seed=SEED)
        for world in range(3):
            features = features_in(result.inventory, world)['c02']
            assert features[EXCLUDED_KEY] is False
            assert features[STRUCTURE_TYPE_KEY]
            assert printed.cell_is_unpublished(features)

    def test_pair_redraw(self, compiled: object) -> None:
        """The NSI pair RES2 at four stories is replaced by a donor's pair."""
        result = compiled(n_worlds=3)
        records = result.report.records
        rows = records[
            (records['asset_id'] == 'c03') & (records['stage'] == 's07_pair_redraw')
        ]
        assert set(rows['action']) == {'withheld', 'redrawn'}
        assert set(rows.loc[rows['action'] == 'withheld', 'source']) == {'nsi'}
        record = result.freeze.read('s07_withheld').get_world_realization(0)
        assert set(record.inventory['c03'].features.values()) == {'nsi'}
        for world in range(3):
            features = features_in(result.inventory, world)['c03']
            codes = features[SOURCE_STRING_KEY]
            assert (codes[0], codes[STORIES_CODE]) == ('I', 'I')
            pair = (features[OCCUPANCY_KEY], features[STORIES_KEY])
            assert pair != ('RES2', 4)

    def test_carve_out(self, compiled: object, study_area: object) -> None:
        """A drawn Pre-Code W1 leaves as Moderate-Code in California alone."""
        result = compiled(n_worlds=3)
        records = result.report.records
        rows = records[records['stage'] == 's08_carve_out']
        assert not rows.empty
        for row in rows.itertuples():
            features = features_in(result.inventory, row.world)[row.asset_id]
            assert features[STRUCTURE_TYPE_KEY] == 'W1'
            assert features[DESIGN_LEVEL_KEY] == 'Moderate-Code'
            assert (row.old, row.new) == ('Pre-Code', 'Moderate-Code')
        oregon = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, state='Oregon', projected_crs=METRIC_CRS
        ).compile(study_area)
        assert 's08_carve_out' not in set(oregon.report.records['stage'])
        levels = [
            (features[STRUCTURE_TYPE_KEY], features[DESIGN_LEVEL_KEY])
            for world in range(3)
            for features in features_in(oregon.inventory, world).values()
            if STRUCTURE_TYPE_KEY in features and DESIGN_LEVEL_KEY in features
        ]
        assert ('W1', 'Pre-Code') in levels

    def test_schema_check(self, compiled: object) -> None:
        """A local URM of 1990, which gives High-Code, fails the schema everywhere."""
        result = compiled(n_worlds=3)
        rows = result.report.exclusions
        rows = rows[rows['asset_id'] == 'l02']
        assert sorted(rows['world']) == [0, 1, 2]
        assert set(rows['category']) == {'fails_schema'}
        assert result.inventory.inventory['l02'].features[EXCLUDED_KEY] is True
        assert (
            result.inventory.inventory['l02'].features[DESIGN_LEVEL_KEY]
            == 'High-Code'
        )
        # The URM, C3 and S5 clause refuses it, and no feature is missing.
        for detail in rows['detail']:
            assert DESIGN_LEVEL_KEY in detail
            assert 'required property' not in detail

    def test_local_type(self, compiled: object) -> None:
        """A local type the schema accepts takes its derived height class."""
        result = compiled(n_worlds=3)
        height = HAZUS_EQ_SOURCE_FEATURES.index(HEIGHT_CLASS_KEY)
        kind = HAZUS_EQ_SOURCE_FEATURES.index(STRUCTURE_TYPE_KEY)
        rows = result.report.exclusions
        assert 'fails_schema' not in set(
            rows.loc[rows['asset_id'] == 'q03', 'category']
        )
        for world in range(3):
            features = features_in(result.inventory, world)['q03']
            assert features[HEIGHT_CLASS_KEY] == 'Low-Rise'
            codes = features[SOURCE_STRING_KEY]
            assert (codes[kind], codes[height]) == ('L', 'R')

    def test_building_type_sources(self, study_area: object) -> None:
        """A building type local and NSI both hold is labeled local for the loop."""
        real = hc.ResolutionLoop.resolve
        with mock.patch.object(
            hc.ResolutionLoop, 'resolve', autospec=True, side_effect=real
        ) as resolve:
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=2, projected_crs=METRIC_CRS
            ).compile(study_area)
        for call in resolve.call_args_list[:2]:
            labels = call.kwargs['building_type_sources']
            assert labels['h04'] == 'local'
            assert labels['h01'] == 'nsi'

    def test_status_drawn_per_world(self, compiled: object) -> None:
        """A status drawn per world takes its year from s06 or from s09."""
        result = compiled(n_worlds=3)
        year = HAZUS_EQ_SOURCE_FEATURES.index(YEAR_BUILT_KEY)
        mixed = [
            asset_id
            for asset_id, asset in result.inventory.inventory.items()
            if isinstance(asset.features[SOURCE_STRING_KEY], list)
            and {codes[year] for codes in asset.features[SOURCE_STRING_KEY]}
            == {'D', 'I'}
        ]
        assert mixed
        for asset_id in mixed:
            features = result.inventory.inventory[asset_id].features
            assert features[EXCLUDED_KEY] is False
            for world, codes in enumerate(features[SOURCE_STRING_KEY]):
                residential = (
                    features_in(result.inventory, world)[asset_id][OCCUPANCY_KEY]
                    in UNIT_COUNT_CLASSES
                )
                assert codes[year] == ('D' if residential else 'I')

    def test_year_refused(self, study_area: object) -> None:
        """Without year tables, every residential asset is excluded at the year."""
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=2, projected_crs=METRIC_CRS
        ).compile(
            study_area._replace(
                b25034_block_groups=study_area.b25034_block_groups.iloc[0:0],
                b25034_tracts=study_area.b25034_tracts.iloc[0:0],
            )
        )
        rows = result.report.exclusions
        rows = rows[rows['stage'] == 's06_residential_year']
        assert set(rows['category']) == {'missing_input'}
        assert set(rows['world']) == {0, 1}
        assert 'h01' in set(rows['asset_id'])
        assert 'u07' not in set(rows['asset_id'])

    def test_refusals_by_stage(self, study_area: object) -> None:
        """Each refusal is recorded at its stage, with its category."""
        local = build_inputs().local
        for asset_id, features in (
            # residential, without a unit count: the year sampler refuses it
            ('q02', {OCCUPANCY_KEY: 'RES1'}),
            # a year the design level inferer cannot read
            ('h05', {YEAR_BUILT_KEY: 'unknown'}),
            # a type without height classes
            ('h07', {STRUCTURE_TYPE_KEY: 'XX'}),
        ):
            x, y, side = FOOTPRINTS[asset_id][:3]
            local.add_asset(asset_id, Asset(asset_id, square(x, y, side), features))
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=2, projected_crs=METRIC_CRS
        ).compile(study_area._replace(local=local))
        rows = result.report.exclusions.set_index(['asset_id', 'world'])
        for asset_id, stage in (
            ('q02', 's06_residential_year'),
            ('h05', 's08_design_level'),
            ('h07', 's08_height_class'),
        ):
            for world in range(2):
                row = rows.loc[(asset_id, world)]
                assert (row['stage'], row['category'], row['diagnosis']) == (
                    stage,
                    'missing_input',
                    'invalid',
                )

    def test_year_fit_log(self, compiled: object, tmp_path: object) -> None:
        """The written year fit log names the block group of each row."""
        compiled(n_worlds=3).write(tmp_path / 'run.geojson')
        fit = pd.read_parquet(tmp_path / 'run_year_fit.parquet')
        assert {BLOCK_GROUP_KEY, 'world'} <= set(fit.columns)


class TestYearAndValueOfTheRest:
    """Steps s09 and s13: the year of a non-residential building, and every replacement cost."""

    def test_components_receive_worlds(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both draw per world with the imputer's k and cap, s13 its weights too."""
        compiler = HazusEQInventoryCompiler(
            seed=SEED,
            n_worlds=6,
            projected_crs=METRIC_CRS,
            imputer_options={'cap_m': 1500.0, 'stories_weight': 0.5},
        )
        value_imputer = hc.ReplacementCostImputer
        sampler = mock.Mock(wraps=hc.ResidentialNeighborYearSampler)
        imputer = mock.Mock(wraps=value_imputer)
        monkeypatch.setattr(hc, 'ResidentialNeighborYearSampler', sampler)
        monkeypatch.setattr(hc, 'ReplacementCostImputer', imputer)
        real = value_imputer.impute_stage
        with mock.patch.object(
            value_imputer,
            'impute_stage',
            autospec=True,
            side_effect=real,
        ) as impute:
            compiler.compile(study_area)
        assert [c.kwargs['world'] for c in sampler.call_args_list] == list(range(6))
        assert [c.kwargs['world'] for c in impute.call_args_list] == list(range(6))
        for spy in (sampler, imputer):
            assert {
                (c.kwargs['k'], c.kwargs['cap_m']) for c in spy.call_args_list
            } == {(DEFAULT_IMPUTER_OPTIONS['k'], 1500.0)}
        assert {c.args[2:4] for c in impute.call_args_list} == {
            ('s08_view', 's05_donors')
        }
        options = imputer.call_args.kwargs
        assert (
            options['location_scale_m'],
            options['area_weight'],
            options['stories_weight'],
        ) == (
            DEFAULT_IMPUTER_OPTIONS['location_scale_m'],
            DEFAULT_IMPUTER_OPTIONS['area_weight'],
            0.5,
        )

    def test_targets_from_the_frozen_stage(self, study_area: object) -> None:
        """In each world, s09's targets come from that world's realization of s08_view."""
        real = hc.ResidentialNeighborYearSampler.check
        with mock.patch.object(
            hc.ResidentialNeighborYearSampler,
            'check',
            autospec=True,
            side_effect=real,
        ) as check:
            result = HazusEQInventoryCompiler(
                seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
            ).compile(study_area)
        step6 = result.freeze.read('s08_view')
        for world in range(3):
            frozen = features_in(step6, world)
            targets = features_in(check.call_args_list[world].args[1], 0)
            assert targets
            assert all(targets[a] == frozen[a] for a in targets)
            assert all(YEAR_BUILT_KEY not in values for values in targets.values())

    def test_year_from_a_neighbor(self, compiled: object) -> None:
        """A non-residential year is its logged donor's year, coded I."""
        result = compiled(n_worlds=3)
        year = HAZUS_EQ_SOURCE_FEATURES.index(YEAR_BUILT_KEY)
        picks = result.logs['neighbor_year']
        assert not picks.empty
        for row in picks.itertuples():
            features = features_in(result.inventory, row.world)
            assert features[row.asset_id][YEAR_BUILT_KEY] == row.YearBuilt
            assert features[row.donor_id][YEAR_BUILT_KEY] == row.YearBuilt
            assert features[row.asset_id][SOURCE_STRING_KEY][year] == 'I'
        # The donors are those of s08_view, years drawn in s06 among them.
        drawn = {
            features_in(result.inventory, row.world)[row.donor_id][
                SOURCE_STRING_KEY
            ][year]
            for row in picks.itertuples()
        }
        assert 'D' in drawn
        # COM1 with NSI's occupancy class, given a year and a structure type in
        # every world.
        assert set(picks.loc[picks['asset_id'] == 'c01', 'world']) == {0, 1, 2}
        for world in range(3):
            assert features_in(result.inventory, world)['c01'][STRUCTURE_TYPE_KEY]

    def test_no_class_no_year(self, compiled: object) -> None:
        """A footprint beyond the donor cap of every donor has no occupancy class or year in any world."""
        result = compiled(n_worlds=3)
        rows = result.report.exclusions
        rows = rows[rows['asset_id'] == 'f01']
        assert sorted(rows['world']) == [0, 1, 2]
        assert set(rows['stage']) == {'s09_nonresidential_year'}
        assert set(rows['category']) == {'missing_input'}
        assert OCCUPANCY_KEY not in features_in(result.inventory, 0)['f01']

    def test_values(self, compiled: object) -> None:
        """Each value is its log's rate times gross area, coded by its outcome."""
        result = compiled(n_worlds=3)
        value = HAZUS_EQ_SOURCE_FEATURES.index(REPLACEMENT_COST_KEY)
        log = result.logs['value']
        donors = result.freeze.read('s05_donors')
        seen = set()
        for row in log.itertuples():
            features = features_in(result.inventory, row.world)[row.asset_id]
            if features[EXCLUDED_KEY] or row.outcome == OUTCOME_LOCAL_VALUE:
                continue
            assert math.isclose(
                features[REPLACEMENT_COST_KEY],
                row.rate * row.footprint_gross_area_sqft,
            )
            code = features[SOURCE_STRING_KEY][value]
            assert code == OUTCOME_VALUE_CODES[row.outcome]
            seen.add(row.outcome)
            if row.outcome == OUTCOME_IMPUTATION_DONOR:
                donor = features_in(donors, row.world)[row.asset_id]
                assert donor[IMPUTATION_DONOR_KEY] == row.donor_id
        assert {OUTCOME_OWN_RATE, OUTCOME_IMPUTATION_DONOR, OUTCOME_DONOR} <= seen
        for world in range(3):
            features = features_in(result.inventory, world)
            # NSI's own house: its fullrep over its sqft, times its gross area
            # computed from the footprint, which GrossArea holds.
            gross_area = (
                features['h01'][PLAN_AREA_KEY] * features['h01'][STORIES_KEY]
            )
            assert math.isclose(features['h01'][GROSS_AREA_KEY], gross_area)
            assert math.isclose(
                features['h01'][REPLACEMENT_COST_KEY], 350000.0 / 1400.0 * gross_area
            )
            assert features['h01'][SOURCE_STRING_KEY][value] == 'R'
            # A local value is kept, coded L, whatever NSI states.
            assert features['c02'][REPLACEMENT_COST_KEY] == 5.0e6
            assert features['c02'][SOURCE_STRING_KEY][value] == 'L'

    def test_local_value_gives_a_rate(self, compiled: object) -> None:
        """A building whose donor has a local replacement cost takes the donor's rate: that cost divided by the donor's NSI sqft."""
        result = compiled(n_worlds=10)
        log = result.logs['value']
        value = HAZUS_EQ_SOURCE_FEATURES.index(REPLACEMENT_COST_KEY)
        # c02 holds a local value of 5.0e6 beside NSI's 40,000 sq ft.
        own = log[log['asset_id'] == 'c02']
        assert set(own['outcome']) == {OUTCOME_LOCAL_VALUE}
        assert set(own['rate']) == {5.0e6 / 40000.0}
        # The buildings that named c02 as their s05 donor take c02's rate.
        served = log[log['donor_id'] == 'c02']
        assert set(zip(served['asset_id'], served['world'])) == {
            ('u11', 3),
            ('u13', 4),
            ('u13', 9),
            ('n06', 5),
        }
        assert set(served['rate']) == {5.0e6 / 40000.0}
        for row in served.itertuples():
            features = features_in(result.inventory, row.world)[row.asset_id]
            assert not features[EXCLUDED_KEY]
            assert math.isclose(
                features[REPLACEMENT_COST_KEY], 125.0 * features[GROSS_AREA_KEY]
            )
            assert features[SOURCE_STRING_KEY][value] == 'I'

    def test_local_value_rate_ignores_a_local_gross_area(
        self, study_area: object
    ) -> None:
        """A local value's rate divides by NSI's sqft even beside a local `GrossArea`."""
        local = build_inputs().local
        local.inventory['c02'].features[GROSS_AREA_KEY] = 10000.0
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        ).compile(study_area._replace(local=local))
        log = result.logs['value']
        assert set(log[log['asset_id'] == 'c02']['rate']) == {5.0e6 / 40000.0}

    def test_local_gross_area_is_no_rate(self, compiled: object) -> None:
        """A local `GrossArea` changes neither a rate nor the value it gives."""
        result = compiled(n_worlds=3)
        log = result.logs['value'].set_index(['asset_id', 'world'])
        # l01 and h01 are the same NSI house on footprints of one size, and
        # l01 has a local gross area of 777 sq ft beside NSI's 1,400.
        for world in range(3):
            features = features_in(result.inventory, world)
            assert log.loc[('l01', world), 'rate'] == 350000.0 / 1400.0
            assert log.loc[('l01', world), 'rate'] == log.loc[('h01', world), 'rate']
            # Equal up to the plan areas the projection gives the two squares.
            assert math.isclose(
                features['l01'][REPLACEMENT_COST_KEY],
                features['h01'][REPLACEMENT_COST_KEY],
                rel_tol=1e-5,
            )
        served = result.logs['value']
        served = served[served['donor_id'] == 'l01']
        assert not served.empty
        assert set(served['rate']) == {350000.0 / 1400.0}

    def test_no_class_fails_the_schema_first(self, study_area: object) -> None:
        """A footprint without an occupancy class fails the schema, and s13 files no exclusion for it."""
        local = build_inputs().local
        x, y, side = FOOTPRINTS['f01'][:3]
        local.add_asset(
            'f01',
            Asset(
                'f01',
                square(x, y, side),
                {YEAR_BUILT_KEY: 1970, STRUCTURE_TYPE_KEY: 'W1', STORIES_KEY: 1},
            ),
        )
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        ).compile(study_area._replace(local=local))
        rows = result.report.exclusions
        rows = rows[rows['asset_id'] == 'f01']
        assert list(zip(rows['stage'], rows['category'])) == [
            ('s11_schema_check', 'fails_schema')
        ]
        assert OCCUPANCY_KEY in rows['detail'].iloc[0]
        log = result.logs['value']
        assert list(log.loc[log['asset_id'] == 'f01', 'outcome']) == ['no_class']

    def test_no_value_excluded(self, compiled: object) -> None:
        """A building without a donor is excluded as no_plausible_value, and any other building without a value as missing_input."""
        result = compiled(n_worlds=3)
        log = result.logs['value']
        rows = result.report.exclusions.set_index(['asset_id', 'world'])
        without = log[~log['outcome'].isin(list(OUTCOME_VALUE_CODES))]
        assert not without.empty
        for row in without.itertuples():
            assert (row.asset_id, row.world) in rows.index
            features = features_in(result.inventory, row.world)[row.asset_id]
            assert features[EXCLUDED_KEY] is True
            assert REPLACEMENT_COST_KEY not in features
        valued = rows[rows['stage'] == 's13_replacement_cost']
        assert set(valued.index) <= set(zip(without['asset_id'], without['world']))
        outcomes = without.set_index(['asset_id', 'world'])['outcome']
        expected = [
            'no_plausible_value'
            if outcomes[key] == OUTCOME_NO_DONOR
            else 'missing_input'
            for key in valued.index
        ]
        assert list(valued['category']) == expected
        assert 'no_plausible_value' in expected

    def test_schema_check_takes_no_value(self, compiled: object) -> None:
        """A building the schema refuses takes no value, NSI's included."""
        result = compiled(n_worlds=3)
        rows = result.report.exclusions
        assert set(rows.loc[rows['asset_id'] == 'u08', 'category']) == {
            'fails_schema'
        }
        log = result.logs['value']
        assert set(log.loc[log['asset_id'] == 'u08', 'world']) == {0, 1, 2}
        for world in range(3):
            features = features_in(result.inventory, world)
            # u08 has no NSI point, and l02 the join's sum over two.
            assert REPLACEMENT_COST_KEY not in features['u08']
            assert REPLACEMENT_COST_KEY not in features['l02']
        withheld = result.freeze.read('s13_withheld').inventory
        assert withheld['l02'].features == {REPLACEMENT_COST_KEY: 'derived'}
        assert withheld['h01'].features == {REPLACEMENT_COST_KEY: 'nsi'}
        assert 'c02' not in withheld


class TestStructureStages:
    """s08 and s10 file their rows under report stages of their own."""

    def test_s10_files_under_its_own_stages(self) -> None:
        """A building s09 gives a year is filed under s10, every other under s08."""
        # c05 is non-residential and has no year. NSI's MH, which no COM1
        # structure type has, makes s10's loop withhold the letter.
        inputs = build_inputs()
        inputs.nsi_points.inventory[28].features['bldgtype'] = 'MH'
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        ).compile(inputs)
        picks = result.logs['neighbor_year']
        dated_by_s09 = set(zip(picks['asset_id'], picks['world']))
        seen = set()
        for frame in (result.report.records, result.report.exclusions):
            for row in frame.itertuples():
                prefix = row.stage[:4]
                if prefix not in ('s08_', 's10_'):
                    continue
                seen.add(prefix)
                in_s10 = (row.asset_id, row.world) in dated_by_s09
                assert prefix == ('s10_' if in_s10 else 's08_'), row
        assert seen == {'s08_', 's10_'}
        records = result.report.records
        rows = records[records['asset_id'] == 'c05']
        assert list(zip(rows['world'], rows['stage'], rows['action'])) == [
            (world, 's10_resolution', 'withheld') for world in range(3)
        ]


class TestLastSteps:
    """The schema check, the gross area, the value, the letter and the output."""

    def test_order_in_each_world(self, study_area: object) -> None:
        """In each world the schema check runs first, then s12, s13 and s14."""
        methods = (
            '_check_schema',
            '_compute_gross_areas',
            '_impute_replacement_costs',
            '_derive_building_types',
        )
        manager = mock.Mock()
        with contextlib.ExitStack() as stack:
            for name in methods:
                real = getattr(hc.HazusEQInventoryCompiler, name)
                manager.attach_mock(
                    stack.enter_context(
                        mock.patch.object(
                            hc.HazusEQInventoryCompiler,
                            name,
                            autospec=True,
                            side_effect=real,
                        )
                    ),
                    name,
                )
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=2, projected_crs=METRIC_CRS
            ).compile(study_area)
        # Each call's arguments are the compiler, then the world.
        assert [(args[1], name) for name, args, _ in manager.mock_calls] == [
            (world, name) for world in range(2) for name in methods
        ]

    def test_gross_area(self, compiled: object) -> None:
        """The area of s12 is the footprint's in each world and the output's."""
        result = compiled(n_worlds=3)
        view = result.freeze.read(hc.STAGE_S08_VIEW)
        areas = result.freeze.read(hc.STAGE_S12_GROSS_AREA)
        log = result.logs['gross_area'].set_index(['asset_id', 'world'])
        assert len(log) == 3 * len(view.inventory)
        # A building the schema check excludes in a world keeps its row.
        refused = result.report.exclusions.query("stage == 's11_schema_check'")
        assert len(refused)
        assert all(key in log.index for key in zip(refused.asset_id, refused.world))
        for world in range(3):
            realization = view.get_world_realization(world)
            stage = features_in(areas, world)
            output = features_in(result.inventory, world)
            for asset_id, asset in realization.inventory.items():
                area, area_source = footprint_gross_area(asset_id, asset)
                assert stage[asset_id][GROSS_AREA_KEY] == area
                assert log.loc[(asset_id, world), 'area_source'] == area_source
                if not output[asset_id][EXCLUDED_KEY]:
                    assert output[asset_id][GROSS_AREA_KEY] == area
        assert any(
            isinstance(asset.features[GROSS_AREA_KEY], list)
            for asset in areas.inventory.values()
        )

    def test_value_inputs(self, compiled: object) -> None:
        """Step s13 reads ``STAGE_S08_VIEW``, the donors and the areas of s12."""
        result = compiled(n_worlds=3)
        log = result.logs['value']
        from_footprints = ReplacementCostImputer(
            seed=SEED,
            **dict(DEFAULT_IMPUTER_OPTIONS),
            gross_area_key=NSI_GROSS_AREA_KEY,
        ).impute_stage(result.freeze, hc.STAGE_S08_VIEW, hc.STAGE_S05_DONORS)
        columns = [c for c in log.columns if c != 'area_source']
        pd.testing.assert_frame_equal(log[columns], from_footprints.log[columns])
        has_area = log['footprint_gross_area_sqft'].notna()
        assert set(log.loc[has_area, 'area_source']) == {AREA_FROM_FEATURE}
        assert set(log.loc[~has_area, 'outcome']) == {OUTCOME_LOCAL_VALUE}
        # The value of a building the schema refuses is computed and kept in
        # the stage, and the view still holds what the rates read.
        costs = result.freeze.read(hc.STAGE_S13_REPLACEMENT_COST)
        assert costs.inventory['u08'].features[REPLACEMENT_COST_KEY] is not None
        view = features_in(result.freeze.read(hc.STAGE_S08_VIEW), 0)
        assert view['l01'][GROSS_AREA_KEY] == 777.0
        assert view['h01'][REPLACEMENT_COST_KEY] == 350000.0
        assert view['h01'][NSI_GROSS_AREA_KEY] == 1400.0
        assert view['c01'][BUILDING_TYPE_KEY] == 'C'

    def test_excluded_buildings(self, compiled: object) -> None:
        """An excluded building keeps its local values and letter, and no computed value or area."""
        result = compiled(n_worlds=3)
        rows = result.report.exclusions
        assert set(zip(rows['asset_id'], rows['world'])) >= {
            (asset_id, world)
            for asset_id in ('f01', 'l02', 'u03', 'u08', 'u10')
            for world in range(3)
        }
        for asset_id, world in set(zip(rows['asset_id'], rows['world'])):
            features = features_in(result.inventory, world).get(asset_id)
            if features is None:
                continue
            assert features[EXCLUDED_KEY] is True
            assert GROSS_AREA_KEY not in features
            local = LOCAL.get(asset_id, {})
            assert features.get(REPLACEMENT_COST_KEY) == local.get(
                REPLACEMENT_COST_KEY
            )
            assert features.get(LOCAL_GROSS_AREA_KEY) == local.get(GROSS_AREA_KEY)
            letter = local.get(
                BUILDING_TYPE_KEY,
                hc._LETTER_OF_TYPE.get(features.get(STRUCTURE_TYPE_KEY)),
            )
            assert features.get(BUILDING_TYPE_KEY) == letter
        l02 = features_in(result.inventory, 0)['l02']
        assert (l02[BUILDING_TYPE_KEY], l02[NSI_GROSS_AREA_KEY]) == ('M', 7000.0)

    def test_schema_ahead_of_the_value(self, study_area: object) -> None:
        """A building that fails the schema and has no value is filed by the schema check."""
        # u10 has no value in any world; a local URM of 1980 fails the schema.
        local = build_inputs().local
        x, y, side = FOOTPRINTS['u10'][:3]
        local.add_asset(
            'u10',
            Asset(
                'u10',
                square(x, y, side),
                {STRUCTURE_TYPE_KEY: 'URM', YEAR_BUILT_KEY: 1980},
            ),
        )
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        ).compile(study_area._replace(local=local))
        log = result.logs['value']
        assert not set(log.loc[log['asset_id'] == 'u10', 'outcome']) & set(
            OUTCOME_VALUE_CODES
        )
        rows = result.report.exclusions
        rows = rows[rows['asset_id'] == 'u10']
        assert sorted(rows['world']) == [0, 1, 2]
        assert set(zip(rows['stage'], rows['category'])) == {
            ('s11_schema_check', 'fails_schema')
        }
        schema_check = result.report.summary()['schema_check']['excluded_by_world']
        assert {int(w): n for w, n in schema_check.items()} == {0: 3, 1: 3, 2: 3}

    def test_areas_of_other_sources(self, compiled: object) -> None:
        """The output keeps the local and NSI gross areas beside the computed one."""
        result = compiled(n_worlds=3)
        for world in range(3):
            features = features_in(result.inventory, world)
            l01 = features['l01']
            assert math.isclose(
                l01[GROSS_AREA_KEY], l01[PLAN_AREA_KEY] * l01[STORIES_KEY]
            )
            assert (l01[LOCAL_GROSS_AREA_KEY], l01[NSI_GROSS_AREA_KEY]) == (
                777.0,
                1400.0,
            )
        read = result.freeze.read
        assert {
            asset_id: asset.features
            for asset_id, asset in read(
                hc.STAGE_S12_WITHHELD_LOCAL
            ).inventory.items()
        } == {'l01': {GROSS_AREA_KEY: 'local'}}
        joined = read(hc.STAGE_S12_WITHHELD_JOIN).inventory
        holders = {
            asset_id
            for stage in (hc.STAGE_S02_NSI_JOIN, hc.STAGE_S02_DERIVED)
            for asset_id, asset in read(stage).inventory.items()
            if GROSS_AREA_KEY in asset.features
        }
        assert set(joined) == holders
        assert joined['l02'].features == {GROSS_AREA_KEY: 'derived'}
        lettered = read(hc.STAGE_S14_WITHHELD).inventory
        assert set(lettered) == {
            asset_id
            for asset_id, asset in read(hc.STAGE_S02_NSI_JOIN).inventory.items()
            if BUILDING_TYPE_KEY in asset.features
        }

    @pytest.mark.parametrize('carried', [False, True])
    def test_the_merge_holds_no_withheld_value(
        self,
        carried: bool,  # noqa: FBT001
    ) -> None:
        """The world's merged inventory holds no value that s12 to s14 replace."""
        inputs = build_inputs()
        if carried:
            # Footprints that carry an area and a letter of their own.
            for asset_id, values in FOOTPRINT_VALUES.items():
                inputs.footprints.inventory[asset_id].features.update(values)
        real = hc.HazusEQInventoryCompiler._check_schema
        merges = []

        def spied(
            compiler: object,
            world: int,
            merged: AssetInventory,
            excluded: set,
            report: object,
        ) -> object:
            """
            Keep a copy of each world's merged inventory, then check its schema.

            Args:
                compiler (object):
                    The compiler whose schema check is wrapped.
                world (int):
                    The world the merged inventory holds.
                merged (AssetInventory):
                    The world's merged inventory.
                excluded (set):
                    The buildings excluded from the world.
                report (object):
                    The run's report.

            Returns:
                object:
                    The schema check's violations.
            """
            merges.append({a: dict(x.features) for a, x in merged.inventory.items()})
            return real(compiler, world, merged, excluded, report)

        with mock.patch.object(
            hc.HazusEQInventoryCompiler,
            '_check_schema',
            autospec=True,
            side_effect=spied,
        ):
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
            ).compile(inputs)
        assert len(merges) == 3
        for merged in merges:
            for asset_id, features in merged.items():
                local = LOCAL.get(asset_id, {})
                assert GROSS_AREA_KEY not in features, asset_id
                for key in (BUILDING_TYPE_KEY, REPLACEMENT_COST_KEY):
                    assert (key in features) == (key in local), (asset_id, key)

    def test_carried_footprint_values_leave_excluded_buildings(self) -> None:
        """An area or a letter the footprints carry is not written on an excluded building."""
        inputs = build_inputs()
        for asset_id, values in FOOTPRINT_VALUES.items():
            inputs.footprints.inventory[asset_id].features.update(values)
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        ).compile(inputs)
        for world in range(3):
            features = features_in(result.inventory, world)
            for asset_id in ('f01', 'u08'):
                assert features[asset_id][EXCLUDED_KEY] is True
                assert GROSS_AREA_KEY not in features[asset_id]
                assert features[asset_id].get(
                    BUILDING_TYPE_KEY
                ) == hc._LETTER_OF_TYPE.get(
                    features[asset_id].get(STRUCTURE_TYPE_KEY)
                )

    def test_the_value_guard_files_a_building_without_a_class(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A building without a class that s11 lets through is missing_input."""
        local = build_inputs().local
        x, y, side = FOOTPRINTS['f01'][:3]
        local.add_asset(
            'f01',
            Asset(
                'f01',
                square(x, y, side),
                {YEAR_BUILT_KEY: 1970, STRUCTURE_TYPE_KEY: 'W1', STORIES_KEY: 1},
            ),
        )
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        )
        real = compiler._validator.validate
        # The schema check refuses nothing, as if the schema did not require
        # an occupancy class.
        monkeypatch.setattr(
            compiler._validator,
            'validate',
            lambda _inventory: real(AssetInventory()),
        )
        result = compiler.compile(study_area._replace(local=local))
        rows = result.report.exclusions
        rows = rows[rows['asset_id'] == 'f01']
        assert list(zip(rows['stage'], rows['category'])) == [
            ('s13_replacement_cost', 'missing_input')
        ]
        assert 'no_class' in rows['detail'].iloc[0]
