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

import functools
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import brails.compilers
from brails.compilers import hazus_eq_inventory_compiler as hc
from brails.compilers.hazus_eq_inventory_compiler import (
    ALL_PROPERTIES,
    DEFAULT_IMPUTER_OPTIONS,
    DEFAULT_PROPERTIES,
    CompilationResult,
    CompilerInputs,
    HazusEQInventoryCompiler,
)
from brails.constants import (
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
    LOCAL_GROSS_AREA_KEY,
    NSI_GROSS_AREA_KEY,
    OCCUPANCY_KEY,
    PLAN_AREA_KEY,
    SOURCE_STRING_KEY,
    STORIES_FROM_HEIGHT_KEY,
    STORIES_KEY,
    STRUCTURE_TYPE_KEY,
    UNITS_KEY,
    YEAR_BUILT_KEY,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_height_classes_by_type,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    OVERTURE_PROVIDER,
)
from brails.types.asset_inventory import (
    HAZUS_EQ_SOURCE_FEATURES,
    Asset,
    AssetInventory,
)
from brails.utils import Importer
from brails.utils.unit_count_rules import UNIT_COUNT_CLASSES
from tests.compilers.compiler_fixtures import (
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
# The footprints no NSI record reaches.
UNREACHED = set(FOOTPRINTS) - {record[0] for record in NSI_RECORDS.values()}
FREEZE_ORDER = (
    'footprints',
    'local',
    'join_nsi',
    'join_derived',
    'step4_derived',
    'step4_dropped',
    'step4_withheld',
    'step4_imputed',
    'step4',
    'step6_withheld',
    'step6_imputed',
    'step6_drawn',
    'step6_derived',
    'step6',
    'output',
)
# The timed units of a run, once and per world.
TIMED_ONCE = (
    'footprints',
    'join',
    'step4',
    'step4_freeze',
    'step6_freeze',
    'output_freeze',
    'manifest',
)
TIMED_PER_WORLD = ('step4_view', 'step6', 'step6_view', 'output')
SQUARE_FEET_PER_SQUARE_METER = 10.7639


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
            The study area.

    Returns:
        Callable[..., CompilationResult]:
            ``compiled(n_worlds=3, worlds=None)`` gives the result of the
            compiler at the fixture seed and CRS.
    """

    @functools.cache
    def run(n_worlds: int = 3, worlds: tuple[int, ...] | None = None) -> object:
        """
        Compile the study area with the given settings.

        Args:
            n_worlds (int, optional):
                The number of worlds.
            worlds (tuple[int, ...] | None, optional):
                The worlds written.

        Returns:
            CompilationResult:
                The result, shared between calls with the same settings.
        """
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=n_worlds, projected_crs=METRIC_CRS
        )
        return compiler.compile(study_area, worlds=worlds)

    return run


def written(path: object) -> dict[str, dict]:
    """
    Read a written GeoJSON's properties by feature id.

    Args:
        path (object):
            The file.

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
    run holds ``None`` where a shorter run holds no key. The merges the run
    builds hold no ``None`` (``test_merge_keys_do_not_depend_on_n_worlds``),
    and the comparison of realizations leaves ``None`` out.

    Args:
        inventory (AssetInventory):
            An inventory of several worlds.
        world (int):
            The world.

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
            The records or the exclusions of a report.
        world (int):
            The world.

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
    """`compile` checks its worlds and inputs and leaves the inputs alone."""

    @pytest.mark.parametrize(
        ('worlds', 'error'),
        [
            ((), ValueError),
            ((1, 1), ValueError),
            ((-1,), ValueError),
            ((1.5,), TypeError),
            ((True,), TypeError),
        ],
    )
    def test_worlds(self, study_area: object, worlds: tuple, error: type) -> None:
        """Worlds are distinct non-negative integers, at least one."""
        with pytest.raises(error):
            HazusEQInventoryCompiler(seed=SEED).compile(study_area, worlds=worlds)

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

    @pytest.mark.parametrize('key', hc.RESERVED_LOCAL_FEATURES)
    def test_reserved_local_feature_refused(self, key: str) -> None:
        """A local inventory holding a feature the compiler reads back is refused."""
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
    """Steps 1 to 3: the spine, the footprint features and the join's parts."""

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
                side**2 * SQUARE_FEET_PER_SQUARE_METER,
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

    def test_join_parts(self, compiled: object) -> None:
        """The spine holds every footprint, and the other parts centroids."""
        result = compiled(n_worlds=3)
        read = result.freeze.read
        assert result.freeze.order == FREEZE_ORDER
        assert [e['stage'] for e in result.manifest['stages']] == list(FREEZE_ORDER)
        assert set(read('footprints').inventory) == set(FOOTPRINTS)
        assert all(
            len(asset.coordinates) > 1
            for asset in read('footprints').inventory.values()
        )
        assert set(read('local').inventory) == set(LOCAL)
        for stage in ('local', 'join_nsi', 'join_derived'):
            for asset in read(stage).inventory.values():
                assert len(asset.coordinates) == 1, stage
        assert read('output') is result.inventory

    def test_changed_part_raises_at_the_manifest(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A component that changes a frozen part it read stops the run."""
        real = hc.compose_source

        def changing(
            freeze: object, parts: list, **kwargs: object
        ) -> AssetInventory:
            """
            Compose a source, then change a record it shares with a part.

            Args:
                freeze (object):
                    The freeze.
                parts (list):
                    The parts composed.
                **kwargs (object):
                    The other arguments.

            Returns:
                AssetInventory:
                    The composition.
            """
            composed = real(freeze, parts, **kwargs)
            if 'join_nsi' in parts:
                composed.inventory['h01'].features[STORIES_KEY] = 9
            return composed

        monkeypatch.setattr(hc, 'compose_source', changing)
        with pytest.raises(RuntimeError, match='join_nsi'):
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
            ).compile(study_area)


class TestOutput:
    """The output step and what `write` makes of it."""

    def test_output_marks_exclusions(self, study_area: object) -> None:
        """An exclusion recorded before the output step marks its world."""
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
                    stage='resolution',
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
            marks = result.inventory.inventory[asset_id].features[EXCLUDED_KEY]
            assert marks == [False, True, False]

    def test_excluded_assets(self, study_area: object, tmp_path: object) -> None:
        """Excluded in every written world: left out; in some: on request."""
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        ).compile(study_area)
        report = result.report
        assert result.inventory.inventory['h01'].features[EXCLUDED_KEY] is False
        report.record_exclusion(
            'h01', world=1, stage='year', category='missing_driver', detail='x'
        )
        for world in range(3):
            report.record_exclusion(
                'h02',
                world=world,
                stage='resolution',
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

        result._replace(worlds=(0,)).write(tmp_path / 'world0.geojson')
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
        """A write of every world assembles nothing new, a subset is assembled."""
        result = compiled(n_worlds=3)
        assembler = mock.Mock(wraps=hc.WorldAssembler)
        monkeypatch.setattr(hc, 'WorldAssembler', assembler)
        result.write(tmp_path / 'all.geojson', keep_partly_excluded=True)
        result.write(tmp_path / 'default.geojson')
        assert not assembler.called
        assert result.freeze.manifest() == result.manifest
        result._replace(worlds=(0, 1)).write(tmp_path / 'subset.geojson')
        assert assembler.called

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
        assert GROSS_AREA_KEY not in inventory['h01'].features
        assert inventory['l01'].features[LOCAL_GROSS_AREA_KEY] == 777.0
        assert inventory['l01'].features[NSI_GROSS_AREA_KEY] == 1400.0
        assert GROSS_AREA_KEY not in inventory['l01'].features

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


class TestStep4:
    """Step 4: story counts, the unit budget and the donor imputer."""

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
        stories = records[records['stage'] == 'stories']
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
        withheld = result.freeze.read('step4_withheld').inventory
        assert withheld['p01'].features == {STORIES_KEY: 'nsi'}
        assert 'h02' not in withheld
        assert 'l02' not in withheld

    def test_imputed_targets(self, compiled: object) -> None:
        """Reached footprints without a story count are imputed one, not units."""
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

    def test_local_class_is_reached(self, study_area: object) -> None:
        """A footprint with a local occupancy class is reached and a story donor."""
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
        # l01 is a local RES1 over NSI's COM1 record.
        assert 'l01' in allocated.call_args.args[2].inventory
        assert 'l01' not in allocated.call_args.args[1].inventory
        assert 'l01' in imputed.call_args.args[2].inventory

    def test_validation(self, study_area: object) -> None:
        """The imputer's copies are checked for implausible pairs."""
        assert not hc._plausible('a', 0, {OCCUPANCY_KEY: 'RES2', STORIES_KEY: 4})
        assert hc._plausible('a', 0, {OCCUPANCY_KEY: 'RES2', STORIES_KEY: 1})
        assert hc._plausible('a', 0, {OCCUPANCY_KEY: 'RES2'})
        real = hc.ObservedDonorImputer.impute_stage
        with mock.patch.object(
            hc.ObservedDonorImputer, 'impute_stage', autospec=True, side_effect=real
        ) as impute:
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
            ).compile(study_area)
        assert impute.call_args.kwargs['validate'] is hc._plausible

    def test_dropped(self, compiled: object) -> None:
        """A small unreached footprint far from the park is dropped in every world."""
        result = compiled(n_worlds=3)
        assert 'd01' not in result.inventory.inventory
        rows = result.report.exclusions
        rows = rows[rows['asset_id'] == 'd01']
        assert sorted(rows['world']) == [0, 1, 2]
        assert set(rows['category']) == {'dropped'}
        assert set(rows['stage']) == {'units'}

    def test_local_values_unchanged(self, compiled: object) -> None:
        """No local value is withheld or replaced, the mobile-home rule's included."""
        result = compiled(n_worlds=3)
        assert math.dist(MH_PARK, FOOTPRINTS['l01'][:2]) < 300.0
        records = result.report.records
        assert 'r05' in set(records.loc[records['stage'] == 'units', 'asset_id'])
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

    def test_step4_parts(self, compiled: object) -> None:
        """Each part and record of step 4 holds only what its kind says."""
        result = compiled(n_worlds=3)
        read = result.freeze.read
        kept = set(result.inventory.inventory)
        assert set(read('step4_dropped').inventory) == {'d01'}
        assert all(
            a.features == {DROPPED_KEY: True}
            for a in read('step4_dropped').inventory.values()
        )
        labels = {
            value
            for asset in read('step4_withheld').inventory.values()
            for value in asset.features.values()
        }
        assert labels == {'nsi', 'derived'}
        # The mobile-home rule's relabel of an occupancy class the join derived.
        assert read('step4_withheld').inventory['r05'].features == {
            OCCUPANCY_KEY: 'derived'
        }
        assert {
            key
            for a in read('step4_derived').inventory.values()
            for key in a.features
        } == {STORIES_FROM_HEIGHT_KEY, STORIES_KEY}
        imputed = read('step4_imputed')
        assert imputed.n_pw == 3
        assert set(imputed.inventory) == kept
        assert {k for a in imputed.inventory.values() for k in a.features} <= {
            OCCUPANCY_KEY,
            UNITS_KEY,
            STORIES_KEY,
        }
        for stage in (
            'step4_derived',
            'step4_dropped',
            'step4_withheld',
            'step4_imputed',
        ):
            for asset in read(stage).inventory.values():
                assert len(asset.coordinates) == 1, stage
        assert set(read('step4').inventory) == kept


class TestWorlds:
    """A world does not depend on the number of worlds compiled."""

    def test_merge_keys_do_not_depend_on_n_worlds(self, study_area: object) -> None:
        """World 0's last merge holds the same features at one world and seven."""
        real = hc._WorldSources.merge
        keys = {}
        for n_worlds in (1, 7):
            merges: list = []
            with mock.patch.object(
                hc._WorldSources,
                'merge',
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

    @pytest.mark.parametrize('worlds', [(3,), (2, 5)])
    def test_subsets(
        self, compiled: object, tmp_path: object, worlds: tuple
    ) -> None:
        """A run asked for some worlds writes them as the full run computes them."""
        subset, full = compiled(n_worlds=10, worlds=worlds), compiled(n_worlds=10)
        assert subset.worlds == worlds
        assert subset.inventory.n_pw == max(worlds) + 1
        subset.write(
            tmp_path / 'subset.geojson',
            keep_partly_excluded=True,
            extra_properties=ALL_PROPERTIES,
        )
        properties = written(tmp_path / 'subset.geojson')
        compiled_in = {
            asset_id
            for world in worlds
            for asset_id, values in features_in(full.inventory, world).items()
            if not values[EXCLUDED_KEY]
        }
        assert set(properties) == compiled_in
        for position, world in enumerate(worlds):
            expected = features_in(full.inventory, world)
            for asset_id, values in properties.items():
                for key, value in expected[asset_id].items():
                    held = values.get(key)
                    if isinstance(held, list):
                        held = held[position]
                    assert held == value or (
                        isinstance(value, float) and math.isclose(held, value)
                    ), (asset_id, key)
        records = pd.read_parquet(tmp_path / 'subset_report.parquet')
        exclusions = pd.read_parquet(tmp_path / 'subset_report_exclusions.parquet')
        assert set(records['world']) == set(worlds)
        assert set(exclusions['world']) == set(worlds)
        summary = json.loads(
            (tmp_path / 'subset_report_summary.json').read_text(encoding='utf-8')
        )
        assert summary['worlds']['written'] == list(worlds)
        imputer_log = pd.read_parquet(tmp_path / 'subset_imputer.parquet')
        assert set(imputer_log['world']) == set(worlds)

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
    """Steps 5 and 6: years, design levels, pairs, types and height classes."""

    def test_components_receive_worlds(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each per-world component is told the number of the world it computes."""
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=10, projected_crs=METRIC_CRS
        )
        spies = {}
        for name in (
            'ACSYearBuiltSampler',
            'HazusEQStructureTypeInferer',
            'PairRedraw',
        ):
            spies[name] = mock.Mock(wraps=getattr(hc, name))
            monkeypatch.setattr(hc, name, spies[name])
        compiler.compile(study_area, worlds=(2, 5))
        for spy in spies.values():
            worlds = Counter(c.kwargs['world'] for c in spy.call_args_list)
            assert set(worlds) == set(range(6))
            assert len(set(worlds.values())) == 1

    def test_frozen_inputs_cross_the_boundary(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The year fit and the re-draw's donors read realization w of step4."""
        # The spies keep the inventories they were given, so the comparison
        # after the run also sees a later change to them.
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
        step4 = result.freeze.read('step4')
        occupancy = HAZUS_EQ_SOURCE_FEATURES.index(OCCUPANCY_KEY)
        for world in range(3):
            frozen = features_in(step4, world)
            assert features_in(fit.call_args_list[world].args[1], 0) == frozen
            reached = {
                a: values
                for a, values in frozen.items()
                if values[SOURCE_STRING_KEY][occupancy] in 'LNR'
                and STORIES_KEY in values
            }
            donors = redraw.call_args_list[world].args[1]
            assert features_in(donors, 0) == reached

    def test_changed_view_raises_at_the_manifest(
        self, study_area: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A component that changes the frozen step4 view it read stops the run."""
        real = hc.StageFreeze.read

        def changing(self: object, stage: str) -> AssetInventory:
            """
            Return a frozen stage, changing the step4 view.

            Args:
                self (object):
                    The freeze.
                stage (str):
                    The stage read.

            Returns:
                AssetInventory:
                    The stage.
            """
            held = real(self, stage)
            if stage == 'step4':
                held.inventory['h01'].features[STORIES_KEY] = 9
            return held

        monkeypatch.setattr(hc.StageFreeze, 'read', changing)
        with pytest.raises(RuntimeError, match='step4'):
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
            ).compile(study_area)

    def test_step6_parts(self, compiled: object) -> None:
        """Each per-world part of steps 5 and 6 holds the spine after the drops."""
        result = compiled(n_worlds=3)
        kept = set(result.inventory.inventory)
        for stage in (
            'step6_withheld',
            'step6_imputed',
            'step6_drawn',
            'step6_derived',
        ):
            part = result.freeze.read(stage)
            assert part.n_pw == 3
            assert set(part.inventory) == kept
            for asset in part.inventory.values():
                assert len(asset.coordinates) == 1, stage
        drawn = {
            key
            for asset in result.freeze.read('step6_drawn').inventory.values()
            for key in asset.features
        }
        assert drawn == {YEAR_BUILT_KEY, STRUCTURE_TYPE_KEY}

    def test_loop_in_the_run(self, compiled: object) -> None:
        """An NSI type its cell rejects is withheld and the asset compiled."""
        result = compiled(n_worlds=3)
        records = result.report.records
        rows = records[
            (records['asset_id'] == 'p01') & (records['stage'] == 'resolution')
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
        """A local letter is kept, and every other follows the structure type."""
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
        # q03's local W stands beside its local C1; NSI's C is not written,
        # and a notice counts the disagreement in each world.
        assert local['q03'] != hc._LETTER_OF_TYPE[LOCAL['q03'][STRUCTURE_TYPE_KEY]]
        assert derived & UNREACHED
        notices = result.report.summary()['notices']
        for world in range(3):
            assert (
                f'World {world}: 1 compiled building(s) keep a local '
                f'{BUILDING_TYPE_KEY} other than the letter of their '
                f'{STRUCTURE_TYPE_KEY}.'
            ) in notices

    def test_letters_disagreeing_are_counted(self, study_area: object) -> None:
        """The notice counts compiled buildings whose local letter is not the type's."""
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
                'u06', world=1, stage='resolution', category='conflict', detail='x'
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
        notices = result.report.summary()['notices']
        for world, count in ((0, 2), (1, 1), (2, 2)):
            assert (
                f'World {world}: {count} compiled building(s) keep a local '
                f'{BUILDING_TYPE_KEY} other than the letter of their '
                f'{STRUCTURE_TYPE_KEY}.'
            ) in notices

    def test_pairing_covers_every_type(self) -> None:
        """Every Hazus structure type has a general type letter."""
        assert set(hc._LETTER_OF_TYPE) == set(get_hazus_height_classes_by_type())

    def test_completed_tables(self, compiled: object) -> None:
        """The NSI EDU1 at five stories, an empty printed cell, compiles."""
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
            (records['asset_id'] == 'c03') & (records['stage'] == 'pair_redraw')
        ]
        assert set(rows['action']) == {'withheld', 'redrawn'}
        assert set(rows.loc[rows['action'] == 'withheld', 'source']) == {'nsi'}
        record = result.freeze.read('step6_withheld').get_world_realization(0)
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
        rows = records[records['stage'] == 'carve_out']
        assert not rows.empty
        for row in rows.itertuples():
            features = features_in(result.inventory, row.world)[row.asset_id]
            assert features[STRUCTURE_TYPE_KEY] == 'W1'
            assert features[DESIGN_LEVEL_KEY] == 'Moderate-Code'
            assert (row.old, row.new) == ('Pre-Code', 'Moderate-Code')
        oregon = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, state='Oregon', projected_crs=METRIC_CRS
        ).compile(study_area)
        assert 'carve_out' not in set(oregon.report.records['stage'])
        levels = [
            (features[STRUCTURE_TYPE_KEY], features[DESIGN_LEVEL_KEY])
            for world in range(3)
            for features in features_in(oregon.inventory, world).values()
            if STRUCTURE_TYPE_KEY in features and DESIGN_LEVEL_KEY in features
        ]
        assert ('W1', 'Pre-Code') in levels

    def test_certificate(self, compiled: object) -> None:
        """A local URM of 1990, which gives High-Code, fails the schema everywhere."""
        result = compiled(n_worlds=3)
        rows = result.report.exclusions
        rows = rows[rows['asset_id'] == 'l02']
        assert sorted(rows['world']) == [0, 1, 2]
        assert set(rows['category']) == {'certificate'}
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
        """A local type the schema accepts compiles with its derived height class."""
        result = compiled(n_worlds=3)
        height = HAZUS_EQ_SOURCE_FEATURES.index(HEIGHT_CLASS_KEY)
        kind = HAZUS_EQ_SOURCE_FEATURES.index(STRUCTURE_TYPE_KEY)
        for world in range(3):
            features = features_in(result.inventory, world)['q03']
            assert features[EXCLUDED_KEY] is False
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
        """A status drawn per world gives a list of source strings and of marks."""
        result = compiled(n_worlds=3)
        year = HAZUS_EQ_SOURCE_FEATURES.index(YEAR_BUILT_KEY)
        exclusions = result.report.exclusions
        mixed = [
            asset_id
            for asset_id, asset in result.inventory.inventory.items()
            if isinstance(asset.features[SOURCE_STRING_KEY], list)
            and {codes[year] for codes in asset.features[SOURCE_STRING_KEY]}
            == {'D', '-'}
        ]
        assert mixed
        for asset_id in mixed:
            features = result.inventory.inventory[asset_id].features
            listed = set(exclusions.loc[exclusions['asset_id'] == asset_id, 'world'])
            assert features[EXCLUDED_KEY] == [w in listed for w in range(3)]
            for world, codes in enumerate(features[SOURCE_STRING_KEY]):
                assert (codes[year] == '-') == (world in listed)

    def test_undated_excluded(self, compiled: object) -> None:
        """A non-residential building without a year is excluded at its stage."""
        result = compiled(n_worlds=3)
        rows = result.report.exclusions.set_index(['asset_id', 'world'])
        for asset_id in ('c01', 'c04', 'c05'):
            for world in range(3):
                assert rows.loc[(asset_id, world), 'stage'] == 'nonresidential_year'

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
        rows = rows[rows['stage'] == 'year']
        assert set(rows['category']) == {'missing_driver'}
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
            ('q02', 'year'),
            ('h05', 'design_level'),
            ('h07', 'height_class'),
        ):
            for world in range(2):
                row = rows.loc[(asset_id, world)]
                assert (row['stage'], row['category'], row['diagnosis']) == (
                    stage,
                    'missing_driver',
                    'invalid',
                )

    def test_year_fit_log(self, compiled: object, tmp_path: object) -> None:
        """The written year fit log names the block group of each row."""
        compiled(n_worlds=3).write(tmp_path / 'run.geojson')
        fit = pd.read_parquet(tmp_path / 'run_year_fit.parquet')
        assert {BLOCK_GROUP_KEY, 'world'} <= set(fit.columns)
