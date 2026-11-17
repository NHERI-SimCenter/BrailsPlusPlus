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
    DEFAULT_PROPERTIES,
    CompilationResult,
    CompilerInputs,
    HazusEQInventoryCompiler,
)
from brails.constants import (
    EXCLUDED_KEY,
    FOOTPRINT_DATASETS_KEY,
    FOOTPRINT_PROVENANCE_FEATURES,
    FOOTPRINT_PROVIDER_KEY,
    FOOTPRINT_RELEASE_KEY,
    GROSS_AREA_KEY,
    LOCAL_GROSS_AREA_KEY,
    NSI_GROSS_AREA_KEY,
    PLAN_AREA_KEY,
    SOURCE_STRING_KEY,
    STORIES_KEY,
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
from tests.compilers.compiler_fixtures import (
    FOOTPRINTS,
    LOCAL,
    METRIC_CRS,
    RELEASE,
    SEED,
    build_inputs,
)

if TYPE_CHECKING:
    from brails.compilers.compilation_report import CompilationReport

FREEZE_ORDER = (
    'footprints',
    'local',
    'join_nsi',
    'join_derived',
    'output',
)
# The timed units of a run, once and per world.
TIMED_ONCE = (
    'footprints',
    'join',
    'output_freeze',
    'manifest',
)
TIMED_PER_WORLD = ('output',)
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


class TestConstructor:
    """The constructor checks its settings."""

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
        ],
    )
    def test_refusals(self, arguments: dict, error: type, match: str | None) -> None:
        """A setting no component accepts is refused before any run."""
        with pytest.raises(error, match=match):
            HazusEQInventoryCompiler(**arguments)

    def test_seed(self) -> None:
        """Without a seed the compiler draws one and records it."""
        assert isinstance(HazusEQInventoryCompiler().seed, int)


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
        """One seed gives one file, in any input order."""
        files = []
        for name, inputs, seed in (
            ('first', build_inputs(), SEED),
            ('second', build_inputs(), SEED),
            ('reversed', build_inputs(reverse=True), SEED),
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
        first, second, reversed_order = files
        assert first == second == reversed_order


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
