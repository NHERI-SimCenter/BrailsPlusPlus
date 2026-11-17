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
import math
from typing import TYPE_CHECKING, Callable
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import brails.compilers
from brails.compilers import hazus_eq_inventory_compiler as hc
from brails.compilers.hazus_eq_inventory_compiler import (
    CompilationResult,
    CompilerInputs,
    HazusEQInventoryCompiler,
)
from brails.constants import (
    EXCLUDED_KEY,
    FOOTPRINT_DATASETS_KEY,
    FOOTPRINT_PROVIDER_KEY,
    FOOTPRINT_RELEASE_KEY,
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
    """The output step marks the exclusions of each world."""

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


class TestRandomness:
    """The run depends on the seed alone."""

    def test_repeated_and_reordered(self) -> None:
        """One seed gives one inventory, in any input order."""
        collections = []
        for inputs in (build_inputs(), build_inputs(), build_inputs(reverse=True)):
            result = HazusEQInventoryCompiler(
                seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
            ).compile(inputs)
            collection = result.inventory.get_geojson()
            collection.pop('generated')
            collections.append(collection)
        first, second, reversed_order = collections
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
