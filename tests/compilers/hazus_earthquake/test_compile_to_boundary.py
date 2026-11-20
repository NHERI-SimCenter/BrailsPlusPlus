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


"""Tests of the Hazus earthquake inventory compiler around a boundary."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pandas as pd
import pytest
import shapely
from shapely.geometry import shape

from brails.compilers.hazus_earthquake.hazus_eq_inventory_compiler import (
    CompilationResult,
    CompilerInputs,
    HazusEQInventoryCompiler,
)
from brails.constants import YEAR_BUILT_KEY
from brails.types.asset_inventory import AssetInventory
from tests.compilers.hazus_earthquake.compiler_fixtures import (
    FOOTPRINTS,
    METRIC_CRS,
    SEED,
    build_inputs,
    in_state,
    lonlat,
)

if TYPE_CHECKING:
    from pathlib import Path

# A place inside the western block group, in meters east and north of the fixture
# origin, and the width of the area compiled around it.
PLACE = ((-40, -40), (190, -40), (190, 240), (-40, 240))
BUFFER_M = 2000.0

# The keys of the summary and of its settings after a compilation without a boundary,
# in order.
SUMMARY_KEYS = [
    'seed',
    'settings',
    'worlds',
    'manifest_stages',
    'timings',
    'actions',
    'exclusions',
    'assets_excluded',
    'notices',
    'coverage',
    'schema_check',
]
SETTINGS_KEYS = [
    'study_area',
    'footprint_release',
    'height_unit',
    'n_worlds',
    'worlds',
    'state',
    'seismic_zone',
    'design_level_method',
    'join_cap_m',
    'projected_crs',
    'imputer_options',
    'budget_options',
    'sampler_options',
    'merge_stacked',
]


pytestmark = pytest.mark.usefixtures('no_network')


def place_polygon(corners: tuple[tuple[float, float], ...]) -> shapely.Polygon:
    """
    Return a polygon of the fixture area in longitude and latitude.

    Args:
        corners (tuple[tuple[float, float], ...]):
            The corners in meters east and north of the fixture origin.

    Returns:
        shapely.Polygon:
            The polygon in EPSG:4326.
    """
    return shapely.Polygon([lonlat(x, y) for x, y in corners])


@pytest.fixture(scope='module')
def study_area() -> CompilerInputs:
    """
    Return the study area with every input given, which no test may change.

    Returns:
        CompilerInputs:
            The study area's data.
    """
    return build_inputs()


@pytest.fixture(scope='module')
def around_place(study_area: CompilerInputs) -> CompilerInputs:
    """
    Return the study area cut to the place widened by the donor cap, with the place.

    Args:
        study_area (CompilerInputs):
            The study area's data, from the ``study_area`` fixture.

    Returns:
        CompilerInputs:
            The footprints within 2,000 m of the place, every footprint but the one 3
            km away, with the other inputs as given and the place as boundary.
    """
    footprints = AssetInventory()
    footprints.inventory = {
        asset_id: asset
        for asset_id, asset in study_area.footprints.inventory.items()
        if FOOTPRINTS[asset_id][0] < PLACE[1][0] + BUFFER_M
    }
    return study_area._replace(
        footprints=footprints,
        boundary=place_polygon(PLACE),
        buffer_m=BUFFER_M,
        boundary_record={'provider': 'polygon'},
    )


def compile_with(inputs: CompilerInputs) -> CompilationResult:
    """
    Compile three worlds of the inputs at the fixture seed and CRS.

    Args:
        inputs (CompilerInputs):
            The study area's data.

    Returns:
        CompilationResult:
            The result of the compilation.
    """
    compiler = HazusEQInventoryCompiler(
        seismic_zone='4', seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
    )
    return compiler.compile(inputs)


@pytest.fixture(scope='module')
def whole(study_area: CompilerInputs) -> CompilationResult:
    """
    Return the compilation of the study area without a boundary.

    Args:
        study_area (CompilerInputs):
            The study area's data, from the ``study_area`` fixture.

    Returns:
        CompilationResult:
            The result, which no test may change.
    """
    return compile_with(study_area)


@pytest.fixture(scope='module')
def place(around_place: CompilerInputs) -> CompilationResult:
    """
    Return the compilation of the area around the place, with the place as boundary.

    Args:
        around_place (CompilerInputs):
            The area's data, from the ``around_place`` fixture.

    Returns:
        CompilationResult:
            The result, which no test may change.
    """
    return compile_with(around_place)


def read_files(stem: Path) -> dict[str, object]:
    """
    Read the files a write saved under a stem, the GeoJSON without its date.

    Args:
        stem (Path):
            The path of the GeoJSON without its suffix.

    Returns:
        dict[str, object]:
            The GeoJSON's features, the summary, and the exclusion rows.
    """
    geojson = json.loads(stem.with_suffix('.geojson').read_text(encoding='utf-8'))
    return {
        'features': geojson['features'],
        'summary': json.loads(
            stem.with_name(f'{stem.name}_report_summary.json').read_text(
                encoding='utf-8'
            )
        ),
        'exclusions': pd.read_parquet(
            stem.with_name(f'{stem.name}_report_exclusions.parquet')
        ),
    }


class TestInside:
    """The result and the written files hold the buildings inside the boundary."""

    def test_inside_values_as_compiled_without_cut(
        self, whole: CompilationResult, place: CompilationResult
    ) -> None:
        """Compiled over the place widened by the cap, the place keeps its values."""
        inside = place.inventory.inventory
        assert inside
        for asset_id, asset in inside.items():
            assert asset.features == whole.inventory.inventory[asset_id].features
        for frame in ('records', 'exclusions'):
            expected, got = (
                getattr(result.report, frame) for result in (whole, place)
            )
            pd.testing.assert_frame_equal(
                got[got['asset_id'].isin(inside)].reset_index(drop=True),
                expected[expected['asset_id'].isin(inside)].reset_index(drop=True),
            )

    def test_result_holds_inside_and_area(self, place: CompilationResult) -> None:
        """The result holds the area and, as its inventory, the buildings inside."""
        area = place.area_inventory.inventory
        assert set(place.inventory.inventory) == set(area) & place.boundary_assets
        assert set(area) - place.boundary_assets
        assert all(
            place.inventory.inventory[a] is area[a]
            for a in place.inventory.inventory
        )
        assert place.inventory.n_pw == place.area_inventory.n_pw == 3
        assert place.boundary_assets == {
            a for a, (x, y, *_) in FOOTPRINTS.items() if x < PLACE[1][0] and y < 240
        }
        summary = place.report.summary()
        assert shape(summary['settings']['boundary']).equals(place_polygon(PLACE))
        assert summary['settings']['buffer_m'] == BUFFER_M
        assert summary['settings']['boundary_record'] == {'provider': 'polygon'}
        spine = len(place.freeze.read('s01_footprints').inventory)
        inside = len(place.boundary_assets)
        assert summary['boundary'] == {
            'footprints': spine,
            'inside': inside,
            'outside': spine - inside,
        }
        # l02 and u08, which fail the schema in every world, lie outside.
        refused = summary['schema_check']['excluded_by_world']
        assert refused == {0: 0, 1: 0, 2: 0}
        outside = place.report.summary(keep_outside=True)['schema_check']
        assert outside['excluded_by_world'] == {0: 2, 1: 2, 2: 2}
        exclusions = place.report.exclusions
        inside_rows = exclusions['asset_id'].isin(place.boundary_assets)
        assert not inside_rows.all()
        assert summary['exclusions']['total'] == inside_rows.sum()
        assert place.report.summary(keep_outside=True)['exclusions']['total'] == len(
            exclusions
        )

    def test_write_saves_inside(
        self, place: CompilationResult, tmp_path: Path
    ) -> None:
        """A write saves the buildings inside and their rows, keep_outside all."""
        place.write(tmp_path / 'place.geojson')
        place.write(tmp_path / 'area.geojson', keep_outside=True)
        written = read_files(tmp_path / 'place')
        area = read_files(tmp_path / 'area')

        assert written['features'] == [
            feature
            for feature in area['features']
            if feature['id'] in place.boundary_assets
        ]
        assert len(written['features']) < len(area['features'])
        pd.testing.assert_frame_equal(
            written['exclusions'],
            area['exclusions'][
                area['exclusions']['asset_id'].isin(place.boundary_assets)
            ].reset_index(drop=True),
        )
        assert not area['exclusions']['asset_id'].isin(place.boundary_assets).all()
        for files, keep_outside in ((written, False), (area, True)):
            assert files['summary'] == json.loads(
                json.dumps(
                    place.report.summary(keep_outside=keep_outside), default=str
                )
            )

    def test_missing_default_outside_does_not_block_write(
        self,
        around_place: CompilerInputs,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A default missing around the boundary refuses only the area's file."""
        assemble = HazusEQInventoryCompiler._assemble_output

        def assemble_lacking(
            compiler: HazusEQInventoryCompiler, world: int, *args: object
        ) -> list[tuple]:
            return [
                *assemble(compiler, world, *args),
                ('r01', world, YEAR_BUILT_KEY),
            ]

        monkeypatch.setattr(
            HazusEQInventoryCompiler, '_assemble_output', assemble_lacking
        )
        lacking = compile_with(around_place)
        assert 'r01' not in lacking.boundary_assets

        assert (
            '3 default value(s) that apply to compiled buildings are missing, 0 of '
            'them of buildings inside the boundary; the log missing_defaults names '
            'them, and write refuses a file that holds them.'
        ) in lacking.report.summary()['notices']
        lacking.write(tmp_path / 'place.geojson')
        assert (tmp_path / 'place.geojson').is_file()
        with pytest.raises(RuntimeError, match='lack 3 default value'):
            lacking.write(tmp_path / 'area.geojson', keep_outside=True)
        assert not (tmp_path / 'area.geojson').exists()


class TestRefusalsAndNotices:
    """A boundary that holds no footprint is refused, and a narrow area noted."""

    def test_no_footprint_inside_refused_before_s02(
        self, around_place: CompilerInputs
    ) -> None:
        """A boundary without a footprint inside stops the run right after s01."""
        empty = around_place._replace(
            boundary=place_polygon(((1000, 0), (1100, 0), (1100, 100), (1000, 100)))
        )
        with (
            mock.patch.object(
                HazusEQInventoryCompiler, '_join_nsi_to_footprints'
            ) as join,
            pytest.raises(ValueError, match='None of the 50 footprints lies inside'),
        ):
            compile_with(empty)
        join.assert_not_called()

    def test_invalid_boundary_repaired(
        self, around_place: CompilerInputs, place: CompilationResult
    ) -> None:
        """An invalid boundary keeps inside the buildings of its repair."""
        # A hole outside the shell, which the repair turns into a second polygon.
        hole = ((210, 90), (360, 90), (360, 160), (210, 160))
        invalid = shapely.Polygon(
            place_polygon(PLACE).exterior.coords,
            [place_polygon(hole).exterior.coords],
        )
        assert not invalid.is_valid
        repaired = compile_with(around_place._replace(boundary=invalid))

        in_hole = {
            a
            for a, (x, y, side, *_) in FOOTPRINTS.items()
            if 210 < x + side / 2 < 360 and 90 < y + side / 2 < 160
        }
        assert in_hole
        assert repaired.boundary_assets == place.boundary_assets | in_hole

    def test_boundary_not_a_polygon_refused(
        self, around_place: CompilerInputs
    ) -> None:
        """A boundary that is not a polygon is refused."""
        with pytest.raises(TypeError, match='Polygon or MultiPolygon, not Point'):
            compile_with(around_place._replace(boundary=shapely.Point(lonlat(0, 0))))

    @pytest.mark.parametrize(
        ('buffer_m', 'width'), [(None, 'of unknown width'), (100.0, '100 m wide')]
    )
    def test_narrow_area_noted(
        self,
        around_place: CompilerInputs,
        place: CompilationResult,
        buffer_m: float | None,
        width: str,
    ) -> None:
        """An area narrower than the donor cap, or of unknown width, is noted."""
        notice = (
            f'The area around the boundary is {width}, and the donor cap is 2000 m'
        )
        result = compile_with(around_place._replace(buffer_m=buffer_m))
        assert any(n.startswith(notice) for n in result.report.summary()['notices'])
        assert not any(
            n.startswith('The area around')
            for n in place.report.summary()['notices']
        )


def test_without_boundary_writes_as_with_keep_outside(
    whole: CompilationResult, tmp_path: Path
) -> None:
    """Without a boundary, summary keys stay and keep_outside changes no file."""
    assert whole.area_inventory is whole.inventory
    assert whole.boundary_assets is None
    summary = whole.report.summary()
    assert list(summary) == SUMMARY_KEYS
    assert list(summary['settings']) == SETTINGS_KEYS
    whole.write(tmp_path / 'default.geojson')
    whole.write(tmp_path / 'outside.geojson', keep_outside=True)
    for suffix in ('.geojson', '_report_summary.json', '_manifest.json'):
        default, outside = (
            [
                line
                for line in (tmp_path / f'{name}{suffix}')
                .read_text('utf-8')
                .split('\n')
                if '"generated"' not in line
            ]
            for name in ('default', 'outside')
        )
        assert default == outside
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / 'default_report_exclusions.parquet'),
        pd.read_parquet(tmp_path / 'outside_report_exclusions.parquet'),
    )


@pytest.mark.parametrize('zone', ['0', '1', '2A', '2B', '3', '4', '9'])
@pytest.mark.parametrize('state', [None, 'California', 'Oregon', 'Texas'])
def test_state_and_zone_accepted_where_their_table_holds_them(
    study_area: CompilerInputs, state: str | None, zone: str
) -> None:
    """A state and zone are accepted where their design level table holds them."""
    # California's design level table holds zones 3 and 4, that of other states zones
    # 0 to 4; Texas draws structure types from tables the compiler refuses. A
    # compiler without a state takes California from the block groups.
    accepted = {
        None: {'3', '4'},
        'California': {'3', '4'},
        'Oregon': {'0', '1', '2A', '2B', '3', '4'},
        'Texas': set(),
    }[state]
    arguments = {} if state is None else {'state': state}
    area = in_state(study_area, '41') if state == 'Oregon' else study_area
    if zone not in accepted:
        with pytest.raises(ValueError, match=r'zone|structure types'):
            HazusEQInventoryCompiler(
                seed=SEED, n_worlds=1, seismic_zone=zone, **arguments
            ).compile(area)
        return
    result = HazusEQInventoryCompiler(
        seed=SEED,
        n_worlds=1,
        projected_crs=METRIC_CRS,
        seismic_zone=zone,
        **arguments,
    ).compile(area)
    settings = result.report.summary()['settings']
    assert (settings['state'], settings['seismic_zone']) == (
        state or 'California',
        zone,
    )


@pytest.mark.parametrize('merge_stacked', [False, True])
def test_merge_stacked_of_compiler_applies(
    study_area: CompilerInputs,
    merge_stacked: bool,  # noqa: FBT001
) -> None:
    """merge_stacked given to the compiler applies whatever the inputs say."""
    compiler = HazusEQInventoryCompiler(
        seismic_zone='4',
        seed=SEED,
        n_worlds=1,
        projected_crs=METRIC_CRS,
        merge_stacked=merge_stacked,
    )
    if merge_stacked:
        with pytest.raises(ValueError, match='cannot be given with merge_stacked'):
            compiler.compile(study_area)
        return
    result = compiler.compile(study_area)
    assert result.report.summary()['settings']['merge_stacked'] is False


def test_defaults_compile_as_their_values(
    study_area: CompilerInputs, whole: CompilationResult
) -> None:
    """A compiler without a state compiles stateless inputs in California."""
    given = HazusEQInventoryCompiler(
        seed=SEED,
        n_worlds=3,
        projected_crs=METRIC_CRS,
        state='California',
        seismic_zone='4',
        merge_stacked=False,
    ).compile(study_area)

    assert {a: asset.features for a, asset in whole.inventory.inventory.items()} == {
        a: asset.features for a, asset in given.inventory.inventory.items()
    }
    for frame in ('records', 'exclusions'):
        pd.testing.assert_frame_equal(
            getattr(whole.report, frame), getattr(given.report, frame)
        )
    assert {
        key: value
        for key, value in whole.report.summary().items()
        if key != 'timings'
    } == {
        key: value
        for key, value in given.report.summary().items()
        if key != 'timings'
    }
