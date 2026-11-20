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

"""Tests of the Hazus earthquake inventory compiler reading inputs left as None."""

from __future__ import annotations

import functools
import json
import socket
import sys
from typing import TYPE_CHECKING, Callable
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests

from brails.compilers.hazus_earthquake import hazus_eq_inventory_compiler as hc
from brails.compilers.hazus_earthquake.hazus_eq_inventory_compiler import (
    CompilationResult,
    CompilerInputs,
    HazusEQInventoryCompiler,
)
from brails.constants import BLOCK_GROUP_KEY, GEOGRAPHIC_CRS
from brails.scrapers.us_census_scrapers.acs_table_scraper import ACS5_URL
from brails.scrapers.us_census_scrapers.tiger_line_scraper import STATE_NAME_COLUMN
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.census_api_key import CENSUS_API_KEY_NAME
from brails.utils.tiger_files import DEFAULT_TIGER_VINTAGE
from tests.compilers.hazus_earthquake.compiler_fixtures import (
    BG_EAST,
    BG_WEST,
    METRIC_CRS,
    MH_PARK,
    SEED,
    TRACT,
    build_inputs,
    lonlat,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# The table and geography requested for each census table field of CompilerInputs.
CENSUS_TABLES = {
    'b25024_block_groups': ('B25024', 'block group'),
    'b25034_block_groups': ('B25034', 'block group'),
    'b25034_tracts': ('B25034', 'tract'),
    'b25127_tracts': ('B25127', 'tract'),
}
CENSUS_API_KEY = 'notarealcensuskey0123456789abcdef'

# A value the Census Bureau writes for an estimate it does not publish.
CODED_ESTIMATE = -666666666.0

# The request the compiler sends to check the key before s01.
KEY_CHECK = {'get': 'NAME', 'for': 'state:01'}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every connection and name lookup, so no test reaches a service."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = 'A test tried to open a network connection.'
        raise OSError(msg)

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'getaddrinfo', refuse)


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
def compiled(study_area: CompilerInputs) -> Callable[[int], CompilationResult]:
    """
    Return a function that compiles the study area as given, once per world count.

    Args:
        study_area (CompilerInputs):
            The study area's data, from the ``study_area`` fixture.

    Returns:
        Callable[[int], CompilationResult]:
            ``compiled(n_worlds)`` gives the result at the fixture seed and CRS,
            shared by every test, which must not change it.
    """

    @functools.cache
    def run(n_worlds: int) -> CompilationResult:
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=n_worlds, projected_crs=METRIC_CRS
        )
        return compiler.compile(study_area)

    return run


def assert_same_compilation(
    result: CompilationResult, expected: CompilationResult
) -> None:
    """
    Assert that two runs compiled the same inventory and report rows.

    Args:
        result (CompilationResult):
            The run under test.
        expected (CompilationResult):
            The run with every input given.
    """
    assert {
        a: asset.features for a, asset in result.inventory.inventory.items()
    } == {a: asset.features for a, asset in expected.inventory.inventory.items()}
    pd.testing.assert_frame_equal(result.report.records, expected.report.records)
    pd.testing.assert_frame_equal(
        result.report.exclusions, expected.report.exclusions
    )


@pytest.fixture
def census_api(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[dict]]:
    """
    Return a function that makes requests.Session.get answer as the Census Data API.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Replaces requests.Session.get and sets the key's environment variable.

    Returns:
        Callable[..., list[dict]]:
            A function taking the study area whose tables the API serves, and
            ``refuse=True`` to answer every request with the page for a key that is
            not valid. It returns the list that receives the URL and parameters of
            each request, in order.
    """
    monkeypatch.setenv(CENSUS_API_KEY_NAME, CENSUS_API_KEY)

    def install(inputs: CompilerInputs, *, refuse: bool = False) -> list[dict]:
        served = {
            request: getattr(inputs, name) for name, request in CENSUS_TABLES.items()
        }
        served['B25024', 'block group'] = served['B25024', 'block group'].set_index(
            'geoid'
        )
        calls = []

        def get(
            _session: requests.Session,
            url: str,
            params: dict[str, str],
            timeout: float,  # noqa: ARG001
        ) -> requests.Response:
            calls.append({'url': url, **params})
            answer = requests.Response()
            answer.status_code = 200
            answer.url = url
            if refuse:
                answer._content = b'<html>invalid key</html>'
                answer.headers['Content-Type'] = 'text/html'
                answer.url = 'https://api.census.gov/data/invalid_key.html'
                return answer
            if params['get'] == KEY_CHECK['get']:
                rows = [['NAME', 'state'], ['Alabama', '01']]
            else:
                frame = served[
                    params['get'][len('group(') : -1], params['for'][: -len(':*')]
                ]
                state, county = params['in'].replace('state:', '').split(' county:')
                geography = ['state', 'county', 'tract', 'block group']
                levels = 3 if len(frame.index[0]) == len(TRACT) else 4
                bounds = (0, 2, 5, 11, 12)
                rows = [[*frame.columns, *geography[:levels]]]
                for geoid, values in frame.iterrows():
                    if geoid.startswith(state + county):
                        parts = [
                            geoid[a:b]
                            for a, b in zip(bounds, bounds[1:])
                            if b <= len(geoid)
                        ]
                        rows.append([*(str(value) for value in values), *parts])
            answer._content = json.dumps(rows).encode()
            answer.headers['Content-Type'] = 'application/json;charset=utf-8'
            return answer

        monkeypatch.setattr(requests.Session, 'get', get)
        return calls

    return install


class TestCensusTables:
    """Census tables given as None are read from the Census Data API."""

    def test_tables_read_compile_as_given(
        self,
        study_area: CompilerInputs,
        compiled: Callable[[int], CompilationResult],
        census_api: Callable[..., list[dict]],
    ) -> None:
        """Tables read from the API give the inventory and report of tables given."""
        calls = census_api(study_area)
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS, acs_vintage=2023
        ).compile(study_area._replace(**dict.fromkeys(CENSUS_TABLES)))

        release = ACS5_URL.format(vintage=2023)
        assert calls[0] == {'url': release, **KEY_CHECK, 'key': CENSUS_API_KEY}
        assert [
            (call['url'], call['get'], call['for'], call['in'], call['key'])
            for call in calls[1:]
        ] == [
            (
                release,
                f'group({table})',
                f'{geography}:*',
                'state:06 county:075',
                CENSUS_API_KEY,
            )
            for table, geography in CENSUS_TABLES.values()
        ]
        assert_same_compilation(result, compiled(3))
        summary = result.report.summary()
        assert summary['settings']['acs_vintage'] == 2023
        assert any(
            'ACS 2023 five-year' in notice and '06075' in notice
            for notice in summary['notices']
        )

    @pytest.mark.parametrize('county', ['06081', '32003'])
    def test_each_county_is_requested_once_per_table(
        self,
        study_area: CompilerInputs,
        census_api: Callable[..., list[dict]],
        county: str,
    ) -> None:
        """Each table is requested once per county, in one state or across two."""
        # The eastern block group moves to the other county, with its rows of the
        # block-group tables, and its tract's rows are copied under the new tract.
        block_group, tract = county + BG_EAST[5:], county + BG_EAST[5:11]
        polygons = study_area.block_groups.copy()
        polygons['GEOID'] = polygons['GEOID'].replace(BG_EAST, block_group)
        b25024 = study_area.b25024_block_groups.copy()
        b25024['geoid'] = b25024['geoid'].replace(BG_EAST, block_group)
        spread = study_area._replace(
            block_groups=polygons,
            b25024_block_groups=b25024,
            b25034_block_groups=study_area.b25034_block_groups.rename(
                index={BG_EAST: block_group}
            ),
            **{
                name: pd.concat(
                    [
                        getattr(study_area, name),
                        getattr(study_area, name).rename(index={TRACT: tract}),
                    ]
                )
                for name in ('b25034_tracts', 'b25127_tracts')
            },
        )
        calls = census_api(spread)
        compiler = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=2, projected_crs=METRIC_CRS
        )
        result = compiler.compile(spread._replace(**dict.fromkeys(CENSUS_TABLES)))

        assert sorted(
            (call['get'], call['for'], call['in']) for call in calls[1:]
        ) == sorted(
            (f'group({table})', f'{geography}:*', place)
            for table, geography in CENSUS_TABLES.values()
            for place in (
                'state:06 county:075',
                f'state:{county[:2]} county:{county[2:]}',
            )
        )
        assert block_group in {
            asset.features[BLOCK_GROUP_KEY]
            for asset in result.inventory.inventory.values()
        }
        assert_same_compilation(result, compiler.compile(spread))

    @pytest.mark.parametrize('name', list(CENSUS_TABLES))
    def test_given_tables_are_not_requested(
        self,
        study_area: CompilerInputs,
        compiled: Callable[[int], CompilationResult],
        census_api: Callable[..., list[dict]],
        name: str,
    ) -> None:
        """Only the table given as None is requested, and the run is unchanged."""
        calls = census_api(study_area)
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        ).compile(study_area._replace(**{name: None}))

        table, geography = CENSUS_TABLES[name]
        assert [(call['get'], call['for']) for call in calls] == [
            (KEY_CHECK['get'], KEY_CHECK['for']),
            (f'group({table})', f'{geography}:*'),
        ]
        assert_same_compilation(result, compiled(1))

    def test_no_key_stops_before_s01(
        self,
        study_area: CompilerInputs,
        census_api: Callable[..., list[dict]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A table given as None and no key in any store stop the run before s01."""
        calls = census_api(study_area)
        # No key in the environment, in a Colab secret or in the home folder.
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('USERPROFILE', str(tmp_path))
        monkeypatch.delenv(CENSUS_API_KEY_NAME)
        monkeypatch.setitem(sys.modules, 'google.colab.userdata', None)
        with (
            mock.patch.object(
                hc.CensusTractScraper, 'get_census_geographies'
            ) as assign,
            pytest.raises(
                LookupError, match=f'{CENSUS_API_KEY_NAME}.*leaves b25127_tracts'
            ),
        ):
            HazusEQInventoryCompiler(seed=SEED, n_worlds=1).compile(
                study_area._replace(b25127_tracts=None)
            )
        assert not assign.called
        assert calls == []

    def test_value_that_cannot_be_a_key_names_the_tables(
        self,
        study_area: CompilerInputs,
        census_api: Callable[..., list[dict]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A variable that cannot be a key stops the run naming the tables."""
        calls = census_api(study_area)
        monkeypatch.setenv(CENSUS_API_KEY_NAME, 'notakey-')
        with pytest.raises(ValueError, match='environment variable.*leaves b25127'):
            HazusEQInventoryCompiler(seed=SEED, n_worlds=1).compile(
                study_area._replace(b25127_tracts=None)
            )
        assert calls == []

    def test_refused_key_stops_before_s01(
        self,
        study_area: CompilerInputs,
        census_api: Callable[..., list[dict]],
    ) -> None:
        """A refused key stops the run before s01 and is named by its store only."""
        calls = census_api(study_area, refuse=True)
        with (
            mock.patch.object(
                hc.CensusTractScraper, 'get_census_geographies'
            ) as assign,
            pytest.raises(
                RuntimeError, match='environment variable.*leaves b25127_tracts'
            ) as raised,
        ):
            HazusEQInventoryCompiler(seed=SEED, n_worlds=1).compile(
                study_area._replace(b25127_tracts=None)
            )
        assert not assign.called
        assert len(calls) == 1
        assert CENSUS_API_KEY not in str(raised.value)

    def test_given_tables_need_no_key(
        self,
        study_area: CompilerInputs,
        census_api: Callable[..., list[dict]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """With every table given, the run needs no key and records no release."""
        calls = census_api(study_area)
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('USERPROFILE', str(tmp_path))
        monkeypatch.delenv(CENSUS_API_KEY_NAME)
        monkeypatch.setitem(sys.modules, 'google.colab.userdata', None)
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        ).compile(study_area)

        assert calls == []
        summary = result.report.summary()
        assert 'acs_vintage' not in summary['settings']
        assert not any('Census Data API' in notice for notice in summary['notices'])

    def test_coded_estimate_the_sampler_reads_is_refused(
        self,
        study_area: CompilerInputs,
        census_api: Callable[..., list[dict]],
    ) -> None:
        """A coded estimate in a footprint's block group stops the run, naming it."""
        b25034 = study_area.b25034_block_groups.copy()
        b25034.loc[BG_WEST, 'B25034_002E'] = CODED_ESTIMATE
        census_api(study_area._replace(b25034_block_groups=b25034))
        with pytest.raises(
            ValueError, match=f'b25034_block_groups, by block group.*{BG_WEST}'
        ):
            HazusEQInventoryCompiler(seed=SEED, n_worlds=1).compile(
                study_area._replace(b25034_block_groups=None)
            )

    def test_coded_estimates_the_sampler_does_not_read_are_ignored(
        self,
        study_area: CompilerInputs,
        compiled: Callable[[int], CompilationResult],
        census_api: Callable[..., list[dict]],
    ) -> None:
        """Coded estimates the sampler does not read do not stop the run."""
        # A block group of the county that holds no footprint, and the table's total,
        # which the sampler does not read.
        b25034 = pd.concat(
            [
                study_area.b25034_block_groups,
                study_area.b25034_block_groups.loc[[BG_WEST]].rename(
                    index={BG_WEST: '060759999001'}
                )
                * CODED_ESTIMATE,
            ]
        ).assign(B25034_001E=CODED_ESTIMATE)
        census_api(study_area._replace(b25034_block_groups=b25034))
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        ).compile(study_area._replace(b25034_block_groups=None))
        assert_same_compilation(result, compiled(1))

    @pytest.mark.parametrize('vintage', ['2024', True, 2024.0])
    def test_vintage_must_be_an_integer(self, vintage: object) -> None:
        """An ACS vintage that is not an integer year is refused."""
        with pytest.raises(TypeError, match='acs_vintage must be an integer'):
            HazusEQInventoryCompiler(seed=SEED, acs_vintage=vintage)


class TestBlockGroups:
    """Block groups given as None are read from the TIGER/Line files."""

    @pytest.fixture
    def read(self, study_area: CompilerInputs) -> Iterator[mock.MagicMock]:
        """
        Stand in for TIGERLineScraper.get_block_groups with the fixture's polygons.

        Args:
            study_area (CompilerInputs):
                The study area, whose block groups are served with the columns of a
                TIGER/Line file that the compiler does not read.

        Yields:
            unittest.mock.MagicMock:
                The stand-in, whose calls give the scraper and the area.
        """
        served = study_area.block_groups.to_crs('EPSG:4269').assign(
            **{STATE_NAME_COLUMN: 'California', 'ALAND': 1000}
        )
        with mock.patch.object(
            hc.TIGERLineScraper,
            'get_block_groups',
            autospec=True,
            return_value=served,
        ) as stand_in:
            yield stand_in

    def test_block_groups_read_compile_as_given(
        self,
        study_area: CompilerInputs,
        compiled: Callable[[int], CompilationResult],
        read: mock.MagicMock,
    ) -> None:
        """Block groups read for the footprints' bounding box give the given run."""
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
        ).compile(study_area._replace(block_groups=None))

        ((scraper, area),) = [call.args for call in read.call_args_list]
        assert scraper.vintage == DEFAULT_TIGER_VINTAGE
        assert all(
            area.geometry.covers(asset.get_geometry())
            for asset in study_area.footprints.inventory.values()
        )
        assert_same_compilation(result, compiled(3))
        summary = result.report.summary()
        assert summary['settings']['block_group_vintage'] == DEFAULT_TIGER_VINTAGE
        assert any(
            f'TIGER/Line {DEFAULT_TIGER_VINTAGE} files of California' in notice
            for notice in summary['notices']
        )

    def test_given_block_groups_record_no_release(
        self, compiled: Callable[[int], CompilationResult]
    ) -> None:
        """A run given its block groups records no TIGER/Line release."""
        assert 'block_group_vintage' not in compiled(1).report.summary()['settings']

    def test_failed_download_names_the_input_before_s01(
        self, study_area: CompilerInputs, read: mock.MagicMock
    ) -> None:
        """A failed download stops the run before s01, naming the field to give."""
        read.side_effect = requests.ConnectionError('no route')
        with (
            mock.patch.object(
                hc.CensusTractScraper, 'get_census_geographies'
            ) as assign,
            pytest.raises(
                requests.ConnectionError,
                match='no route.*CompilerInputs.block_groups',
            ),
        ):
            HazusEQInventoryCompiler(seed=SEED).compile(
                study_area._replace(block_groups=None)
            )
        assert not assign.called

    @pytest.mark.parametrize('vintage', ['2023', True])
    def test_block_group_vintage_must_be_an_integer(self, vintage: object) -> None:
        """A block-group vintage that is not an integer year is refused."""
        with pytest.raises(
            TypeError, match='block_group_vintage must be an integer'
        ):
            HazusEQInventoryCompiler(seed=SEED, block_group_vintage=vintage)

    def test_vintages_of_two_decades_refused_only_where_both_are_read(
        self,
        study_area: CompilerInputs,
        census_api: Callable[..., list[dict]],
        read: mock.MagicMock,
    ) -> None:
        """Vintages of two decades stop a run reading both, before any request."""
        calls = census_api(study_area)
        compiler = HazusEQInventoryCompiler(seed=SEED, acs_vintage=2019)
        with pytest.raises(ValueError, match='release from 2010 to 2019'):
            compiler.compile(
                study_area._replace(block_groups=None, b25127_tracts=None)
            )
        assert calls == []
        assert not read.called
        # Given every table, the block groups alone are read without a refusal.
        compiler.compile(study_area._replace(block_groups=None))
        assert read.called


class TestParkPoints:
    """Park points given as None are read from the HIFLD archive."""

    @pytest.fixture
    def parks(self) -> AssetInventory:
        """
        Return the fixture's park point as MobileHomeParkScraper returns it.

        Returns:
            AssetInventory:
                One point asset at the fixture's park.
        """
        inventory = AssetInventory()
        inventory.add_asset(7, Asset(7, [lonlat(*MH_PARK)], {'MHPID': 7}))
        return inventory

    def test_park_points_read_compile_as_given(
        self,
        study_area: CompilerInputs,
        compiled: Callable[[int], CompilationResult],
        parks: AssetInventory,
    ) -> None:
        """Park points read from the archive give the run with the points given."""

        def read(
            scraper: hc.MobileHomeParkScraper, _region: object
        ) -> AssetInventory:
            scraper.marker, scraper.written = '_committed_7', '2025-11-17T21:19:11Z'
            return parks

        with mock.patch.object(
            hc.MobileHomeParkScraper, 'get_assets', autospec=True, side_effect=read
        ):
            result = HazusEQInventoryCompiler(
                seed=SEED, n_worlds=3, projected_crs=METRIC_CRS
            ).compile(study_area._replace(mh_park_points=None))

        assert_same_compilation(result, compiled(3))
        summary = result.report.summary()
        assert summary['settings']['mh_park_write'] == {
            'marker': '_committed_7',
            'written': '2025-11-17T21:19:11Z',
        }
        assert any(
            notice.startswith('1 mobile-home park point(s)')
            and '_committed_7' in notice
            for notice in summary['notices']
        )

    def test_park_inventory_given_compiles_as_its_points(
        self,
        study_area: CompilerInputs,
        compiled: Callable[[int], CompilationResult],
        parks: AssetInventory,
    ) -> None:
        """An AssetInventory of park points compiles as the array of its points."""
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        ).compile(study_area._replace(mh_park_points=parks))
        assert_same_compilation(result, compiled(1))

    def test_empty_park_points_compile_without_a_read(
        self, study_area: CompilerInputs
    ) -> None:
        """An empty array of park points compiles, and no point is read."""
        result = HazusEQInventoryCompiler(seed=SEED, n_worlds=1).compile(
            study_area._replace(mh_park_points=np.zeros((0, 2)))
        )
        assert 'mh_park_write' not in result.report.summary()['settings']

    @pytest.mark.parametrize(
        ('budget_options', 'radius'),
        [({}, 300.0), ({'mh_park_radius_m': 500.0}, 500.0)],
    )
    def test_area_is_the_bounding_box_widened_by_the_budget_radius(
        self, study_area: CompilerInputs, budget_options: dict, radius: float
    ) -> None:
        """The points are read over the bounding box widened by the park radius."""
        # The stand-in stops the run, so the test sees only the area it was given.
        with (
            mock.patch.object(
                hc.MobileHomeParkScraper,
                'get_assets',
                autospec=True,
                side_effect=LookupError,
            ) as read,
            pytest.raises(LookupError),
        ):
            HazusEQInventoryCompiler(
                seed=SEED, budget_options=budget_options
            ).compile(study_area._replace(mh_park_points=None))

        ((_, area),) = [call.args for call in read.call_args_list]
        extent = gpd.GeoSeries(
            [study_area.footprints.get_extent(buffer='none')], crs=GEOGRAPHIC_CRS
        ).to_crs(METRIC_CRS)
        widened = gpd.GeoSeries([area.geometry], crs=GEOGRAPHIC_CRS).to_crs(
            METRIC_CRS
        )
        assert widened.iloc[0].covers(extent.iloc[0])
        gaps = [
            abs(a - b) for a, b in zip(widened.total_bounds, extent.total_bounds)
        ]
        assert all(abs(gap - radius) < 1.0 for gap in gaps)

    def test_failed_read_names_the_input_before_s01(
        self, study_area: CompilerInputs
    ) -> None:
        """A failed park read stops the run before s01, naming the field."""
        with (
            mock.patch.object(
                hc.MobileHomeParkScraper,
                'get_assets',
                autospec=True,
                side_effect=requests.ConnectionError('no route'),
            ),
            mock.patch.object(
                hc.CensusTractScraper, 'get_census_geographies'
            ) as assign,
            pytest.raises(
                requests.ConnectionError,
                match='no route.*CompilerInputs.mh_park_points',
            ),
        ):
            HazusEQInventoryCompiler(seed=SEED).compile(
                study_area._replace(mh_park_points=None)
            )
        assert not assign.called
