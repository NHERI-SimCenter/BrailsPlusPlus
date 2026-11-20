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

import pandas as pd
import pytest
import requests

from brails.compilers.hazus_earthquake import hazus_eq_inventory_compiler as hc
from brails.compilers.hazus_earthquake.hazus_eq_inventory_compiler import (
    CompilationResult,
    CompilerInputs,
    HazusEQInventoryCompiler,
)
from brails.constants import BLOCK_GROUP_KEY
from brails.scrapers.us_census_scrapers.acs_table_scraper import ACS5_URL
from brails.utils.census_api_key import CENSUS_API_KEY_NAME
from tests.compilers.hazus_earthquake.compiler_fixtures import (
    BG_EAST,
    BG_WEST,
    METRIC_CRS,
    SEED,
    TRACT,
    build_inputs,
)

if TYPE_CHECKING:
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
