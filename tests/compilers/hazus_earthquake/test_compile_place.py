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


"""Tests of the compilation of a place from its name or boundary."""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING, Any
from unittest import mock

import geopandas as gpd
import pyarrow.parquet as pq
import pytest
import shapely

from brails.compilers.hazus_earthquake import hazus_eq_inventory_compiler as hc
from brails.compilers.hazus_earthquake.hazus_eq_inventory_compiler import (
    CompilerInputs,
    HazusEQInventoryCompiler,
)
from brails.constants import DEFAULT_TIGER_VINTAGE, GEOGRAPHIC_CRS
from brails.scrapers.nsi_parser.nsi_parser import NSI_Parser
from brails.scrapers.overture_maps_scraper import overture_maps_scraper
from brails.scrapers.overture_maps_scraper.overture_maps_footprint_scraper import (
    OvertureMapsFootprintScraper,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    OVERTURE_ID_KEY,
)
from brails.scrapers.us_census_scrapers import tiger_files
from brails.scrapers.us_census_scrapers.tiger_line_scraper import TIGERLineScraper
from brails.types.census_boundary import CensusBoundary
from brails.types.region_boundary import RegionBoundary
from brails.utils.census_api_key import CENSUS_API_KEY_NAME
from tests.compilers.hazus_earthquake.compiler_fixtures import (
    BG_EAST,
    BG_FAR,
    BG_NORTH,
    BG_WEST,
    FOOTPRINTS,
    METRIC_CRS,
    RELEASE,
    SEED,
    build_inputs,
    lonlat,
    square,
)
from tests.local_files import overture_table, zip_shapefile

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('no_network')

# The place, inside the western block group, in meters east and north of the
# fixture origin.
PLACE = ((-40, -40), (190, -40), (190, 240), (-40, 240))

# The corners of each block group of California, and whether it has land.
BLOCK_GROUPS = {
    BG_WEST: (((-50, -50), (200, -50), (200, 250), (-50, 250)), True),
    BG_EAST: (((200, -50), (450, -50), (450, 250), (200, 250)), True),
    BG_NORTH: (((-50, 250), (450, 250), (450, 450), (-50, 450)), True),
    BG_FAR: (((2900, -100), (3100, -100), (3100, 100), (2900, 100)), True),
}

# A land block group that overlaps the place by a sliver, holds no building of
# the place, and reaches 6 km east.
BG_SLIVER = '060750101005'
SLIVER = (((185, -3000), (6000, -3000), (6000, -35), (185, -35)), True)

# Each state's FIPS code, name and corners.
STATES = {
    '06': (
        'California',
        ((-9000, -9000), (9000, -9000), (9000, 9000), (-9000, 9000)),
    ),
    '32': ('Nevada', ((9000, -9000), (20000, -9000), (20000, 9000), (9000, 9000))),
}


def polygon(corners: tuple[tuple[float, float], ...]) -> shapely.Polygon:
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


class Services:
    """
    The files and stand-ins from_place reads, with each call recorded in order.

    The TIGER/Line files are zipped shapefiles placed in ``tiger_files``'s memory,
    the Overture footprints a local GeoParquet file, and the NSI records come from
    a stand-in for ``NSI_Parser._get_nsi_data``; the Census API key check is a
    stand-in.

    Attributes:
        calls (list[tuple[str, object]]):
            Each call's service and what it was asked for.
        states (dict[str, tuple]):
            Each state's FIPS code to its name and corners.
        block_groups (dict[str, tuple]):
            The block groups of California, as :data:`BLOCK_GROUPS` holds them.
        footprints (dict[str, tuple]):
            Overture ids to their corners as the fixture gives them.
    """

    def __init__(self, folder: Path) -> None:
        """
        Serve the fixture area in California.

        Args:
            folder (Path):
                A folder for the files.
        """
        self.folder = folder
        self.calls: list[tuple[str, object]] = []
        self.states = dict(STATES)
        self.block_groups = dict(BLOCK_GROUPS)
        self.footprints = {
            asset_id: (x, y, side)
            for asset_id, (x, y, side, *_) in FOOTPRINTS.items()
        }

    def serve_tiger(self, vintage: int) -> None:
        """
        Place the state, block-group and place files of a release in memory.

        Args:
            vintage (int):
                The year of the release.
        """
        files = {
            ('STATE', 'us'): gpd.GeoDataFrame(
                {
                    'STATEFP': list(self.states),
                    'NAME': [name for name, _ in self.states.values()],
                },
                geometry=[polygon(corners) for _, corners in self.states.values()],
                crs=GEOGRAPHIC_CRS,
            ),
            ('PLACE', '06'): gpd.GeoDataFrame(
                {
                    'GEOID': ['0600001'],
                    'NAME': ['Placeville'],
                    'NAMELSAD': ['Placeville city'],
                    'CLASSFP': ['C1'],
                    'FUNCSTAT': ['A'],
                    'ALAND': [1.0],
                },
                geometry=[polygon(PLACE)],
                crs=GEOGRAPHIC_CRS,
            ),
        }
        for fips in self.states:
            rows = (
                self.block_groups
                if fips == '06'
                else {
                    f'{fips}0000000001': (
                        ((15000, 0), (15100, 0), (15100, 100), (15000, 100)),
                        True,
                    )
                }
            )
            files['BG', fips] = gpd.GeoDataFrame(
                {
                    'GEOID': list(rows),
                    'ALAND': [1000.0 if land else 0.0 for _, land in rows.values()],
                },
                geometry=[polygon(ring) for ring, _ in rows.values()],
                crs=GEOGRAPHIC_CRS,
            )
        for (layer, area), frame in files.items():
            url = tiger_files.tiger_file_url(layer, area, vintage)
            folder = self.folder / f'{vintage}_{layer}_{area}'
            tiger_files._downloads[url] = zip_shapefile(frame, folder)

    def overture_file(self) -> str:
        """
        Write the footprints as a local file of Overture rows.

        Returns:
            str:
                The file's path.
        """
        geometries = [
            shapely.Polygon(square(x, y, side))
            for x, y, side in self.footprints.values()
        ]
        table = overture_table(list(self.footprints), geometries)
        path = self.folder / 'buildings.parquet'
        pq.write_table(table, path)
        return str(path)

    def overture_options(self) -> dict[str, str]:
        """
        Return the Overture options that read the local file of the footprints.

        Returns:
            dict[str, str]:
                ``'local_path'`` and ``'overtureRelease'``.
        """
        return {'local_path': self.overture_file(), 'overtureRelease': RELEASE}

    def nsi_records(self, bbox: tuple) -> dict:
        """
        Return the fixture's NSI records as NSI_Parser._get_nsi_data does.

        Args:
            bbox (tuple):
                The bounding box asked for.

        Returns:
            dict:
                Each record's ``fd_id`` to its point and its fields, one of them
                null.
        """
        self.calls.append(('nsi', bbox))
        records = {}
        for fd_id, asset in build_inputs().nsi_points.inventory.items():
            fields = {**asset.features, 'bldgtype': None}
            records[fd_id] = (shapely.Point(asset.coordinates[0]), fields)
        return records


@pytest.fixture
def services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Services:
    """
    Install the files and stand-ins of every service from_place reads.

    Args:
        tmp_path (Path):
            The folder of the files.
        monkeypatch (pytest.MonkeyPatch):
            Replaces the Census API key check and NSI's request, and records the
            reads of the block groups and of the footprints.

    Returns:
        Services:
            The services, whose ``calls`` record what was asked.
    """
    served = Services(tmp_path)
    monkeypatch.setattr(tiger_files, '_downloads', {})

    class KeyCheck:
        def check_key(self, vintage: int) -> None:
            served.calls.append(('key', vintage))

    read_block_groups = TIGERLineScraper.get_block_groups
    read_footprints = OvertureMapsFootprintScraper.get_footprints

    def get_block_groups(scraper: TIGERLineScraper, region: RegionBoundary) -> Any:
        served.calls.append(('block groups', region.get_boundary()[0]))
        return read_block_groups(scraper, region)

    def get_footprints(
        scraper: OvertureMapsFootprintScraper, region: RegionBoundary
    ) -> Any:
        served.calls.append(('overture', region.get_boundary()[0]))
        return read_footprints(scraper, region)

    monkeypatch.setattr(hc, 'ACSTableScraper', KeyCheck)
    monkeypatch.setattr(TIGERLineScraper, 'get_block_groups', get_block_groups)
    monkeypatch.setattr(
        OvertureMapsFootprintScraper, 'get_footprints', get_footprints
    )
    monkeypatch.setattr(
        NSI_Parser, '_get_nsi_data', staticmethod(served.nsi_records)
    )

    # The routes to the Overture bucket and its release list, which the socket
    # guard does not cover: a scraper built without local_path fails at once.
    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = 'A test tried to reach the Overture bucket or its release list.'
        raise AssertionError(msg)

    monkeypatch.setattr(overture_maps_scraper.fs, 'S3FileSystem', refuse)
    monkeypatch.setattr(urllib.request, 'urlopen', refuse)
    return served


def gather(services: Services, place: object, **options: Any) -> CompilerInputs:
    """
    Serve the files and run from_place on the local Overture file.

    Args:
        services (Services):
            The services, as the test left them.
        place (object):
            The place.
        **options (Any):
            Further keyword arguments of from_place.

    Returns:
        CompilerInputs:
            What from_place returns.
    """
    services.serve_tiger(options.get('block_group_vintage', DEFAULT_TIGER_VINTAGE))
    return CompilerInputs.from_place(
        place,
        overture_options={
            **services.overture_options(),
            **options.pop('overture_options', {}),
        },
        **options,
    )


def called(services: Services, name: str) -> list[object]:
    """
    Return what one service was asked, in order.

    Args:
        services (Services):
            The services.
        name (str):
            The service's name in ``calls``.

    Returns:
        list[object]:
            The arguments of its calls.
    """
    return [argument for service, argument in services.calls if service == name]


def overture_ids(inputs: CompilerInputs) -> set[str]:
    """
    Return the Overture ids of the footprints of the inputs.

    Args:
        inputs (CompilerInputs):
            The inputs.

    Returns:
        set[str]:
            The ids.
    """
    return {
        asset.features[OVERTURE_ID_KEY]
        for asset in inputs.footprints.inventory.values()
    }


def widened_bounds(geometry: object) -> Any:
    """
    Return the bounds of a geometry in the fixture's metric CRS.

    Args:
        geometry (object):
            A geometry in longitude and latitude.

    Returns:
        numpy.ndarray:
            Its bounds in meters.
    """
    return (
        gpd.GeoSeries([geometry], crs=GEOGRAPHIC_CRS).to_crs(METRIC_CRS).total_bounds
    )


class TestFromPlace:
    """from_place reads the area of a place once and keeps the inputs of its area."""

    def test_inputs_of_a_polygon(self, services: Services) -> None:
        """A polygon gives the footprints, points and block groups of its area."""
        inputs = gather(services, polygon(PLACE))

        assert [service for service, _ in services.calls][:1] == ['key']
        assert called(services, 'overture')
        # d01, 25 m², is below the scraper's area floor, and f01 is 3 km away.
        assert overture_ids(inputs) == set(FOOTPRINTS) - {'d01', 'f01'}
        assert sorted(inputs.block_groups['GEOID']) == [BG_WEST, BG_EAST, BG_NORTH]
        assert inputs.boundary.equals(polygon(PLACE))
        assert (inputs.buffer_m, inputs.study_area) == (hc.DEFAULT_DONOR_CAP_M, '')
        assert inputs.boundary_record == {'provider': 'polygon'}
        assert (inputs.state, inputs.merge_stacked) == ('California', True)
        assert (inputs.footprint_release, inputs.height_unit) == (RELEASE, 'ft')
        assert inputs.notices == ()
        assert inputs.nsi_points.inventory
        assert all(
            'bldgtype' not in asset.features
            for asset in inputs.nsi_points.inventory.values()
        )

    def test_area_compiled_cut_from_the_area_read(self, services: Services) -> None:
        """Footprints read past the area compiled are cut; NSI reads that area."""
        # The sliver brings its 6 km into the area read but holds no building of
        # the place, so the area compiled ends 2 km past the western block group.
        services.block_groups[BG_SLIVER] = SLIVER
        services.footprints['z01'] = (5000, -1000, 12)
        inputs = gather(services, polygon(PLACE))

        (read,) = called(services, 'overture')
        assert read.covers(shapely.Point(lonlat(5006, -994)))
        assert read.covers(shapely.Point(lonlat(3006, 6)))
        assert overture_ids(inputs) == set(FOOTPRINTS) - {'d01', 'f01'}
        assert {BG_SLIVER, BG_FAR} <= set(inputs.block_groups['GEOID'])
        (bbox,) = called(services, 'nsi')
        west = widened_bounds(polygon(BLOCK_GROUPS[BG_WEST][0]))
        nsi = widened_bounds(shapely.box(*bbox))
        # The area compiled, widened by the join cap, seen through its bounding box
        # in longitude and latitude.
        assert all(2015.0 <= abs(gap) < 2050.0 for gap in nsi - west)
        assert not shapely.box(*bbox).covers(read)

    def test_one_read_over_the_place_and_its_land_block_groups(
        self, services: Services
    ) -> None:
        """One Overture read covers the place and its land block groups, widened."""
        gather(services, polygon(PLACE), buffer_m=500.0)

        (read,) = called(services, 'overture')
        west = widened_bounds(polygon(BLOCK_GROUPS[BG_WEST][0]))
        assert all(
            abs(abs(gap) - 500.0) < 1.0 for gap in widened_bounds(read) - west
        )

    def test_footprints_in_no_block_group_dropped(self, services: Services) -> None:
        """A footprint of the area in no block group is left out, with a notice."""
        services.footprints['x01'] = (1000, 1000, 12)
        inputs = gather(services, polygon(PLACE))

        assert 'x01' not in overture_ids(inputs)
        assert inputs.notices == (
            '1 footprint(s) of the area compiled lie in no block group, such as '
            'across a national border or offshore, and were left out.',
        )

    def test_block_group_without_land_noted(self, services: Services) -> None:
        """An inside building in a block group without land is named."""
        # The western block group, without land, reaches 5 km east.
        services.block_groups[BG_WEST] = (
            ((-50, -50), (5000, -50), (5000, 250), (-50, 250)),
            False,
        )
        inputs = gather(services, polygon(PLACE))

        (notice,) = inputs.notices
        assert notice.startswith('22 building(s) inside the region lie in block')

    def test_census_name(self, services: Services) -> None:
        """A name is found in the census files of the release, its state given."""
        # Most of the place lies in Nevada by the state file, which a census
        # boundary does not read.
        services.states['32'] = ('Nevada', PLACE)
        inputs = gather(services, 'Placeville, CA', block_group_vintage=2022)

        assert inputs.state == 'California'
        assert inputs.study_area == 'Placeville city, California'
        assert inputs.boundary_record['geoid'] == '0600001'
        assert inputs.boundary_record['vintage'] == 2022

    def test_census_boundary_of_another_release_noted(
        self, services: Services
    ) -> None:
        """A census boundary from a release other than the block groups' is noted."""
        services.serve_tiger(2022)
        inputs = gather(services, CensusBoundary('Placeville, CA', vintage=2022))

        assert inputs.notices == (
            'The boundary comes from the TIGER/Line 2022 files and the block groups '
            'from the 2023 files.',
        )

    def test_osm_name(self, services: Services) -> None:
        """A name with source='osm' reaches OpenStreetMap, which the record names."""
        with mock.patch.object(
            RegionBoundary,
            'get_boundary',
            return_value=(polygon(PLACE), 'Placeville', 123),
        ) as lookup:
            inputs = gather(services, 'Placeville, CA', source='osm')

        lookup.assert_called_once()
        assert inputs.boundary_record == {
            'provider': 'osm',
            'name': 'Placeville',
            'osm_id': 123,
        }

    def test_repaired_polygon_keeps_its_area(self, services: Services) -> None:
        """An invalid polygon is repaired, and only its polygonal parts are kept."""
        spiked = polygon(
            ((-40, -40), (190, -40), (190, 240), (300, 300), (190, 240), (-40, 240))
        )
        assert not spiked.is_valid
        inputs = gather(services, spiked)

        assert isinstance(inputs.boundary, shapely.Polygon)
        assert inputs.boundary.equals(polygon(PLACE))

    def test_boundary_without_area_refused(self, services: Services) -> None:
        """A boundary that encloses no area, even as repaired, is refused."""
        folded = polygon(((-40, -40), (190, -40), (-40, -40)))
        with pytest.raises(ValueError, match='encloses no area, even as repaired'):
            gather(services, folded)
        assert not called(services, 'block groups')

    @pytest.mark.parametrize(
        ('change', 'options', 'match'),
        [
            ('nevada', {}, 'in each of California and Nevada, and a compilation'),
            ('texas', {}, 'Census West region only, not Texas'),
            (
                'texas and nevada',
                {},
                'in each of Texas and Nevada, and a compilation',
            ),
            ('nowhere', {}, 'Less than 1% of the area of the region lies in any'),
            (
                None,
                {'max_block_groups': 2},
                'area read around the region intersects 3',
            ),
            (None, {'overture_options': {'id_scheme': 'index'}}, 'sets id_scheme'),
        ],
    )
    def test_refused_before_footprints(
        self, services: Services, change: str | None, options: dict, match: str
    ) -> None:
        """No state, two states, a refused state, a large area or a fixed option."""
        if change and 'nevada' in change:
            services.states['06'] = (
                'California',
                ((-9000, -9000), (100, -9000), (100, 9000), (-9000, 9000)),
            )
            services.states['32'] = (
                'Nevada',
                ((100, -9000), (20000, -9000), (20000, 9000), (100, 9000)),
            )
        if change and 'texas' in change:
            services.states['06'] = ('Texas', services.states['06'][1])
        if change == 'nowhere':
            services.states['06'] = (
                'California',
                ((9000, 9000), (9100, 9000), (9100, 9100)),
            )
        with pytest.raises(ValueError, match=match):
            gather(services, polygon(PLACE), **options)

        assert not called(services, 'overture')
        if change:
            assert not called(services, 'block groups')

    def test_two_states_refused_before_the_compiler_zone(
        self, services: Services
    ) -> None:
        """A second state is refused before a compiler without a zone for either."""
        services.states['06'] = (
            'Oregon',
            ((-9000, -9000), (100, -9000), (100, 9000), (-9000, 9000)),
        )
        services.states['32'] = (
            'Nevada',
            ((100, -9000), (20000, -9000), (20000, 9000), (100, 9000)),
        )
        with pytest.raises(ValueError, match='in each of Oregon and Nevada'):
            gather(
                services,
                polygon(PLACE),
                compiler=HazusEQInventoryCompiler(seed=SEED),
            )
        assert not called(services, 'block groups')

    @pytest.mark.parametrize(
        ('buffer_m', 'error'),
        [(-1.0, ValueError), (float('nan'), ValueError), (None, TypeError)],
    )
    def test_buffer_checked(
        self, services: Services, buffer_m: object, error: type
    ) -> None:
        """A buffer that is not a finite number of at least 0 is refused first."""
        with pytest.raises(error, match='buffer_m'):
            gather(services, polygon(PLACE), buffer_m=buffer_m)
        assert services.calls == []

    def test_no_key_stops_before_downloads(
        self, services: Services, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a key, from_place stops before any file is read."""
        monkeypatch.delenv(CENSUS_API_KEY_NAME, raising=False)

        class NoKey:
            def __init__(self) -> None:
                msg = 'No Census API key was found.'
                raise LookupError(msg)

        monkeypatch.setattr(hc, 'ACSTableScraper', NoKey)
        with pytest.raises(LookupError, match='No Census API key'):
            gather(services, polygon(PLACE))
        assert services.calls == []

    def test_no_footprint_inside_refused(self, services: Services) -> None:
        """A place with no footprint inside it is refused before NSI is read."""
        with pytest.raises(
            ValueError, match='No Overture footprint read lies inside'
        ):
            gather(
                services, polygon(((1000, 0), (1100, 0), (1100, 100), (1000, 100)))
            )
        assert not called(services, 'nsi')


class TestCompilePlace:
    """compile_place compiles what from_place gathers, refusing a zone early."""

    def test_compile_place_compiles_from_place(self) -> None:
        """compile_place passes itself and its donor cap to from_place to compile."""
        inputs = build_inputs()._replace(notices=('A notice of the inputs.',))
        compiler = HazusEQInventoryCompiler(
            seed=SEED,
            n_worlds=1,
            projected_crs=METRIC_CRS,
            imputer_options={'cap_m': 3000.0},
        )
        with mock.patch.object(
            CompilerInputs, 'from_place', return_value=inputs
        ) as gather_inputs:
            result = compiler.compile_place('Placeville, CA', source='osm')

        gather_inputs.assert_called_once_with(
            'Placeville, CA', compiler=compiler, source='osm', buffer_m=3000.0
        )
        expected = compiler.compile(inputs)
        assert {
            a: asset.features for a, asset in result.inventory.inventory.items()
        } == {a: asset.features for a, asset in expected.inventory.inventory.items()}
        assert result.report.summary()['notices'][0] == 'A notice of the inputs.'

    def test_compile_place_source_defaults_to_census(self) -> None:
        """compile_place without source passes source='census' to from_place."""
        compiler = HazusEQInventoryCompiler(seed=SEED, n_worlds=1)
        with (
            mock.patch.object(
                CompilerInputs, 'from_place', return_value=build_inputs()
            ) as gather_inputs,
            mock.patch.object(compiler, 'compile'),
        ):
            compiler.compile_place('Placeville, CA')

        assert gather_inputs.call_args.kwargs['source'] == 'census'

    def test_zone_refused_before_footprints(self, services: Services) -> None:
        """A place outside California without a zone stops before footprints."""
        services.states['06'] = ('Oregon', services.states['06'][1])
        services.serve_tiger(DEFAULT_TIGER_VINTAGE)
        with pytest.raises(ValueError, match='built without seismic_zone'):
            HazusEQInventoryCompiler(seed=SEED).compile_place(
                polygon(PLACE), overture_options=services.overture_options()
            )
        assert not called(services, 'overture')

    def test_compile_of_from_place_inputs(self, services: Services) -> None:
        """A compiler without settings merges, takes the state, keeps the inside."""
        given = build_inputs()
        inputs = gather(services, polygon(PLACE))._replace(
            **{
                name: getattr(given, name)
                for name in (
                    'b25024_block_groups',
                    'b25034_block_groups',
                    'b25034_tracts',
                    'b25127_tracts',
                    'mh_park_points',
                )
            }
        )
        result = HazusEQInventoryCompiler(
            seed=SEED, n_worlds=1, projected_crs=METRIC_CRS
        ).compile(inputs)

        summary = result.report.summary()
        settings = summary['settings']
        assert (settings['state'], settings['merge_stacked']) == ('California', True)
        assert 'stacked_parts' in result.logs
        assert summary['boundary']['outside'] > 0
        assert any(n.endswith('so zone 4 applies.') for n in summary['notices'])
