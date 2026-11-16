# Copyright (c) 2025 The Regents of the University of California
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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.

"""Tests for CensusTractScraper in brails.scrapers.us_census_scrapers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
from requests import Response, exceptions
from shapely.geometry import Polygon, box, mapping

from brails.constants import BLOCK_GROUP_KEY, TRACT_KEY
from brails.scrapers.us_census_scrapers.census_tract_scraper import (
    CENSUS_BLOCK_GROUP_QUERY_URL,
    CENSUS_TRACT_QUERY_URL,
    LEVEL_BLOCK_GROUP,
    LEVEL_TRACT,
    CensusTractScraper,
)
from brails.types.asset_inventory import Asset, AssetInventory

# Use a local alias to avoid a hard dependency on pytest-mock at import time
MockerFixture = Any


@pytest.fixture
def census_api_fixture_path() -> Path:
    """Return the path to the canned TIGERweb GeoJSON fixture file."""
    return Path(__file__).parents[1] / 'fixtures' / 'census_api_response.json'


@pytest.fixture
def census_api_response(census_api_fixture_path: Path) -> dict[str, Any]:
    """Load and return the JSON content of the TIGERweb fixture file."""
    text = census_api_fixture_path.read_text(encoding='utf-8')
    return json.loads(text)


def _make_http_error(status_code: int) -> exceptions.HTTPError:
    """Create an HTTPError carrying a Response with the given status code."""
    resp = Response()
    resp.status_code = status_code
    # Minimal URL to satisfy potential tooling; not used by code under test
    resp.url = 'https://tigerweb.geo.census.gov/mock'
    return exceptions.HTTPError(response=resp)


# ---------------------
# Unit tests for _fetch_features
# ---------------------


def test_fetch_success_contract(
    mocker: MockerFixture, census_api_response: dict[str, Any]
) -> None:
    """Verify method parses a known, canned API response correctly.

    Arrange: mock requests.get to return the loaded fixture JSON.
    Act: call _fetch_features with dummy coords.
    Assert: the features are returned and requests.get was called once.
    """
    scraper = CensusTractScraper()

    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = census_api_response

    mock_get = mocker.patch('requests.get', return_value=mock_response)

    features = scraper._fetch_features(
        lon=-122.2585, lat=37.8719, retries=3, timeout=5, delay=0
    )

    assert features == census_api_response['features']
    feature = features[0]
    assert 'type' in feature
    assert 'geometry' in feature
    assert 'properties' in feature
    assert mock_get.call_count == 1
    assert mock_get.call_args.args[0] == CENSUS_TRACT_QUERY_URL


@pytest.mark.parametrize(
    'side_effect_factory',
    [
        lambda: exceptions.ConnectionError('conn reset'),
        lambda: exceptions.Timeout('timed out'),
        lambda: _make_http_error(500),  # 5xx should be retried
    ],
)
def test_fetch_retries_on_transient_errors(
    mocker: MockerFixture,
    side_effect_factory: Any,
    census_api_response: dict[str, Any],
) -> None:
    """Ensure transient errors are retried and succeed on the second attempt.

    The first call raises the transient error; the second returns success.
    """
    scraper = CensusTractScraper()

    class _OK:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return census_api_response

    # First call: raise transient error; second: success
    mock_get = mocker.patch(
        'requests.get',
        side_effect=[side_effect_factory(), _OK()],
    )
    # Avoid real sleep during retries
    mocker.patch(
        'brails.scrapers.us_census_scrapers.census_tract_scraper.time.sleep',
        return_value=None,
    )

    features = scraper._fetch_features(
        lon=-120.0, lat=35.0, retries=3, timeout=5, delay=0
    )

    assert features == census_api_response['features']
    assert mock_get.call_count == 2


def test_fetch_fails_fast_on_a_client_error(mocker: MockerFixture) -> None:
    """A 4xx HTTPError raised on the first call propagates without a retry."""
    scraper = CensusTractScraper()
    mock_get = mocker.patch('requests.get', side_effect=_make_http_error(400))
    # Avoid real sleep even though we do not expect retries
    mocker.patch(
        'brails.scrapers.us_census_scrapers.census_tract_scraper.time.sleep',
        return_value=None,
    )

    with pytest.raises(exceptions.HTTPError):
        scraper._fetch_features(lon=-77.0, lat=38.9, retries=3, timeout=5, delay=0)

    assert mock_get.call_count == 1


def test_fetch_returns_no_features_outside_the_layer(mocker: MockerFixture) -> None:
    """A 2xx response without features gives an empty list without a retry."""
    scraper = CensusTractScraper()
    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {'type': 'FeatureCollection', 'features': []}
    mock_get = mocker.patch('requests.get', return_value=mock_response)

    features = scraper._fetch_features(
        lon=-77.0, lat=38.9, url=CENSUS_BLOCK_GROUP_QUERY_URL, delay=0
    )

    assert features == []
    assert mock_get.call_count == 1
    assert mock_get.call_args.args[0] == CENSUS_BLOCK_GROUP_QUERY_URL


def test_fetch_exhausts_retries(mocker: MockerFixture) -> None:
    """Verify the method gives up after all retry attempts fail.

    Mock requests.get to always raise ConnectionError; assert it is called
    exactly `retries` times and that a final ConnectionError is raised.
    """
    scraper = CensusTractScraper()

    mock_get = mocker.patch(
        'requests.get',
        side_effect=exceptions.ConnectionError('dns failure'),
    )
    mocker.patch(
        'brails.scrapers.us_census_scrapers.census_tract_scraper.time.sleep',
        return_value=None,
    )

    with pytest.raises(exceptions.ConnectionError):
        scraper._fetch_features(lon=-100.0, lat=40.0, retries=3, timeout=5, delay=0)

    assert mock_get.call_count == 3


# ---------------------
# Integration tests for get_census_tracts
# ---------------------


def _square_polygon(
    center_lon: float, center_lat: float, half_size: float = 0.01
) -> Polygon:
    """Create a simple square polygon around a center point.

    The polygon is returned as a Shapely Polygon; use ``mapping(poly)`` to get
    a GeoJSON geometry dict.
    """
    return Polygon(
        [
            (center_lon - half_size, center_lat - half_size),
            (center_lon - half_size, center_lat + half_size),
            (center_lon + half_size, center_lat + half_size),
            (center_lon + half_size, center_lat - half_size),
            (center_lon - half_size, center_lat - half_size),
        ]
    )


@pytest.fixture
def empty_inventory() -> AssetInventory:
    """Return an empty AssetInventory instance."""
    return AssetInventory()


@pytest.fixture
def single_tract_inventory() -> AssetInventory:
    """Inventory with two assets located within the same tract polygon."""
    inv = AssetInventory()
    # Two nearby points in the same vicinity
    inv.add_asset_coordinates('a1', [[-120.0, 35.0]])
    inv.add_asset_coordinates('a2', [[-120.001, 35.0005]])
    return inv


@pytest.fixture
def two_tracts_inventory() -> tuple[
    AssetInventory, list[tuple[str, tuple[float, float]]]
]:
    """Inventory with assets placed in two distinct tracts.

    Returns a tuple of (inventory, [(asset_id, (lon, lat)), ...]) for convenience
    in building corresponding polygons and assertions.
    """
    inv = AssetInventory()
    # Add assets in a known order to control which tract is processed first
    inv.add_asset_coordinates('t1_a1', [[-120.0, 35.0]])  # tract T1
    inv.add_asset_coordinates('t1_a2', [[-120.002, 35.001]])  # tract T1
    inv.add_asset_coordinates('t2_a1', [[-77.0, 39.0]])  # tract T2
    asset_coords: list[tuple[str, tuple[float, float]]] = [
        ('t1_a1', (-120.0, 35.0)),
        ('t1_a2', (-120.002, 35.001)),
        ('t2_a1', (-77.0, 39.0)),
    ]
    return inv, asset_coords


def _feature_for_polygon(geoid: str, poly: Polygon) -> dict[str, Any]:
    """Create a TIGER-like feature dict for a given GEOID and polygon."""
    return {
        'type': 'Feature',
        'properties': {'GEOID': geoid},
        'geometry': mapping(poly),
    }


def test_get_tracts_input_validation() -> None:
    """get_census_tracts should raise TypeError for invalid input types."""
    scraper = CensusTractScraper()
    with pytest.raises(TypeError):
        scraper.get_census_tracts(asset_inventory=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        scraper.get_census_tracts(asset_inventory={})  # type: ignore[arg-type]


def test_get_tracts_happy_path(
    mocker: MockerFixture,
    two_tracts_inventory: tuple[
        AssetInventory, list[tuple[str, tuple[float, float]]]
    ],
) -> None:
    """End-to-end success: assets across two tracts are updated and cached.

    Assertions:
    1. _fetch_features is called once per unique tract (2 calls).
    2. Each asset gains the correct TRACT_GEOID.
    3. The returned cache includes GEOIDs with shapely geometries.
    """
    inv, asset_coords = two_tracts_inventory

    # Build two non-overlapping tract polygons covering the two regions
    tract1_poly = _square_polygon(-120.0, 35.0, half_size=0.02)
    tract2_poly = _square_polygon(-77.0, 39.0, half_size=0.02)

    feature_t1 = _feature_for_polygon('06001400100', tract1_poly)
    feature_t2 = _feature_for_polygon('11001006202', tract2_poly)

    # Side effects: first call returns T1 (covers first two points),
    # second call returns T2 (covers the remaining point)
    mock_fetch = mocker.patch.object(
        CensusTractScraper,
        '_fetch_features',
        side_effect=[[feature_t1], [feature_t2]],
    )

    scraper = CensusTractScraper()
    downloaded = scraper.get_census_tracts(inv)

    # 1. Called once per unique tract
    assert mock_fetch.call_count == 2

    # 2. Inventory updated with expected GEOIDs
    expected = {
        't1_a1': '06001400100',
        't1_a2': '06001400100',
        't2_a1': '11001006202',
    }
    for asset_id, geoid in expected.items():
        found, features = inv.get_asset_features(asset_id)
        assert found is True
        assert features.get(TRACT_KEY) == geoid

    # 3. Cache contains shapely geometries keyed by GEOID and matches expected polygons
    assert set(downloaded.keys()) == {'06001400100', '11001006202'}
    assert downloaded['06001400100'].equals(tract1_poly)
    assert downloaded['11001006202'].equals(tract2_poly)


def test_get_tracts_uses_cache_correctly(
    mocker: MockerFixture, single_tract_inventory: AssetInventory
) -> None:
    """Multiple assets in the same tract should not trigger redundant fetches."""
    poly = _square_polygon(-120.0, 35.0, half_size=0.02)
    feature = _feature_for_polygon('06001400100', poly)

    mock_fetch = mocker.patch.object(
        CensusTractScraper,
        '_fetch_features',
        return_value=[feature],
    )

    scraper = CensusTractScraper()
    inv = single_tract_inventory
    _ = scraper.get_census_tracts(inv)

    # Should be called only once because the first polygon will capture both points
    assert mock_fetch.call_count == 1

    # All assets updated with same GEOID
    for asset_id in inv.get_asset_ids():
        found, features = inv.get_asset_features(asset_id)
        assert found is True
        assert features.get(TRACT_KEY) == '06001400100'


@pytest.mark.parametrize(
    ('fetch', 'expected_exception', 'match'),
    [
        (
            {'return_value': []},
            ValueError,
            r'2 asset\(s\) lie outside every tract polygon: a1, a2\.',
        ),
        (
            {'side_effect': exceptions.ConnectionError('dns failure')},
            exceptions.ConnectionError,
            'dns failure',
        ),
    ],
)
def test_get_tracts_raises_and_writes_nothing(
    mocker: MockerFixture,
    single_tract_inventory: AssetInventory,
    fetch: dict[str, Any],
    expected_exception: type[BaseException],
    match: str,
) -> None:
    """A point outside every tract, or a failed query, raises before any write."""
    mocker.patch.object(CensusTractScraper, '_fetch_features', **fetch)
    before = deepcopy(single_tract_inventory.inventory)

    with pytest.raises(expected_exception, match=match):
        CensusTractScraper().get_census_tracts(single_tract_inventory)

    assert {
        asset_id: asset.features
        for asset_id, asset in single_tract_inventory.inventory.items()
    } == {asset_id: asset.features for asset_id, asset in before.items()}


def test_get_tracts_with_empty_inventory(
    mocker: MockerFixture, empty_inventory: AssetInventory
) -> None:
    """Empty inventory should result in zero API calls and an empty cache."""
    mock_fetch = mocker.patch.object(
        CensusTractScraper, '_fetch_features', autospec=True
    )

    scraper = CensusTractScraper()
    downloaded = scraper.get_census_tracts(empty_inventory)

    assert mock_fetch.call_count == 0
    assert downloaded == {}


@pytest.mark.live
def test_get_tracts_live_api_call() -> None:
    """Live E2E call against TIGERweb API to verify GEOID assignment.

    This test intentionally performs a real network call (no mocking) and is
    marked with the `live` marker so it can be excluded from default CI runs.

    Arrange: create an AssetInventory with one valid US coordinate (UC Berkeley).
    Act: call get_census_tracts with no mocking.
    Assert: the asset receives an 11-digit TRACT_GEOID string.
    """
    # Arrange
    inv = AssetInventory()
    # UC Berkeley: (lon, lat)
    inv.add_asset_coordinates('live1', [[-122.2585, 37.8719]])

    scraper = CensusTractScraper()

    # Act: perform a real API call
    _ = scraper.get_census_tracts(inv)

    # Assert: check that a valid 11-digit GEOID string was assigned
    geoid_length = 11
    found, features = inv.get_asset_features('live1')
    assert found is True
    geoid = features.get(TRACT_KEY)
    assert isinstance(geoid, str)
    assert len(geoid) == geoid_length
    assert geoid.isdigit()


# -------------------------------------------------
# Additional fixtures and tests: geometry handling for get_census_tracts
# -------------------------------------------------


@pytest.fixture
def point_only_inventory() -> AssetInventory:
    """AssetInventory containing only Point geometries.

    Two nearby points around (-120, 35) to ensure they fall in the same tract
    for a sufficiently large mock polygon.
    """
    inv = AssetInventory()
    inv.add_asset_coordinates('p1', [[-120.000, 35.000]])
    inv.add_asset_coordinates('p2', [[-120.002, 35.001]])
    return inv


@pytest.fixture
def polygon_only_inventory() -> AssetInventory:
    """AssetInventory containing only Polygon geometries.

    Create two small square polygons near (-120, 35). Coordinates are provided
    as closed linear rings as required by AssetInventory polygon handling.
    """
    inv = AssetInventory()
    poly1 = _square_polygon(-120.005, 35.002, half_size=0.01)
    poly2 = _square_polygon(-119.995, 34.998, half_size=0.01)

    coords1 = list(mapping(poly1)['coordinates'][0])
    coords2 = list(mapping(poly2)['coordinates'][0])

    inv.add_asset(
        'g1', Asset('g1', coordinates=[list(pt) for pt in coords1], features={})
    )
    inv.add_asset(
        'g2', Asset('g2', coordinates=[list(pt) for pt in coords2], features={})
    )
    return inv


@pytest.fixture
def mixed_geometry_inventory(
    polygon_only_inventory: AssetInventory,
) -> AssetInventory:
    """AssetInventory containing a mix of Point and Polygon geometries."""
    inv = AssetInventory()

    # Add one point
    inv.add_asset_coordinates('m_p', [[-120.001, 35.0005]])

    # Add one polygon by reusing coordinates from polygon_only_inventory's 'g1'
    poly_geojson = polygon_only_inventory.get_geojson()
    poly_feats = [
        f for f in poly_geojson['features'] if f['properties'].get('id') == 'g1'
    ]
    if poly_feats:
        coords = poly_feats[0]['geometry']['coordinates'][0]
        inv.add_asset(
            'm_g', Asset('m_g', coordinates=[list(pt) for pt in coords], features={})
        )

    return inv


@pytest.mark.parametrize(
    'inventory_fixture_name',
    [
        'point_only_inventory',
        'polygon_only_inventory',
        'mixed_geometry_inventory',
    ],
)
def test_get_census_tracts_handles_various_geometries(
    request: Any, mocker: MockerFixture, inventory_fixture_name: str
) -> None:
    """Ensure TRACT_GEOID is assigned correctly for Point, Polygon, and mixed inventories.

    Mock _fetch_features to return a tract polygon that covers the centroids
    of all assets so that a single GEOID is assigned to all assets regardless of
    original geometry type.
    """
    inv: AssetInventory = request.getfixturevalue(inventory_fixture_name)

    # Build a covering polygon based on centroids of current inventory
    gdf = gpd.GeoDataFrame.from_features(
        inv.get_geojson()['features'], crs='EPSG:4326'
    )
    if not gdf.empty:
        gdf['geometry'] = gdf['geometry'].centroid
        center_lon = float(gdf.geometry.x.mean())
        center_lat = float(gdf.geometry.y.mean())
    else:
        center_lon, center_lat = -120.0, 35.0

    covering_poly = _square_polygon(center_lon, center_lat, half_size=0.1)
    geoid = '12345678901'  # Dummy 11-digit GEOID
    feature = _feature_for_polygon(geoid, covering_poly)

    mocker.patch.object(
        CensusTractScraper, '_fetch_features', return_value=[feature]
    )

    scraper = CensusTractScraper()
    _ = scraper.get_census_tracts(inv)

    # Assert every asset received the mocked GEOID
    for asset_id in inv.get_asset_ids():
        found, features = inv.get_asset_features(asset_id)
        assert found is True
        assert features.get(TRACT_KEY) == geoid


# -------------------------------------------------
# get_census_geographies: given polygons and mocked TIGERweb block groups
# -------------------------------------------------

WEST = '060010001001'
EAST = '060010001002'


def block_groups(
    polygons: dict[str, Any], crs: str | None = 'EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Build a census polygon frame from GEOIDs and polygons.

    Args:
        polygons (dict[str, Any]):
            GEOID to Shapely polygon in EPSG:4326.
        crs (str | None, optional):
            The CRS to declare. Defaults to EPSG:4326.

    Returns:
        geopandas.GeoDataFrame:
            One row per polygon with a ``GEOID`` column.
    """
    return gpd.GeoDataFrame(
        {'GEOID': list(polygons)}, geometry=list(polygons.values()), crs=crs
    )


def two_halves() -> dict[str, Polygon]:
    """
    Return two block groups sharing the meridian -122.0.

    Returns:
        dict[str, shapely.geometry.Polygon]:
            West of the meridian :data:`WEST`, east of it :data:`EAST`.
    """
    return {
        EAST: box(-122.0, 37.99, -121.99, 38.01),
        WEST: box(-122.01, 37.99, -122.0, 38.01),
    }


def square(lon: float, lat: float, half: float = 0.0001) -> list[list[float]]:
    """
    Return a closed square ring centered on a point.

    Args:
        lon (float):
            Center longitude.
        lat (float):
            Center latitude.
        half (float, optional):
            Half the side in degrees. Defaults to 0.0001.

    Returns:
        list[list[float]]:
            The ring in BRAILS coordinates.
    """
    return [list(point) for point in _square_polygon(lon, lat, half).exterior.coords]


def inventory(records: dict[str, list]) -> AssetInventory:
    """
    Build an inventory from ids and coordinates.

    Args:
        records (dict[str, list]):
            Asset id to coordinates.

    Returns:
        AssetInventory:
            The inventory, each asset with one feature ``name``.
    """
    inv = AssetInventory()
    for asset_id, coordinates in records.items():
        inv.add_asset(asset_id, Asset(asset_id, coordinates, {'name': asset_id}))
    return inv


def assigned(inv: AssetInventory, key: str = BLOCK_GROUP_KEY) -> dict[str, str]:
    """
    Return each asset's GEOID under one key.

    Args:
        inv (AssetInventory):
            An inventory after :meth:`CensusTractScraper.get_census_geographies`.
        key (str, optional):
            The feature read. Defaults to the block-group key.

    Returns:
        dict[str, str]:
            Asset id to GEOID.
    """
    return {
        asset_id: inv.get_asset_features(asset_id)[1][key]
        for asset_id in inv.get_asset_ids()
    }


class TestContainment:
    """A point takes the GEOID of the polygon holding it, the lowest on a shared edge."""

    def test_inside_one_polygon(self) -> None:
        """A centroid inside one polygon takes its GEOID and its tract."""
        inv = inventory({'w': square(-122.005, 38.0), 'e': square(-121.995, 38.0)})
        result, table = CensusTractScraper().get_census_geographies(
            inv, polygons=block_groups(two_halves())
        )
        assert assigned(result) == {'w': WEST, 'e': EAST}
        assert assigned(result, TRACT_KEY) == {'w': WEST[:11], 'e': EAST[:11]}
        assert table.empty

    def test_shared_edge_takes_lowest_geoid(self) -> None:
        """A centroid on the shared edge takes the lowest GEOID and is listed."""
        inv = inventory({'edge': square(-122.0, 38.0)})
        result, table = CensusTractScraper().get_census_geographies(
            inv, polygons=block_groups(two_halves())
        )
        assert assigned(result) == {'edge': min(WEST, EAST)}
        assert table.to_dict('records') == [
            {'asset_id': 'edge', 'geoid': min(WEST, EAST), 'polygons': 2}
        ]

    def test_point_asset_uses_the_point(self) -> None:
        """A point asset is located by the point itself."""
        inv = inventory({'p': [[-121.995, 38.0]]})
        result, _ = CensusTractScraper().get_census_geographies(
            inv, polygons=block_groups(two_halves())
        )
        assert assigned(result) == {'p': EAST}

    def test_levels_select_the_columns(self) -> None:
        """Each level writes its own key and no other."""
        inv = inventory({'w': square(-122.005, 38.0)})
        result, _ = CensusTractScraper().get_census_geographies(
            inv,
            levels=(LEVEL_TRACT,),
            polygons=block_groups(two_halves()),
            tract_key='tract',
        )
        assert result.inventory['w'].features == {'name': 'w', 'tract': WEST[:11]}

    def test_tract_polygons_give_tracts(self) -> None:
        """Polygons with tract GEOIDs are read as tracts."""
        inv = inventory({'w': square(-122.005, 38.0)})
        tracts = block_groups({WEST[:11]: two_halves()[WEST]})
        result, _ = CensusTractScraper().get_census_geographies(
            inv, levels=(LEVEL_TRACT,), polygons=tracts
        )
        assert assigned(result, TRACT_KEY) == {'w': WEST[:11]}


class TestOutside:
    """A point in no polygon raises and names the assets."""

    def test_outside_raises_and_names_the_assets(self) -> None:
        """Centroids outside every polygon raise with their ids and a count."""
        records = {f'out{index:02d}': square(-121.985, 38.0) for index in range(12)}
        records['w'] = square(-122.005, 38.0)
        with pytest.raises(
            ValueError,
            match=(
                r'12 asset\(s\) lie outside every block group polygon: '
                r'out00, out01, .*, out09 and 2 more\.'
            ),
        ):
            CensusTractScraper().get_census_geographies(
                inventory(records), polygons=block_groups(two_halves())
            )


class TestInputs:
    """Inputs are checked, reprojected, and not mutated."""

    @pytest.mark.parametrize(
        ('polygons', 'match'),
        [
            ({'0600100010': two_halves()[WEST]}, '11 digits'),
            ({WEST: two_halves()[WEST], EAST[:11]: two_halves()[EAST]}, 'mixes'),
        ],
    )
    def test_geoid_lengths_are_checked(
        self, polygons: dict[str, Polygon], match: str
    ) -> None:
        """A GEOID of neither length, or of both lengths at once, raises."""
        with pytest.raises(ValueError, match=match):
            CensusTractScraper().get_census_geographies(
                inventory({'w': square(-122.005, 38.0)}),
                polygons=block_groups(polygons),
            )

    def test_numeric_geoid_raises(self) -> None:
        """A numeric GEOID column raises."""
        frame = block_groups(two_halves())
        frame['GEOID'] = frame['GEOID'].astype('int64')
        with pytest.raises(ValueError, match='numeric'):
            CensusTractScraper().get_census_geographies(
                inventory({'w': square(-122.005, 38.0)}), polygons=frame
            )

    def test_missing_crs_raises(self) -> None:
        """Polygons without a CRS raise."""
        with pytest.raises(ValueError, match='no CRS'):
            CensusTractScraper().get_census_geographies(
                inventory({'w': square(-122.005, 38.0)}),
                polygons=block_groups(two_halves(), crs=None),
            )

    def test_tracts_cannot_give_block_groups(self) -> None:
        """Tract polygons with the block-group level raise."""
        tracts = block_groups({WEST[:11]: two_halves()[WEST]})
        with pytest.raises(ValueError, match='are tracts'):
            CensusTractScraper().get_census_geographies(
                inventory({'w': square(-122.005, 38.0)}), polygons=tracts
            )

    @pytest.mark.parametrize('levels', [(), ('county',), (LEVEL_TRACT, LEVEL_TRACT)])
    def test_levels_are_checked(self, levels: tuple[str, ...]) -> None:
        """Levels outside the two, none, or a repeated level raise."""
        with pytest.raises(ValueError, match='levels'):
            CensusTractScraper().get_census_geographies(
                inventory({'w': square(-122.005, 38.0)}), levels=levels
            )

    def test_projected_polygons_give_the_same_result(self) -> None:
        """Polygons in a projected CRS are reprojected to the same result."""
        inv = inventory(
            {
                'w': square(-122.005, 38.0),
                'e': square(-121.995, 38.0),
                'edge': square(-122.0, 38.0),
            }
        )
        scraper = CensusTractScraper()
        expected, expected_table = scraper.get_census_geographies(
            inv, polygons=block_groups(two_halves())
        )
        projected = block_groups(two_halves()).to_crs('EPSG:32610')
        result, table = scraper.get_census_geographies(inv, polygons=projected)
        assert assigned(result) == assigned(expected)
        assert table['geoid'].tolist() == expected_table['geoid'].tolist()

    def test_inputs_not_mutated(self) -> None:
        """Neither the inventory nor the frame changes."""
        inv = inventory({'w': square(-122.005, 38.0)})
        frame = block_groups(two_halves())
        before_features = deepcopy(inv.inventory['w'].features)
        before_frame = frame.copy()
        CensusTractScraper().get_census_geographies(inv, polygons=frame)
        assert inv.inventory['w'].features == before_features
        assert frame.equals(before_frame)


class TestTigerwebBlockGroups:
    """Without polygons, block groups come from TIGERweb's block-group layer."""

    def test_block_groups_and_tracts_from_one_layer(
        self, mocker: MockerFixture
    ) -> None:
        """One query per block group serves its assets, and the tract is its prefix."""
        features = {
            geoid: _feature_for_polygon(geoid, polygon)
            for geoid, polygon in two_halves().items()
        }
        mock_fetch = mocker.patch.object(
            CensusTractScraper,
            '_fetch_features',
            side_effect=[[features[EAST]], [features[WEST]]],
        )
        inv = inventory(
            {
                'a': square(-121.995, 38.0),
                'b': square(-121.996, 38.001),
                'c': square(-122.005, 38.0),
            }
        )
        result, table = CensusTractScraper().get_census_geographies(inv)

        assert mock_fetch.call_count == 2
        assert {call.kwargs['url'] for call in mock_fetch.call_args_list} == {
            CENSUS_BLOCK_GROUP_QUERY_URL
        }
        assert assigned(result) == {'a': EAST, 'b': EAST, 'c': WEST}
        assert assigned(result, TRACT_KEY) == {
            'a': EAST[:11],
            'b': EAST[:11],
            'c': WEST[:11],
        }
        assert table.empty
        assert BLOCK_GROUP_KEY not in inv.inventory['a'].features

    def test_tract_level_queries_the_tract_layer(
        self, mocker: MockerFixture
    ) -> None:
        """With the tract level alone, the tract layer is queried."""
        tract = _feature_for_polygon(EAST[:11], two_halves()[EAST])
        mock_fetch = mocker.patch.object(
            CensusTractScraper, '_fetch_features', return_value=[tract]
        )
        result, _ = CensusTractScraper().get_census_geographies(
            inventory({'a': square(-121.995, 38.0)}), levels=(LEVEL_TRACT,)
        )
        assert mock_fetch.call_args.kwargs['url'] == CENSUS_TRACT_QUERY_URL
        assert result.inventory['a'].features == {'name': 'a', TRACT_KEY: EAST[:11]}

    @pytest.mark.parametrize(
        ('inside', 'queries'), [('a_inside', 2), ('z_inside', 1)]
    )
    def test_shared_edge_does_not_depend_on_order(
        self, mocker: MockerFixture, inside: str, queries: int
    ) -> None:
        """
        An asset on a shared edge takes the lowest GEOID in either order.

        Queried first, the edge asset fetches both block groups and settles
        the interior asset; queried after it, it fetches both itself.
        """
        features = {
            geoid: _feature_for_polygon(geoid, polygon)
            for geoid, polygon in two_halves().items()
        }
        mock_fetch = mocker.patch.object(
            CensusTractScraper,
            '_fetch_features',
            side_effect=lambda **kwargs: (
                [features[EAST], features[WEST]]
                if kwargs['lon'] == pytest.approx(-122.0)
                else [features[EAST]]
            ),
        )
        inv = inventory(
            {inside: square(-121.995, 38.0), 'm_edge': square(-122.0, 38.0)}
        )
        result, table = CensusTractScraper().get_census_geographies(inv)

        assert assigned(result) == {inside: EAST, 'm_edge': min(WEST, EAST)}
        assert table.to_dict('records') == [
            {'asset_id': 'm_edge', 'geoid': min(WEST, EAST), 'polygons': 2}
        ]
        assert mock_fetch.call_count == queries

    def test_get_census_tracts_returns_only_tracts_with_an_asset(
        self, mocker: MockerFixture
    ) -> None:
        """A tract fetched only because an edge asset touches it is not returned."""
        halves = two_halves()
        low, high = '06001000100', '06001000200'
        features = {
            low: _feature_for_polygon(low, halves[WEST]),
            high: _feature_for_polygon(high, halves[EAST]),
        }
        mocker.patch.object(
            CensusTractScraper,
            '_fetch_features',
            side_effect=lambda **kwargs: (
                [features[high], features[low]]
                if kwargs['lon'] == pytest.approx(-122.0)
                else [features[low]]
            ),
        )
        inv = inventory({'a': square(-122.005, 38.0), 'edge': square(-122.0, 38.0)})
        tracts = CensusTractScraper().get_census_tracts(inv)

        assert list(tracts) == [low]
        assert tracts[low].equals(halves[WEST])
        assert assigned(inv, TRACT_KEY) == {'a': low, 'edge': low}

    def test_outside_the_layer_raises(self, mocker: MockerFixture) -> None:
        """A point TIGERweb places in no block group raises with its id."""
        mocker.patch.object(CensusTractScraper, '_fetch_features', return_value=[])
        with pytest.raises(ValueError, match=r'1 asset\(s\) .*: far\.'):
            CensusTractScraper().get_census_geographies(
                inventory({'far': square(10.0, 50.0)})
            )
