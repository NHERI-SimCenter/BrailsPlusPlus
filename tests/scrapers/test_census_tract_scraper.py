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

"""Tests for CensusScraper in brails.scrapers.us_census_scrapers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
from requests import Response, exceptions
from shapely.geometry import Polygon, mapping

from brails.scrapers.us_census_scrapers.census_tract_scraper import (
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
# Unit tests for _fetch_tract_geometry
# ---------------------


def test_fetch_success_contract(
    mocker: MockerFixture, census_api_response: dict[str, Any]
) -> None:
    """Verify method parses a known, canned API response correctly.

    Arrange: mock requests.get to return the loaded fixture JSON.
    Act: call _fetch_tract_geometry with dummy coords.
    Assert: first feature dict is returned and requests.get was called once.
    """
    scraper = CensusTractScraper()

    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = census_api_response

    mock_get = mocker.patch('requests.get', return_value=mock_response)

    feature = scraper._fetch_tract_geometry(
        lon=-122.2585, lat=37.8719, retries=3, timeout=5, delay=0
    )

    assert isinstance(feature, dict)
    assert 'type' in feature
    assert 'geometry' in feature
    assert 'properties' in feature
    assert mock_get.call_count == 1


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

    feature = scraper._fetch_tract_geometry(
        lon=-120.0, lat=35.0, retries=3, timeout=5, delay=0
    )

    assert isinstance(feature, dict)
    assert mock_get.call_count == 2


@pytest.mark.parametrize(
    ('side_effect', 'expected_exception'),
    [
        (_make_http_error(400), exceptions.HTTPError),  # 4xx should fail fast
        (None, ValueError),  # successful 2xx but empty features
    ],
)
def test_fetch_fails_fast_on_permanent_errors(
    mocker: MockerFixture,
    side_effect: Any,
    expected_exception: type[BaseException],
) -> None:
    """Ensure method fails immediately on unrecoverable errors.

    Cases:
    - 4xx HTTPError raised on first call -> should propagate immediately.
    - 2xx response with empty features -> should raise ValueError without retry.
    """
    scraper = CensusTractScraper()

    if side_effect is not None:
        # Simulate 4xx client error
        mock_get = mocker.patch('requests.get', side_effect=side_effect)
    else:
        # Simulate 2xx with empty features list using a Mock response
        mock_response = mocker.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'type': 'FeatureCollection',
            'features': [],
        }

        mock_get = mocker.patch('requests.get', return_value=mock_response)

    # Avoid real sleep even though we do not expect retries
    mocker.patch(
        'brails.scrapers.us_census_scrapers.census_tract_scraper.time.sleep',
        return_value=None,
    )

    with pytest.raises(expected_exception):
        scraper._fetch_tract_geometry(
            lon=-77.0, lat=38.9, retries=3, timeout=5, delay=0
        )

    # Only one attempt should have been made
    assert mock_get.call_count == 1


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
        scraper._fetch_tract_geometry(
            lon=-100.0, lat=40.0, retries=3, timeout=5, delay=0
        )

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
    1. _fetch_tract_geometry is called once per unique tract (2 calls).
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
        '_fetch_tract_geometry',
        side_effect=[feature_t1, feature_t2],
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
        assert features.get('TRACT_GEOID') == geoid

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
        '_fetch_tract_geometry',
        return_value=feature,
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
        assert features.get('TRACT_GEOID') == '06001400100'


def test_get_tracts_handles_api_failure_gracefully(
    mocker: MockerFixture,
    single_tract_inventory: AssetInventory,
    capsys: Any,
) -> None:
    """If fetching fails permanently, method should halt gracefully without modification.

    Also verify that the new failure-path messages are printed to stdout.
    """
    mocker.patch.object(
        CensusTractScraper,
        '_fetch_tract_geometry',
        side_effect=ValueError('no features for point'),
    )

    scraper = CensusTractScraper()

    # Snapshot original features (should be empty dicts) to compare after call
    before = {
        aid: single_tract_inventory.get_asset_features(aid)[1].copy()
        for aid in single_tract_inventory.get_asset_ids()
    }

    # The method is expected to complete without raising
    try:
        _ = scraper.get_census_tracts(single_tract_inventory)
    except Exception as exc:  # noqa: BLE001 - test must capture unexpected crash
        pytest.fail(f'get_census_tracts raised unexpectedly: {exc}')

    # Verify no mutation of inventory features
    after = {
        aid: single_tract_inventory.get_asset_features(aid)[1].copy()
        for aid in single_tract_inventory.get_asset_ids()
    }
    assert after == before

    # Verify failure-path stdout contains the critical markers
    out = capsys.readouterr().out
    assert '--- CRITICAL ERROR ---' in out
    assert '--- Job Halted ---' in out
    assert 'Total points successfully processed: 0' in out


def test_get_tracts_with_empty_inventory(
    mocker: MockerFixture, empty_inventory: AssetInventory
) -> None:
    """Empty inventory should result in zero API calls and an empty cache."""
    mock_fetch = mocker.patch.object(
        CensusTractScraper, '_fetch_tract_geometry', autospec=True
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
    geoid = features.get('TRACT_GEOID')
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

    Mock _fetch_tract_geometry to return a tract polygon that covers the centroids
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
        CensusTractScraper, '_fetch_tract_geometry', return_value=feature
    )

    scraper = CensusTractScraper()
    _ = scraper.get_census_tracts(inv)

    # Assert every asset received the mocked GEOID
    for asset_id in inv.get_asset_ids():
        found, features = inv.get_asset_features(asset_id)
        assert found is True
        assert features.get('TRACT_GEOID') == geoid
