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

"""Tests for BasicPointsToPolygonsAllocator in brails.aggregators.points_to_polygons.basic."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Callable

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from brails.aggregators.points_to_polygons.basic.basic_points_to_polygons import (
    BasicPointsToPolygonsAllocator,
)
from brails.types import Asset, AssetInventory

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def test_assets() -> dict[str, Asset]:
    """Provide a dictionary of reusable Asset objects for testing."""
    assets = {}

    # --- Polygons ---
    # poly_A: 10x10 square at origin
    assets['poly_A'] = Asset(
        'poly_A',
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
        features={'id': 'poly_A'},
    )

    # poly_U: Concave U-shaped polygon
    assets['poly_U'] = Asset(
        'poly_U',
        [
            [20.0, 0.0],
            [30.0, 0.0],
            [30.0, 10.0],
            [20.0, 10.0],
            [20.0, 8.0],
            [22.0, 8.0],
            [22.0, 2.0],
            [20.0, 2.0],
            [20.0, 0.0],
        ],
        features={'id': 'poly_U'},
    )

    # Overlapping squares for exclusivity tests
    assets['poly_overlap_1'] = Asset(
        'poly_1',
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
        features={'id': 'poly_1'},
    )
    assets['poly_overlap_2'] = Asset(
        'poly_2',
        [[4.0, 4.0], [12.0, 4.0], [12.0, 12.0], [4.0, 12.0], [4.0, 4.0]],
        features={'id': 'poly_2'},
    )

    # --- Points ---
    # Inside poly_A
    assets['pt_in_A'] = Asset(
        'pt_in_A', [[5.0, 5.0]], features={'ffe': 6.8, 'id': 'pt_in_A', 'src': 'A'}
    )

    # In courtyard of poly_U (outside strict, inside hull)
    assets['pt_in_U_courtyard'] = Asset(
        'pt_in_U_courtyard',
        [[21.0, 5.0]],
        features={'tag': 'courtyard', 'id': 'pt_in_U_courtyard'},
    )

    # Near poly_A (approx 111m away at 10.001)
    assets['pt_near_A'] = Asset(
        'pt_near_A', [[10.001, 5.0]], features={'near': True, 'id': 'pt_near_A'}
    )

    # Duplicate inside poly_A
    assets['pt_duplicate'] = Asset(
        'pt_duplicate', [[5.0, 6.0]], features={'dup': True, 'id': 'pt_duplicate'}
    )

    # Overlapping point (in both poly_overlap_1 and 2)
    assets['pt_overlap'] = Asset(
        'pt_overlap', [[5.0, 5.0]], features={'foo': 'bar', 'id': 'pt_overlap'}
    )

    return assets


@pytest.fixture
def make_inventory() -> Callable[[list[Asset]], AssetInventory]:
    """Factory fixture to create an AssetInventory from a list of Assets."""

    def _make(assets: list[Asset]) -> AssetInventory:
        inv = AssetInventory()
        for asset in assets:
            # Deepcopy to ensure test isolation
            asset_copy = deepcopy(asset)
            # Retrieve ID from features (workaround for missing public asset_id)
            asset_id = asset_copy.features['id']
            inv.add_asset(asset_id, asset_copy)
        return inv

    return _make


# -----------------------------------------------------------------------------
# Initialization tests
# -----------------------------------------------------------------------------


def test_init_valid(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """Allocator initializes with valid non-empty inventories."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    assert allocator.polygon_inventory is poly_inv
    assert allocator.point_inventory is point_inv


def test_init_invalid_types() -> None:
    """Passing invalid types should raise TypeError."""
    with pytest.raises(TypeError):
        BasicPointsToPolygonsAllocator('not-inv', AssetInventory())  # type: ignore
    with pytest.raises(TypeError):
        BasicPointsToPolygonsAllocator(AssetInventory(), None)  # type: ignore


def test_init_empty(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """Empty inventories should raise ValueError (fail-fast)."""
    valid_inv = make_inventory([test_assets['poly_A']])
    empty_inv = AssetInventory()

    with pytest.raises(ValueError, match='polygon_inventory is empty'):
        BasicPointsToPolygonsAllocator(empty_inv, valid_inv)
    with pytest.raises(ValueError, match='point_inventory is empty'):
        BasicPointsToPolygonsAllocator(valid_inv, empty_inv)


# -----------------------------------------------------------------------------
# Data conversion tests
# -----------------------------------------------------------------------------


def test_inventory_to_gdf(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """_inventory_to_gdf returns lean GeoDataFrame with correct schema."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)
    gdf = allocator._inventory_to_gdf(poly_inv)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert list(gdf.columns) == ['asset_id', 'geometry']
    assert gdf.crs.to_string() == 'EPSG:4326'
    assert 'poly_A' in gdf['asset_id'].to_numpy()


# -----------------------------------------------------------------------------
# Spatial logic tests
# -----------------------------------------------------------------------------


def test_strict_match(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """pt_in_A strictly contained in poly_A should match in strict pass."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)
    matches = allocator._compute_matches()

    assert len(matches) == 1
    assert matches.iloc[0]['polygon_id'] == 'poly_A'
    assert matches.iloc[0]['point_id'] == 'pt_in_A'


def test_convex_hull(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """pt_in_U_courtyard matches poly_U only when use_convex_hull=True."""
    poly_inv = make_inventory([test_assets['poly_U']])
    point_inv = make_inventory([test_assets['pt_in_U_courtyard']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    # No match without hull
    assert allocator._compute_matches(use_convex_hull=False).empty

    # Match with hull
    matches = allocator._compute_matches(use_convex_hull=True)
    assert len(matches) == 1
    assert matches.iloc[0]['polygon_id'] == 'poly_U'


def test_buffer_match(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """pt_near_A should match poly_A only when buffer_dist is sufficient."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_near_A']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    # No match with 0 buffer
    assert allocator._compute_matches(buffer_dist=0.0).empty

    # Match with 200m buffer (point is ~111m away)
    matches = allocator._compute_matches(buffer_dist=200.0)
    assert len(matches) == 1
    assert matches.iloc[0]['polygon_id'] == 'poly_A'


def test_capacity_rule(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """Only one of multiple interior points should be matched (capacity=1)."""
    poly_inv = make_inventory([test_assets['poly_A']])
    # Two points inside poly_A
    point_inv = make_inventory([test_assets['pt_in_A'], test_assets['pt_duplicate']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)
    matches = allocator._compute_matches()

    # Expect exactly 1 match for poly_A
    assert len(matches) == 1
    assert matches.iloc[0]['polygon_id'] == 'poly_A'
    # Logic is 'first', order depends on iteration but only one should exist
    assert matches.iloc[0]['point_id'] in ['pt_in_A', 'pt_duplicate']


def test_exclusivity_rule(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """A point overlapping two polygons should be assigned to only one."""
    poly_inv = make_inventory(
        [test_assets['poly_overlap_1'], test_assets['poly_overlap_2']]
    )
    point_inv = make_inventory([test_assets['pt_overlap']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)
    matches = allocator._compute_matches()

    # Expect exactly 1 match total (point used once)
    assert len(matches) == 1
    assert matches.iloc[0]['polygon_id'] in ['poly_1', 'poly_2']
    assert matches.iloc[0]['point_id'] == 'pt_overlap'


# -----------------------------------------------------------------------------
# Feature transfer tests
# -----------------------------------------------------------------------------


def test_feature_transfer(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """Allocate should transfer features correctly."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)
    allocator.allocate()

    found, feats = poly_inv.get_asset_features('poly_A')
    assert found
    assert feats['ffe'] == 6.8
    assert feats['src'] == 'A'


def test_overwrite_protection(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """overwrite=False should preserve existing features."""
    # Poly with existing 'ffe'
    poly_A = deepcopy(test_assets['poly_A'])  # noqa: N806
    poly_A.features = {'ffe': 1.0, 'original': True, 'id': 'poly_A'}

    # Point with new 'ffe'
    pt_in_A = deepcopy(test_assets['pt_in_A'])  # has ffe=6.8  # noqa: N806

    poly_inv = make_inventory([poly_A])
    point_inv = make_inventory([pt_in_A])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)
    allocator.allocate(overwrite=False)

    found, feats = poly_inv.get_asset_features('poly_A')
    assert feats['ffe'] == 1.0  # preserved
    assert feats['original'] is True
    assert feats['src'] == 'A'  # new key added


def test_id_sanitization(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """Point's 'id' should not overwrite polygon features."""
    poly_inv = make_inventory([test_assets['poly_A']])

    # Point has 'id' in features
    pt_in_A = deepcopy(test_assets['pt_in_A'])  # noqa: N806
    pt_in_A.features['id'] = 'BAD_ID'

    point_inv = make_inventory([pt_in_A])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)
    allocator.allocate()

    found, feats = poly_inv.get_asset_features('poly_A')
    assert feats['id'] == 'poly_A'  # Should remain original
    assert 'BAD_ID' not in str(feats)


# -----------------------------------------------------------------------------
# Mocked edge-path tests
# -----------------------------------------------------------------------------


def test_inventory_to_gdf_id_in_index(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
    mocker: MockerFixture,
) -> None:
    """_inventory_to_gdf should use index named 'id' as asset_id when no 'id' column exists."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])
    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    # Create a GDF with no 'id' column, but with index name 'id'
    gdf = gpd.GeoDataFrame(
        {'geometry': [Point(0.0, 0.0)]}, geometry='geometry', crs='EPSG:4326'
    )
    gdf.index = pd.Index(['my-id-123'], name='id')

    patcher = mocker.patch('geopandas.GeoDataFrame.from_features', return_value=gdf)

    out = allocator._inventory_to_gdf(poly_inv)

    # Ensure patch was used
    patcher.assert_called()

    assert list(out.columns) == ['asset_id', 'geometry']
    assert out.loc[0, 'asset_id'] == 'my-id-123'


def test_inventory_to_gdf_missing_id(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
    mocker: MockerFixture,
) -> None:
    """_inventory_to_gdf should raise KeyError when neither 'id' column nor index carries IDs."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])
    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    # GDF with no 'id' column and default RangeIndex (no name)
    gdf = gpd.GeoDataFrame(
        {'geometry': [Point(0.0, 0.0)]}, geometry='geometry', crs='EPSG:4326'
    )
    # Ensure default RangeIndex
    assert isinstance(gdf.index, pd.RangeIndex)
    assert gdf.index.name is None

    mocker.patch('geopandas.GeoDataFrame.from_features', return_value=gdf)

    with pytest.raises(KeyError, match='Could not locate asset IDs in GeoJSON'):
        _ = allocator._inventory_to_gdf(poly_inv)


def test_compute_matches_sjoin_alt_column(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
    mocker: MockerFixture,
) -> None:
    """_compute_matches should accept sjoin output with 'asset_id_right' instead of 'index_right'."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])
    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    # Build a minimal sjoin-like DataFrame where right column is 'asset_id_right'
    pass1 = pd.DataFrame({'asset_id_right': ['pt_in_A']})
    pass1.index = pd.Index(
        ['poly_A'], name='asset_id'
    )  # emulate left index as polygon ids

    mocker.patch('geopandas.sjoin', return_value=pass1)

    matches = allocator._compute_matches(use_convex_hull=False, buffer_dist=0.0)

    assert isinstance(matches, pd.DataFrame)
    assert len(matches) == 1
    assert set(matches.columns) == {'polygon_id', 'point_id'}
    assert matches.iloc[0]['polygon_id'] == 'poly_A'
    assert matches.iloc[0]['point_id'] == 'pt_in_A'


def test_compute_matches_sjoin_missing_column(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
    mocker: MockerFixture,
) -> None:
    """_compute_matches should raise KeyError when sjoin output lacks expected right index columns."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])
    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    # sjoin returns a frame without 'index_right' nor 'asset_id_right'
    bad = pd.DataFrame({'something_else': ['pt_in_A']})
    bad.index = pd.Index(['poly_A'], name='asset_id')

    mocker.patch('geopandas.sjoin', return_value=bad)

    with pytest.raises(KeyError, match='Could not determine right index column'):
        _ = allocator._compute_matches()


def test_allocate_empty_matches(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
    mocker: MockerFixture,
) -> None:
    """Allocate should not call add_asset_features when _compute_matches returns empty."""
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    # Force empty matches
    mocker.patch.object(
        BasicPointsToPolygonsAllocator,
        '_compute_matches',
        return_value=pd.DataFrame(columns=['polygon_id', 'point_id']),
    )

    # Spy on add_asset_features
    spy_add = mocker.spy(poly_inv, 'add_asset_features')

    allocator.allocate()

    spy_add.assert_not_called()


def test_allocate_raises_on_missing_point(
    mocker: MockerFixture,
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
) -> None:
    """Allocate should raise KeyError when a matched point is missing from inventory."""
    # Setup valid inventories (Poly A, Point in A)
    poly_inv = make_inventory([test_assets['poly_A']])
    point_inv = make_inventory([test_assets['pt_in_A']])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    # Force point lookup to report missing for the matched point
    mocker.patch.object(
        allocator.point_inventory,
        'get_asset_features',
        return_value=(False, {}),
    )

    # The spatial match should exist; allocate should now fail fast
    with pytest.raises(
        KeyError, match='Point pt_in_A found in match but missing from inventory'
    ):
        allocator.allocate()


def test_filtering_invalid_geoms(
    test_assets: dict[str, Asset],
    make_inventory: Callable[[list[Asset]], AssetInventory],
    capsys: pytest.CaptureFixture,
) -> None:
    """_inventory_to_gdf should filter invalid geometry types with a warning.

    Build a point inventory that includes one valid Point (inside poly_A)
    and one invalid Polygon geometry. The allocator should warn once and
    only the valid point should be considered for matching.
    """
    # Polygon inventory with poly_A
    poly_inv = make_inventory([test_assets['poly_A']])

    # Valid point in A
    valid_pt = deepcopy(test_assets['pt_in_A'])

    # Invalid: create an "asset" that has Polygon coordinates but place in point inventory
    invalid_polygon_as_point = Asset(
        'poly_like_point',
        [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]],
        features={'id': 'poly_like_point'},
    )

    point_inv = make_inventory([valid_pt, invalid_polygon_as_point])

    allocator = BasicPointsToPolygonsAllocator(poly_inv, point_inv)

    matches = allocator._compute_matches()

    # Expect exactly one match for the valid point in poly_A
    assert isinstance(matches, pd.DataFrame)
    assert len(matches) == 1
    assert matches.iloc[0]['polygon_id'] == 'poly_A'
    assert matches.iloc[0]['point_id'] == 'pt_in_A'

    # Verify warning was printed
    captured = capsys.readouterr()
    assert (
        "Warning: 1 assets with invalid geometry types (expected ['Point', 'MultiPoint']) were ignored."
        in captured.out
    )
