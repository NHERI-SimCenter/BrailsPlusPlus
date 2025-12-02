#
# Copyright (c) 2024 The Regents of the University of California
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

"""
The class in this module allocates point-based features to polygon-based assets.

.. autosummary::

    BasicPointsToPolygonsAllocator
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from brails.types import AssetInventory


class BasicPointsToPolygonsAllocator:
    """
    Allocates point-based features to polygon assets using spatial matching.

    The allocator prepares lean GeoDataFrames from input inventories, computes
    injective point-to-polygon matches (strict containment with an optional
    buffered fallback), and transfers point features to matched polygons via the
    public `AssetInventory` API.
    """

    def __init__(
        self, polygon_inventory: AssetInventory, point_inventory: AssetInventory
    ) -> None:
        """
        Initialize the allocator with polygon and point inventories.

        Args:
            polygon_inventory (AssetInventory): Inventory containing polygon assets
                that will receive features.
            point_inventory (AssetInventory): Inventory containing point assets
                whose features will be allocated to polygons.

        Raises:
            TypeError: If either input is not an instance of `AssetInventory`.
            ValueError: If either inventory is empty.
        """
        if not isinstance(polygon_inventory, AssetInventory):
            raise TypeError(
                'polygon_inventory must be an instance of AssetInventory'
            )
        if not isinstance(point_inventory, AssetInventory):
            raise TypeError('point_inventory must be an instance of AssetInventory')

        # Fail-fast: disallow empty inventories
        if not polygon_inventory.inventory:
            raise ValueError(
                'polygon_inventory is empty; allocator requires non-empty polygon data'
            )
        if not point_inventory.inventory:
            raise ValueError(
                'point_inventory is empty; allocator requires non-empty point data'
            )

        self.polygon_inventory = polygon_inventory
        self.point_inventory = point_inventory

    def _inventory_to_gdf(
        self,
        inventory: AssetInventory,
        expected_geom_types: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Convert an inventory to a lean GeoDataFrame containing only `asset_id` and `geometry`.

        When `expected_geom_types` is provided, rows whose geometry type does not
        match any of the expected types are dropped with a warning.

        Args:
            inventory (AssetInventory): Source inventory to convert.
            expected_geom_types (list[str] | None): Optional list of allowed
                geometry type names (e.g., `['Polygon', 'MultiPolygon']`). If
                provided, geometries not in this list are filtered out.

        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame in `EPSG:4326` with columns
                `asset_id` and `geometry`.

        Raises:
            KeyError: If an `asset_id` cannot be derived from an `id` column or
                the index.
        """
        geojson: dict[str, Any] = inventory.get_geojson()
        features: list[dict[str, Any]] = geojson.get('features', [])

        gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')

        if 'id' in gdf.columns:
            gdf = gdf.rename(columns={'id': 'asset_id'})
        elif gdf.index.name == 'id' or not isinstance(gdf.index, pd.RangeIndex):
            # Treat index as the source of IDs
            index_col_name = (
                gdf.index.name if gdf.index.name is not None else 'index'
            )
            gdf = gdf.reset_index().rename(columns={index_col_name: 'asset_id'})
        else:
            raise KeyError(
                "Could not locate asset IDs in GeoJSON: expected an 'id' column or index to contain IDs."
            )

        # Keep only asset_id and geometry to avoid carrying heavy payloads
        lean_gdf = gdf[['asset_id', 'geometry']].copy()

        # Apply optional geometry type filtering
        if expected_geom_types is not None:
            # GeoPandas provides .geom_type (preferred) and .type (also works)
            geom_types = getattr(
                lean_gdf.geometry, 'geom_type', lean_gdf.geometry.type
            )
            invalid_mask = ~geom_types.isin(expected_geom_types)
            dropped = int(invalid_mask.sum())
            if dropped > 0:
                print(
                    f'Warning: {dropped} assets with invalid geometry types (expected {expected_geom_types}) were ignored.'
                )
                lean_gdf = lean_gdf.loc[~invalid_mask]

        return lean_gdf

    def _compute_matches(
        self,
        use_convex_hull: bool = False,  # noqa: FBT001, FBT002
        buffer_dist: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute injective polygon-to-point matches using spatial joins.

        Optionally applies convex hulls to polygons before matching and performs
        a second pass using buffered polygons (meters) to capture near misses.

        Args:
            use_convex_hull (bool): If True, replace polygon geometries with
                their convex hulls prior to matching.
            buffer_dist (float): Buffer distance in meters for a second-pass
                match of previously unmatched polygons against unused points.

        Returns:
            pandas.DataFrame: A tidy DataFrame with columns `polygon_id` and
                `point_id` representing 1-to-1 matches.

        Raises:
            KeyError: If the right-index column produced by `geopandas.sjoin`
                cannot be determined.
        """
        poly_gdf = self._inventory_to_gdf(
            self.polygon_inventory, expected_geom_types=['Polygon', 'MultiPolygon']
        ).set_index('asset_id')
        point_gdf = self._inventory_to_gdf(
            self.point_inventory, expected_geom_types=['Point', 'MultiPoint']
        ).set_index('asset_id')

        if use_convex_hull:
            poly_gdf = poly_gdf.copy()
            poly_gdf.geometry = poly_gdf.geometry.convex_hull

        pass1 = gpd.sjoin(poly_gdf, point_gdf, how='inner', predicate='contains')

        # Determine the right-side id column name produced by sjoin
        if 'index_right' in pass1.columns:
            right_join_col = 'index_right'
        elif 'asset_id_right' in pass1.columns:
            right_join_col = 'asset_id_right'
        else:
            raise KeyError(
                'Could not determine right index column in sjoin output; looked for '
                "'index_right' or 'asset_id_right'. Found columns: "
                + str(list(pass1.columns))
            )

        matches = pass1[[right_join_col]].copy()

        # Conflict Resolution
        # Rule 1: A polygon accepts max 1 point (keep first occurrence)
        matches = matches[~matches.index.duplicated(keep='first')]
        # Rule 2: A point belongs to max 1 polygon (keep first)
        matches = matches.drop_duplicates(subset=right_join_col, keep='first')

        if buffer_dist and buffer_dist > 0:
            unmatched_polys = poly_gdf.loc[~poly_gdf.index.isin(matches.index)]
            unused_points = point_gdf.loc[
                ~point_gdf.index.isin(matches[right_join_col])
            ]

            if not unmatched_polys.empty and not unused_points.empty:
                # Buffer unmatched polygons in meters: project to EPSG:3857, buffer, back to 4326
                buffered = unmatched_polys.to_crs(3857).copy()
                buffered['geometry'] = buffered.geometry.buffer(buffer_dist)
                buffered = buffered.to_crs(4326)

                pass2 = gpd.sjoin(
                    buffered, unused_points, how='inner', predicate='contains'
                )
                # Determine right join col for pass2 as well
                if 'index_right' in pass2.columns:
                    pass2_right_join_col = 'index_right'
                elif 'asset_id_right' in pass2.columns:
                    pass2_right_join_col = 'asset_id_right'
                else:
                    raise KeyError(
                        'Could not determine right index column in pass2 sjoin output.'
                    )
                extra = pass2[[pass2_right_join_col]].copy()
                # Apply Rule 1 and Rule 2 again on the new matches
                extra = extra[~extra.index.duplicated(keep='first')]
                extra = extra.drop_duplicates(
                    subset=pass2_right_join_col, keep='first'
                )

                if not extra.empty:
                    # Normalize the right join column name for concatenation
                    extra = extra.rename(
                        columns={pass2_right_join_col: right_join_col}
                    )
                    matches = pd.concat([matches, extra], axis=0)

        idx_name = matches.index.name if matches.index.name is not None else 'index'
        out = matches.reset_index().rename(
            columns={idx_name: 'polygon_id', right_join_col: 'point_id'}
        )
        return out[['polygon_id', 'point_id']]

    def allocate(
        self,
        overwrite: bool = False,  # noqa: FBT001, FBT002
        use_convex_hull: bool = False,  # noqa: FBT001, FBT002
        buffer_dist: float = 0.0,
    ) -> None:
        """
        Transfer features from matched points to polygons.

        Matches are computed internally and point features are written to the
        corresponding polygon assets using the public `AssetInventory` API.

        Args:
            overwrite (bool): If True, existing polygon features are allowed to be
                overwritten by point features. Defaults to False.
            use_convex_hull (bool): If True, compute matches on polygon convex
                hulls.
            buffer_dist (float): Optional buffer distance in meters to enable a
                second-pass match for near points.

        Returns:
            None: This method modifies the polygon inventory in place.

        Raises:
            KeyError: If a matched point cannot be found in the point inventory.
        """
        matches_df = self._compute_matches(
            use_convex_hull=use_convex_hull, buffer_dist=buffer_dist
        )

        transfers = 0
        if not isinstance(matches_df, pd.DataFrame) or matches_df.empty:
            print('No matching points were found for any polygons.')
            return

        for _, row in matches_df.iterrows():
            polygon_id = row.get('polygon_id', None)
            point_id = row.get('point_id', None)

            found, features = self.point_inventory.get_asset_features(point_id)
            if not found:
                raise KeyError(
                    f'Point {point_id} found in match but missing from inventory'
                )

            # Copy features and remove IDs to avoid conflicting with the polygon's ID
            features_to_transfer = features.copy()
            features_to_transfer.pop('id', None)

            success = self.polygon_inventory.add_asset_features(
                polygon_id, features_to_transfer, overwrite=overwrite
            )
            if success:
                transfers += 1

        print(f'Allocated point features to {transfers} polygons.')
