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
import numpy as np
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
        a second pass using buffered polygons to capture near misses.

        Matching is one point per polygon and one polygon per point. Candidates are
        ranked before they are taken, so the outcome does not depend on the order the
        spatial join returned: containment candidates are ranked by polygon area and
        then by the point's distance from the polygon centroid, and near-miss
        candidates by the point's true distance to the footprint.

        The second pass considers only polygons left unmatched by the first, so
        raising `buffer_dist` cannot recover additional points inside a polygon that
        already holds one.

        Args:
            use_convex_hull (bool): If True, replace polygon geometries with
                their convex hulls prior to matching.
            buffer_dist (float): Buffer distance in meters for a second-pass
                match of previously unmatched polygons against unused points.
                Measured in a projected CRS estimated from the polygons, so the
                distance is meters on the ground.

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

        # Projected CRS for areas, distances, and buffering. Estimated from the
        # polygons so distances are true meters at the data's own location.
        metric_crs = poly_gdf.estimate_utm_crs()
        poly_metric = poly_gdf.to_crs(metric_crs)
        point_metric = point_gdf.to_crs(metric_crs)

        pass1 = gpd.sjoin(poly_gdf, point_gdf, how='inner', predicate='contains')
        right_join_col = self._right_id_column(pass1)

        candidates = pass1[[right_join_col]].reset_index()
        candidates.columns = ['polygon_id', 'point_id']

        # Rank containment candidates by how specific the containing polygon is, then
        # by how central the point sits in it. A point inside overlapping footprints
        # belongs to the smallest one that holds it, and a polygon holding several
        # points keeps the most central. Ids break remaining ties so the result does
        # not depend on the order sjoin returned.
        candidates['rank_area'] = poly_metric.geometry.area.reindex(
            candidates['polygon_id']
        ).to_numpy()
        candidates['rank_distance'] = self._pairwise_distance(
            poly_metric.geometry.centroid,
            point_metric.geometry,
            candidates['polygon_id'],
            candidates['point_id'],
        )
        candidates = candidates.sort_values(
            ['rank_area', 'rank_distance', 'polygon_id', 'point_id'],
            kind='mergesort',
        )
        matches = self._assign_injectively(candidates)

        if buffer_dist and buffer_dist > 0:
            matched_polygons = set(matches['polygon_id'])
            matched_points = set(matches['point_id'])
            unmatched_polys = poly_metric.loc[
                ~poly_metric.index.isin(matched_polygons)
            ]
            unused_points = point_gdf.loc[~point_gdf.index.isin(matched_points)]

            if not unmatched_polys.empty and not unused_points.empty:
                buffered = unmatched_polys.copy()
                buffered['geometry'] = buffered.geometry.buffer(buffer_dist)
                buffered = buffered.to_crs(4326)

                pass2 = gpd.sjoin(
                    buffered, unused_points, how='inner', predicate='contains'
                )
                pass2_right_join_col = self._right_id_column(pass2, stage='pass2')

                extra = pass2[[pass2_right_join_col]].reset_index()
                extra.columns = ['polygon_id', 'point_id']

                # Near misses are ranked by true distance from the point to the
                # unbuffered footprint, so the closest claim wins.
                extra['rank_distance'] = self._pairwise_distance(
                    poly_metric.geometry,
                    point_metric.geometry,
                    extra['polygon_id'],
                    extra['point_id'],
                )
                extra = extra.sort_values(
                    ['rank_distance', 'polygon_id', 'point_id'], kind='mergesort'
                )
                extra = self._assign_injectively(
                    extra, used_polygons=matched_polygons, used_points=matched_points
                )

                if not extra.empty:
                    matches = pd.concat([matches, extra], axis=0, ignore_index=True)

        return matches[['polygon_id', 'point_id']].reset_index(drop=True)

    @staticmethod
    def _right_id_column(joined: pd.DataFrame, stage: str = 'sjoin') -> str:
        """
        Find the column holding right-side ids in a spatial join result.

        Args:
            joined (pandas.DataFrame): The spatial join output.
            stage (str): Matching stage, named in the error message.

        Returns:
            str: The column name.

        Raises:
            KeyError: If neither expected column is present.
        """
        if 'index_right' in joined.columns:
            return 'index_right'
        if 'asset_id_right' in joined.columns:
            return 'asset_id_right'
        if stage == 'pass2':
            raise KeyError(
                'Could not determine right index column in pass2 sjoin output.'
            )
        raise KeyError(
            'Could not determine right index column in sjoin output; looked for '
            "'index_right' or 'asset_id_right'. Found columns: "
            + str(list(joined.columns))
        )

    @staticmethod
    def _pairwise_distance(
        polygon_geometry: gpd.GeoSeries,
        point_geometry: gpd.GeoSeries,
        polygon_ids: pd.Series,
        point_ids: pd.Series,
    ) -> np.ndarray:
        """
        Measure the distance of each candidate pair, in the given CRS's units.

        Args:
            polygon_geometry (geopandas.GeoSeries): Polygon geometries by id.
            point_geometry (geopandas.GeoSeries): Point geometries by id.
            polygon_ids (pandas.Series): Polygon id of each pair.
            point_ids (pandas.Series): Point id of each pair.

        Returns:
            numpy.ndarray: Distance for each pair.
        """
        if len(polygon_ids) == 0:
            return np.empty(0)

        crs = polygon_geometry.crs
        left = gpd.GeoSeries(
            polygon_geometry.reindex(polygon_ids).to_numpy(), crs=crs
        )
        right = gpd.GeoSeries(point_geometry.reindex(point_ids).to_numpy(), crs=crs)
        return left.distance(right, align=False).to_numpy()

    @staticmethod
    def _assign_injectively(
        candidates: pd.DataFrame,
        used_polygons: set | None = None,
        used_points: set | None = None,
    ) -> pd.DataFrame:
        """
        Take candidate pairs in order, keeping the first claim on each side.

        Args:
            candidates (pandas.DataFrame): Ranked pairs, best first, with
                `polygon_id` and `point_id` columns.
            used_polygons (set | None): Polygons already spoken for.
            used_points (set | None): Points already spoken for.

        Returns:
            pandas.DataFrame: The accepted pairs, with `polygon_id` and `point_id`
                columns.
        """
        taken_polygons = set() if used_polygons is None else set(used_polygons)
        taken_points = set() if used_points is None else set(used_points)

        accepted = []
        for polygon_id, point_id in zip(
            candidates['polygon_id'], candidates['point_id']
        ):
            if polygon_id in taken_polygons or point_id in taken_points:
                continue
            taken_polygons.add(polygon_id)
            taken_points.add(point_id)
            accepted.append((polygon_id, point_id))

        return pd.DataFrame(accepted, columns=['polygon_id', 'point_id'])

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

        Notes:
            A polygon receives at most one point. Where a footprint covers several
            structures, as with attached houses dissolved into a single outline or a
            group of buildings mapped as one polygon, only the highest ranked point
            is transferred. With `buffer_dist` set, the points it outranked stay
            available to the second pass, so each may be placed on a neighboring
            footprint that holds no point of its own; the rest are left out of the
            inventory altogether. Splitting such footprints before allocation is the
            way to keep those records with their own structures when using this
            allocator.
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
