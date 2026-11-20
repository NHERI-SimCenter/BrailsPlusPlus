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
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 08-01-2025

"""
This module defines abstract FootprintScraper class.

.. autosummary::

    FootprintScraper
"""

import json
import math
import numbers
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from numpy import append, arctan2, cos, deg2rad, diff, pi, sin, sqrt
from shapely.geometry import Polygon

from brails.constants import FOOTPRINT_PROVIDER_KEY, PROVIDER_ID_FEATURES
from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.region_boundary import RegionBoundary
from brails.utils import UnitConverter
from brails.utils.asset_id import assign_asset_ids
from brails.utils.unit_converter import UNIT_TO_METER

# Global constants for plan area calculations and asset typing:
EARTH_RADIUS_FT = 20925721.784777  # Earth's radius in feet (WGS-84)
ASSET_TYPE = 'building'

# Asset id schemes, described in FootprintScraper.__init__.
ID_SCHEME_INDEX = 'index'
ID_SCHEME_GEOHASH = 'geohash'
ID_SCHEMES = (ID_SCHEME_INDEX, ID_SCHEME_GEOHASH)

# Default minimum plan area under the 'geohash' scheme, in square feet.
DEFAULT_MIN_AREA_SQFT = 450.0

# The default floor in square meters, the unit the plan area is measured in.
DEFAULT_MIN_AREA_M2 = DEFAULT_MIN_AREA_SQFT * UNIT_TO_METER['ft'] ** 2

# Features recorded in the scraper's length unit or its square. They stay out of a
# tiebreak built from a footprint's features, so that ranks do not depend on the
# unit.
LENGTH_UNIT_FEATURES = ('buildingheight', 'footprintArea')


class FootprintScraper(ABC):
    """
    Abstract base class for getting building footprints within a region.

    This class defines the interface for any footprint scraper implementation.

    Attributes:
        name (str):
            Name or identifier for the scraper.
        footprints (List):
            A list to store retrieved footprint geometries.
        centroids (List):
            A list to store centroid coordinates of the footprints.
        id_scheme (str):
            How assets are keyed: ``'index'`` or ``'geohash'``.
        min_area (float):
            The smallest plan area kept, in the square of the scraper's length unit.
        asset_id_log (list[dict]):
            One entry per member of a geohash collision set in the last
            ``get_footprints`` call, with ``position`` the footprint's position in
            that call's footprints.
        asset_id_skipped (list[dict]):
            The footprints the last ``get_footprints`` call dropped before keys were
            assigned, by ``position``, ``provider_id``, ``plan_area`` (in the square
            of the length unit) and ``reason`` (``'below_min_area'`` or
            ``'zero_area'``). Under ``'geohash'``, or with ``min_area`` above 0, a
            footprint without area is dropped as ``'zero_area'`` and a vertex outside
            EPSG:4326 raises ``ValueError``.

    Methods:
        get_footprints(region):
            Abstract method to return building footprints for a given region.
    """

    def __init__(
        self,
        name: str,
        input_dict: Optional[Dict[str, Any]] = None,
        length_unit: str = 'ft'
    ) -> None:
        """
        Initialize the footprint scraper.

        Args:
            name (str):
                Name or identifier for the scraper.
            input_dict (dict, optional):
                Scraper options. The base class reads:

                - ``'id_scheme'`` (str): ``'index'``, the default, keys assets by
                  position. ``'geohash'`` keys them by the geohash of their centroid,
                  see :func:`brails.utils.asset_id.assign_asset_ids`.
                - ``'min_area'`` (float): the smallest plan area kept, in the square
                  of ``length_unit``, measured by
                  :meth:`AssetInventory.centroid_and_plan_area`.
            length_unit (str):
                The scraper's length unit, which sets the unit of ``min_area``.
                Defaults to ``'ft'``.

        Raises:
            ValueError:
                If ``id_scheme`` is not a known scheme or ``min_area`` is not a
                finite, non-negative number.
        """
        self.name: str = name
        self.footprints = []
        self.centroids = []
        self._configure_asset_ids(input_dict or {}, length_unit)

    def _configure_asset_ids(
        self,
        input_dict: Dict[str, Any],
        length_unit: str
    ) -> None:
        """
        Read the id scheme and the area floor from the scraper's options.

        Args:
            input_dict (dict):
                Scraper options, read for ``'id_scheme'`` and ``'min_area'``.
            length_unit (str):
                The scraper's length unit.

        Raises:
            ValueError:
                If ``id_scheme`` is not a known scheme or ``min_area`` is not a
                finite, non-negative number.
        """
        id_scheme = input_dict.get('id_scheme', ID_SCHEME_INDEX)
        if id_scheme not in ID_SCHEMES:
            raise ValueError(
                f"id_scheme must be one of {ID_SCHEMES}, not {id_scheme!r}."
            )
        self.id_scheme = id_scheme

        min_area = input_dict.get('min_area')
        if min_area is None:
            self._min_area_m2 = self._default_min_area_m2(id_scheme)
            min_area = 0.0
            if self._min_area_m2 > 0:
                min_area = DEFAULT_MIN_AREA_SQFT * (
                    UNIT_TO_METER['ft'] ** 2
                    / self._square_meters_per_unit(length_unit)
                )
        elif (
            isinstance(min_area, bool)
            or not isinstance(min_area, numbers.Real)
            or not math.isfinite(min_area)
            or min_area < 0
        ):
            raise ValueError(
                'min_area must be a finite, non-negative number, not '
                f'{min_area!r}.'
            )
        else:
            self._min_area_m2 = float(min_area) * self._square_meters_per_unit(
                length_unit
            )
        self.min_area = float(min_area)
        self.asset_id_log = []
        self.asset_id_skipped = []

    @staticmethod
    def _default_min_area_m2(id_scheme: str) -> float:
        """
        Return the default area floor of an id scheme in square meters.

        Args:
            id_scheme (str):
                ``'index'`` or ``'geohash'``.

        Returns:
            float:
                ``DEFAULT_MIN_AREA_M2`` under ``'geohash'``, and 0 under ``'index'``.
        """
        if id_scheme == ID_SCHEME_GEOHASH:
            return DEFAULT_MIN_AREA_M2
        return 0.0

    @staticmethod
    def _square_meters_per_unit(length_unit: str) -> float:
        """
        Return the square meters in one square of a length unit, unrounded.

        Args:
            length_unit (str):
                A length unit :class:`~brails.utils.UnitConverter` accepts.

        Returns:
            float:
                The square of the unit's length in meters.

        Raises:
            ValueError:
                If the length unit is not supported.
        """
        unit = UnitConverter.normalize_unit(length_unit)
        if unit not in UNIT_TO_METER:
            raise ValueError(f'Unsupported length unit {length_unit!r}.')
        return UNIT_TO_METER[unit] ** 2

    @abstractmethod
    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve building footprints and attributes within a specified region.

        This method must be implemented by subclasses to provide access to
        building-level footprint and attribute data for a given geographic
        boundary. It is typically used by downstream components to extract
        spatial information for analysis or visualization.

        Args:
            region (RegionBoundary):
                The geographic region of interest.

        Returns:
            AssetInventory:
                An inventory containing the footprints of buildings within the
                specified region.
        """
        pass

    def _polygon_area(
        self,
        lats: Union[List[float], Tuple[float, ...]],
        lons: Union[List[float], Tuple[float, ...]],
        length_unit: str = 'ft'
    ) -> float:
        """
        Calculate the approximate area of a polygon on the Earth's surface.

        The area is computed using Green's Theorem applied to a spherical
        surface, which approximates the Earth as a perfect sphere.

        Args:
            lats (list or tuple of float):
                Latitudes of the polygon vertices in degrees.
            lons (list or tuple of float):
                Longitudes of the polygon vertices in degrees.
            length_unit (str):
                Desired output length unit to use for area calculation
                (e.g., 'ft', 'm'). Defaults to 'ft'.

        Returns:
            float:
                Area of the polygon in the requested unit.
        """
        lats = deg2rad(lats)
        lons = deg2rad(lons)

        # Close the polygon if not already closed
        if lats[0] != lats[-1] or lons[0] != lons[-1]:
            lats = append(lats, lats[0])
            lons = append(lons, lons[0])

        # Compute spherical coordinates relative to origin (0,0):
        a = sin(lats / 2) ** 2 + cos(lats) * sin(lons / 2) ** 2
        colat = 2 * arctan2(sqrt(a), sqrt(1 - a))

        # Azimuths relative to (0,0):
        az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2 * pi)

        # Angle differences, adjusted to [-pi, pi]:
        daz = diff(az)
        daz = (daz + pi) % (2 * pi) - pi

        # Midpoint colatitudes:
        colat_mid = colat[:-1] + diff(colat) / 2

        # Green's theorem integral:
        integrands = (1 - cos(colat_mid)) * daz
        spherical_area_ratio = abs(sum(integrands)) / (4 * pi)
        spherical_area_ratio = min(
            spherical_area_ratio, 1 - spherical_area_ratio
        )

        # Convert to area in square feet:
        area_sqft = spherical_area_ratio * 4 * pi * EARTH_RADIUS_FT**2

        # Convert to requested unit:
        return UnitConverter.convert_unit(area_sqft, 'ft2', f'{length_unit}2')


    def _assign_footprint_keys(
        self,
        footprints: Sequence[list],
        length_unit: str,
        tiebreak: Optional[Sequence[Optional[str]]] = None,
        provider_ids: Optional[Sequence[Any]] = None,
        positions: Optional[Sequence[int]] = None,
    ) -> Tuple[List[int], List[Union[int, str]]]:
        """
        Apply the area floor and choose the key of each footprint kept.

        Args:
            footprints (Sequence[list]):
                Footprints as BRAILS coordinate lists.
            length_unit (str):
                Length unit whose square ``min_area`` is expressed in.
            tiebreak (Sequence[str | None], optional):
                One string per footprint ranking footprints equal in normalized
                geometry under ``'geohash'``.
            provider_ids (Sequence, optional):
                The provider's id of each footprint, recorded for dropped footprints.
            positions (Sequence[int], optional):
                The position of each footprint in the scraper's footprints, used in
                the logs. Defaults to ``range(len(footprints))``.

        Returns:
            tuple[list[int], list[int | str]]:
                The indices into ``footprints`` that are kept, in order, and the key
                of each.
        """
        # A subclass that skips FootprintScraper.__init__ gets the defaults.
        id_scheme = getattr(self, 'id_scheme', ID_SCHEME_INDEX)
        min_area_m2 = getattr(self, '_min_area_m2', None)
        if min_area_m2 is None:
            min_area = getattr(self, 'min_area', None)
            min_area_m2 = (
                self._default_min_area_m2(id_scheme) if min_area is None
                else min_area * self._square_meters_per_unit(length_unit)
            )
        if positions is None:
            positions = list(range(len(footprints)))
        area_unit = f'{length_unit}2'
        screen = id_scheme == ID_SCHEME_GEOHASH or min_area_m2 > 0
        # Without a screen nothing is dropped and the unit is not needed.
        square_meters = (
            self._square_meters_per_unit(length_unit) if screen else 1.0
        )

        skipped = []
        kept = []
        for index, footprint in enumerate(footprints):
            if screen:
                reason, area_m2 = None, 0.0
                if not AssetInventory.has_area(footprint):
                    reason = 'zero_area'
                elif min_area_m2 > 0:
                    area_m2 = AssetInventory.centroid_and_plan_area(footprint)[2]
                    if area_m2 < min_area_m2:
                        reason = 'below_min_area'
                if reason is not None:
                    skipped.append({
                        'position': positions[index],
                        'provider_id': (
                            None if provider_ids is None
                            else provider_ids[index]
                        ),
                        'plan_area': area_m2 / square_meters,
                        'reason': reason,
                    })
                    continue
            kept.append(index)

        log = []
        if id_scheme == ID_SCHEME_GEOHASH:
            result = assign_asset_ids(
                [footprints[index] for index in kept],
                None if tiebreak is None else [tiebreak[i] for i in kept],
            )
            keys: List[Union[int, str]] = list(result.ids)
            for entry in result.log:
                log.append({**entry, 'position': positions[kept[entry['position']]]})
        else:
            keys = list(range(len(kept)))

        self.asset_id_log = log
        self.asset_id_skipped = skipped

        below = sum(1 for entry in skipped if entry['reason'] == 'below_min_area')
        zero = len(skipped) - below
        collision_sets = len({entry['geohash'] for entry in log})
        if below or zero or collision_sets:
            print(
                f'Asset keys: dropped {below} footprints under the minimum '
                f'plan area of {min_area_m2 / square_meters:g} {area_unit} '
                f'and {zero} footprints without area. {collision_sets} '
                f'geohash collision sets hold {len(log)} footprints.'
            )
        return kept, keys

    @staticmethod
    def _footprint_tiebreak(
        features: Dict[str, Any],
        id_feature: Optional[str],
    ) -> str:
        """
        Return the string that ranks a footprint among equal geometries.

        Args:
            features (dict):
                The footprint's features.
            id_feature (str or None):
                The feature holding the provider's id, if the provider has one.

        Returns:
            str:
                The provider's id when the footprint carries one. Otherwise the
                sorted JSON of the features outside ``LENGTH_UNIT_FEATURES``.
        """
        value = None if id_feature is None else features.get(id_feature)
        if value is not None and value != 'NA':
            return str(value)
        return json.dumps(
            {
                key: feature
                for key, feature in features.items()
                if key not in LENGTH_UNIT_FEATURES
            },
            sort_keys=True,
            default=str,
        )

    def _build_keyed_inventory(
        self,
        footprints: Sequence[list],
        features: List[Dict[str, Any]],
        length_unit: str,
        provider: Optional[str] = None,
        positions: Optional[Sequence[int]] = None,
        centroids: Optional[Sequence[Any]] = None,
        inventory: Optional[AssetInventory] = None,
    ) -> Tuple[AssetInventory, List[int]]:
        """
        Key a scraper's footprints and add the kept ones to an inventory.

        Args:
            footprints (Sequence[list]):
                Footprints as BRAILS coordinate lists.
            features (list[dict]):
                The features of each footprint. ``footprint_provider`` is added to
                each when ``provider`` is given.
            length_unit (str):
                Length unit whose square ``min_area`` is expressed in.
            provider (str, optional):
                Name of the footprint provider, a key of ``PROVIDER_ID_FEATURES``.
            positions (Sequence[int], optional):
                The position of each footprint in the scraper's footprints, used in
                the logs.
            centroids (Sequence, optional):
                The centroid of each footprint, stored for the kept ones in
                ``self.centroids``.
            inventory (AssetInventory, optional):
                The inventory the assets are added to. Defaults to a new one.

        Returns:
            tuple[AssetInventory, list[int]]:
                The inventory and the indices into ``footprints`` that are kept, in
                order.
        """
        if provider is not None:
            for asset_features in features:
                asset_features[FOOTPRINT_PROVIDER_KEY] = provider
        id_features = PROVIDER_ID_FEATURES.get(provider, ())
        id_feature = id_features[0] if id_features else None

        kept, keys = self._assign_footprint_keys(
            footprints,
            length_unit,
            tiebreak=[
                self._footprint_tiebreak(asset_features, id_feature)
                for asset_features in features
            ],
            provider_ids=None if id_feature is None else [
                asset_features.get(id_feature) for asset_features in features
            ],
            positions=positions,
        )
        self.footprints = [footprints[index] for index in kept]
        self.centroids = (
            [] if centroids is None else [centroids[index] for index in kept]
        )

        if inventory is None:
            inventory = AssetInventory()
        for key, index in zip(keys, kept):
            asset = Asset(key, footprints[index], features[index])
            inventory.add_asset(key, asset)
        return inventory, kept

    def _create_asset_inventory(
        self,
        footprints: list,
        attributes: dict,
        length_unit: str,
        provider: Optional[str] = None
    ) -> AssetInventory:
        """
        Construct AssetInventory object from footprints and attributes.

        This method is intended to be used by subclasses to assemble an
        AssetInventory from raw geometric and attribute data. Footprints are
        screened and keyed by :meth:`_assign_footprint_keys`.

        Args:
            footprints (list):
                A list of building footprints. Each footprint is a list of
                [lon, lat] coordinate pairs defining the polygon.
            attributes (dict):
                A dictionary of attribute lists, where each key corresponds to
                an attribute name (e.g., "height", "material") and each value
                is a list with one entry per building. The lists are reduced
                in place to the footprints kept.
            length_unit (str):
                Unit of length to use for derived measurements
                (e.g., 'ft', 'm').
            provider (str, optional):
                Name of the footprint provider, written as the feature
                ``footprint_provider`` on every asset when given.

        Returns:
            AssetInventory:
                An inventory containing building assets for the specified
                region.
        """
        # TODO: Need to calculate footprint areas after removing bad polygons.
        # Compute footprint areas:
        attributes['footprintArea'] = []
        for footprint in footprints:
            lons, lats = zip(*footprint)
            area = int(self._polygon_area(list(lats), list(lons), length_unit))
            attributes['footprintArea'].append(area)

        # Calculate centroids of the footprints and remove footprint data that
        # does not form a polygon:
        valid_footprints = []
        centroids = []
        valid_indices = []

        for index, footprint in enumerate(footprints):
            try:
                polygon = Polygon(footprint)
                centroid = polygon.centroid
                valid_footprints.append(footprint)
                centroids.append(centroid)
                valid_indices.append(index)
            except Exception as e:
                print(
                    f'[Warning] Invalid footprint at index {index}, '
                    f'skipping. Error: {e}'
                )
                continue

        # Filter attributes to keep only valid entries:
        for key in attributes:
            attributes[key] = [attributes[key][i] for i in valid_indices]

        # Assemble the features of each footprint:
        features = []
        for index in range(len(valid_footprints)):
            asset_features = {'type': ASSET_TYPE}
            for key, values in attributes.items():
                value = values[index]
                asset_features[key] = "NA" if value is None else value
            # TODO: Need remove NA's from the inventory. # Note: This may
            # affect imputers. Need to verify compatibility before proceeding.
            features.append(asset_features)

        # Key the footprints kept and assemble the AssetInventory:
        inventory, kept = self._build_keyed_inventory(
            valid_footprints,
            features,
            length_unit,
            provider=provider,
            positions=valid_indices,
            centroids=centroids,
        )
        for key in attributes:
            attributes[key] = [attributes[key][i] for i in kept]

        return inventory
