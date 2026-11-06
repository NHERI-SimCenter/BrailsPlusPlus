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


"""Fixture builders shared by the observed-donor imputer tests."""

from __future__ import annotations

import math
from typing import Any

from brails.imputers import ObservedDonorImputer
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    ROUND_DONOR,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.asset_id import EARTH_RADIUS_M

LON0 = -122.0
LAT0 = 38.0
METERS_PER_DEGREE = math.pi * EARTH_RADIUS_M / 180.0

BG_A = '060010001001'
BG_B = '060010001002'

SEED = 20261102

# The k of the fixture imputer, under which the golden values were computed.
FIXTURE_K = 5
OTHER_SEED = 7


def lonlat(east: float, north: float) -> tuple[float, float]:
    """
    Return the point a given distance east and north of the fixture origin.

    Args:
        east (float):
            Meters east.
        north (float):
            Meters north.

    Returns:
        tuple[float, float]:
            Longitude and latitude.
    """
    return (
        LON0 + east / (METERS_PER_DEGREE * math.cos(math.radians(LAT0))),
        LAT0 + north / METERS_PER_DEGREE,
    )


def square(east: float, north: float, side: float = 10.0) -> list[list[float]]:
    """
    Return a closed square footprint centered east and north of the origin.

    Args:
        east (float):
            Meters east of the origin.
        north (float):
            Meters north of the origin.
        side (float, optional):
            Side in meters. Defaults to 10.

    Returns:
        list[list[float]]:
            The ring in BRAILS coordinates.
    """
    lon, lat = lonlat(east, north)
    dlat = side / 2.0 / METERS_PER_DEGREE
    dlon = dlat / math.cos(math.radians(lat))
    return [
        [lon - dlon, lat - dlat],
        [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat],
        [lon - dlon, lat + dlat],
        [lon - dlon, lat - dlat],
    ]


def asset(
    east: float, north: float, side: float = 10.0, **features: Any
) -> tuple[list[list[float]], dict[str, Any]]:
    """
    Return the coordinates and features of one fixture asset.

    Args:
        east (float):
            Meters east of the origin.
        north (float):
            Meters north of the origin.
        side (float, optional):
            Side in meters. Defaults to 10.
        **features (Any):
            The asset's features.

    Returns:
        tuple[list[list[float]], dict[str, Any]]:
            Coordinates and features.
    """
    return square(east, north, side), dict(features)


def inventory(
    records: dict[str, tuple[list, dict[str, Any]]],
    order: list[str] | None = None,
) -> AssetInventory:
    """
    Build an inventory from fixture assets.

    Args:
        records (dict[str, tuple[list, dict[str, Any]]]):
            Asset id to coordinates and features.
        order (list[str], optional):
            The insertion order. Defaults to the order of ``records``.

    Returns:
        AssetInventory:
            The inventory.
    """
    inv = AssetInventory()
    for asset_id in order or list(records):
        coordinates, features = records[asset_id]
        inv.add_asset(asset_id, Asset(asset_id, coordinates, dict(features)))
    return inv


def imputer(**kwargs: Any) -> ObservedDonorImputer:
    """
    Build an imputer on the fixture seed and k.

    Args:
        **kwargs (Any):
            Overrides of the constructor arguments.

    Returns:
        ObservedDonorImputer:
            The imputer.
    """
    options = {'seed': SEED, 'k': FIXTURE_K}
    options.update(kwargs)
    return ObservedDonorImputer(**options)


def donors_from_log(result: Any) -> dict[tuple[str, int], str]:
    """
    Return the donor of each target and world from the stage log.

    Args:
        result (Any):
            A stage result.

    Returns:
        dict[tuple[str, int], str]:
            ``(target, world)`` to donor id, for the donor round.
    """
    log = result.log
    rows = log[log['round'] == ROUND_DONOR]
    return {
        (row.asset_id, row.world): row.donor_id
        for row in rows.itertuples(index=False)
    }


def world_value(value: Any, world: int) -> Any:
    """
    Return the value of a stored feature in one world.

    Args:
        value (Any):
            A scalar or a list of realizations.
        world (int):
            The world.

    Returns:
        Any:
            The realization.
    """
    return value[world] if isinstance(value, list) else value


def features_of(inventory: AssetInventory) -> dict[Any, dict[str, Any]]:
    """
    Return the features of every asset of an inventory.

    Args:
        inventory (AssetInventory):
            The inventory.

    Returns:
        dict[Any, dict[str, Any]]:
            A shallow copy of each asset's features, keyed by asset id.
    """
    return {
        asset_id: dict(asset.features)
        for asset_id, asset in inventory.inventory.items()
    }
