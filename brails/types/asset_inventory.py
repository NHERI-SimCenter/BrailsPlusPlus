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
#
# Contributors:
# Barbaros Cetiner
# Frank McKenna
#
# Last updated:
# 10-22-2025

"""
This module defines asset inventories, their assets and the source string reader.

.. autosummary::

    AssetInventory
    Asset
    read_source_string
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from copy import copy, deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from shapely import box
from shapely.geometry import GeometryCollection, mapping
from shapely.geometry.base import BaseGeometry  # noqa: TC002

from brails.constants import (
    DESIGN_LEVEL_KEY,
    HEIGHT_CLASS_KEY,
    OCCUPANCY_KEY,
    REPLACEMENT_COST_KEY,
    SOURCE_STRING_KEY,
    STORIES_KEY,
    STRUCTURE_TYPE_KEY,
    UNITS_KEY,
    YEAR_BUILT_KEY,
)
from brails.exceptions import BrailsError
from brails.utils.asset_id import _centroid_and_plan_area, _has_area
from brails.utils.clean_floats import clean_floats
from brails.types.housing_unit_inventory import HousingUnitInventory
from brails.utils.geo_tools import GeoTools
from brails.utils.input_validator import InputValidator
from brails.utils.spatial_join_methods.base import SpatialJoinMethods

# Features whose values are vectors. A plain list is one value of such
# a feature, not one realization per possible world.
DEFAULT_VECTOR_FEATURES: tuple[str, ...] = ('HousingUnits',)

# Upper bound on the number of problems named in an error message.
MAX_REPORTED_WORLD_PROBLEMS = 10

# Source labels that AssetInventory.merge accepts when it writes a source
# string, and the code each label contributes.
SOURCE_CODES: Mapping[str, str] = MappingProxyType(
    {
        'local': 'L',
        'footprint': 'F',
        'nsi': 'N',
        'derived': 'R',
        'imputed': 'I',
        'drawn': 'D',
    }
)

# The source label of each code of SOURCE_CODES.
SOURCE_LABELS: Mapping[str, str] = MappingProxyType(
    {code: label for label, code in SOURCE_CODES.items()}
)

# Character of the source string for a feature that no source holds.
SOURCE_CODE_NONE = '-'

# Characters a source string may hold.
SOURCE_STRING_CODES = frozenset({*SOURCE_CODES.values(), SOURCE_CODE_NONE})

# Features covered by the source string by default, one character each,
# in the order a Hazus earthquake inventory is built.
HAZUS_EQ_SOURCE_FEATURES: tuple[str, ...] = (
    OCCUPANCY_KEY,
    UNITS_KEY,
    STORIES_KEY,
    YEAR_BUILT_KEY,
    DESIGN_LEVEL_KEY,
    STRUCTURE_TYPE_KEY,
    HEIGHT_CLASS_KEY,
    REPLACEMENT_COST_KEY,
)

# String BRAILS readers, the footprint scrapers among them, write for an
# absent provider value.
MISSING_VALUE_MARKER = 'NA'

def _is_missing_value(value: Any) -> bool:  # noqa: ANN401
    """
    Check whether AssetInventory.merge treats a feature value as missing.

    Args:
        value (Any):
            The feature value.

    Returns:
        bool:
            ``True`` if the value is missing, ``False`` otherwise.
    """
    if isinstance(value, str):
        return value in (MISSING_VALUE_MARKER, '')
    if isinstance(value, (Sequence, Mapping, AbstractSet)):
        return False
    # pandas.isna returns an array for an array, which is a value.
    return pd.isna(value) is True


# Public name of the missing-value test, for code outside this module.
is_missing_value = _is_missing_value


def read_source_string(
    asset_id: Hashable,
    features: Mapping[str, Any],
    *,
    source_string_key: str = SOURCE_STRING_KEY,
    source_features: Sequence[str] = HAZUS_EQ_SOURCE_FEATURES,
) -> str:
    """
    Return an asset's source string, checked against the features it covers.

    A string passes when it holds one character per source feature, each
    one of :data:`SOURCE_STRING_CODES`, as :meth:`AssetInventory.merge`
    writes it.

    The function reads the source string of one world. An asset of a
    realization that :meth:`AssetInventory.get_world_realization` returns
    holds one string, as does an asset of an inventory of several worlds
    whose strings agree. Where the worlds' strings differ, the inventory
    holds a list of one string per world, and the function refuses it. To
    read an asset's codes in one world, read that world's realization.

    Args:
        asset_id (Hashable):
            The asset, named in the error message.
        features (Mapping[str, Any]):
            The asset's features in one world.
        source_string_key (str, optional):
            The feature holding the source string. Defaults to
            :data:`~brails.constants.SOURCE_STRING_KEY`.
        source_features (Sequence[str], optional):
            The features the string covers, in its order. Defaults to
            :data:`HAZUS_EQ_SOURCE_FEATURES`.

    Returns:
        str:
            The source string, whose character ``i`` codes the source of
            ``source_features[i]``.

    Raises:
        ValueError:
            If the value is not a string (a per-world list included) or is
            missing, its length differs from the number of source features,
            or it holds a character outside :data:`SOURCE_STRING_CODES`.
    """
    codes = features.get(source_string_key)
    if (
        not isinstance(codes, str)
        or len(codes) != len(source_features)
        or not set(codes) <= SOURCE_STRING_CODES
    ):
        msg = (
            f'Asset {asset_id!r} carries no source string of '
            f'{len(source_features)} codes from {sorted(SOURCE_STRING_CODES)} in '
            f'{source_string_key!r}: {codes!r}.'
        )
        raise ValueError(msg)
    return codes


def _is_blank_type(value: Any) -> bool:  # noqa: ANN401
    """Check whether an asset ``type`` is ``None``, a NaN or an empty string."""
    if value is None or value == '':
        return True
    return isinstance(value, float) and math.isnan(value)


def _is_plain_digit_string(text: str) -> bool:
    """
    Check whether ``text`` is ASCII digits without a leading zero, or ``'0'``.

    Args:
        text (str):
            The string form of an asset id.

    Returns:
        bool:
            ``True`` if ``text`` is such a string, ``False`` otherwise.
    """
    return text.isascii() and text.isdigit() and (text == '0' or text[0] != '0')


def _number_features_by_type(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Number features for ``write_to_geojson(integer_ids=True)``, in place.

    Args:
        features (list[dict[str, Any]]):
            Features as built by ``AssetInventory.get_geojson``, each holding
            its asset id in the property ``id``.

    Returns:
        list[dict[str, Any]]:
            The features in the order ``write_to_geojson`` states, each
            carrying its number within its ``type`` as the feature-level
            ``id`` and its asset id, as a string, in the property
            ``unique_id``.

    Raises:
        ValueError:
            If an asset already carries a ``unique_id`` or ``AIM_id``
            feature, has no ``type`` or a ``type`` that is ``None``, NaN or
            an empty string, or if two asset ids have the same string form.
    """
    keyed = []
    asset_ids_by_string = {}
    for feature in features:
        properties = feature['properties']
        asset_id = properties.pop('id')
        unique_id = str(asset_id)
        for reserved in ('unique_id', 'AIM_id'):
            if reserved in properties:
                raise ValueError(
                    f"Asset {asset_id!r} already carries a '{reserved}' "
                    'feature, which the integer-id export reserves.'
                )
        asset_type = properties.get('type')
        if _is_blank_type(asset_type):
            raise ValueError(
                f"Asset {asset_id!r} has no 'type' value, which the "
                'integer-id export requires of every asset. Pass asset_type '
                'to write_to_geojson to give such assets a type.'
            )
        if unique_id in asset_ids_by_string:
            raise ValueError(
                f'Asset ids {asset_ids_by_string[unique_id]!r} and '
                f'{asset_id!r} have the same string form, so the integer-id '
                'export cannot tell them apart.'
            )
        asset_ids_by_string[unique_id] = asset_id
        keyed.append((str(asset_type), asset_id, unique_id, feature))

    # A bool fails the rule, since its string form is 'True' or 'False'.
    numeric = all(_is_plain_digit_string(unique_id) for _, _, unique_id, _ in keyed)
    keyed.sort(key=lambda item: (item[0], int(item[2]) if numeric else item[2]))
    last_numbers: dict[str, int] = {}
    for asset_type, _, unique_id, feature in keyed:
        number = last_numbers.get(asset_type, 0) + 1
        last_numbers[asset_type] = number
        feature['properties']['unique_id'] = unique_id
        feature['id'] = number
    return [feature for _, _, _, feature in keyed]


def _write_geojson_output_file(path: str | Path, geojson: dict[str, Any]) -> None:
    """
    Write a GeoJSON file for ``write_to_geojson``.

    Args:
        path (str or Path):
            The output file.
        geojson (dict[str, Any]):
            The GeoJSON ``FeatureCollection``, passed through
            :func:`clean_floats` before it is written.

    Raises:
        ValueError:
            If ``geojson`` holds an infinite value outside a tuple, in which
            case no file is created.
        OSError:
            If writing or replacing the file fails, in which case the
            temporary file is removed and an existing file is left as it was.
    """
    payload = clean_floats(geojson)
    output_path = Path(path)
    descriptor, partial_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=output_path.name + '.', suffix='.partial'
    )
    partial_path = Path(partial_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        # mkstemp creates the file readable by its owner only. Give it the
        # mode a newly created file would have.
        umask = os.umask(0)
        os.umask(umask)
        partial_path.chmod(0o666 & ~umask)
        partial_path.replace(output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def _merge_id_order(asset_id: Hashable) -> tuple[str, str]:
    """Sort key for asset ids: the id as a string, then its type name."""
    return (str(asset_id), type(asset_id).__name__)


def _check_source_features(source_features: Sequence[str], order: list[str]) -> None:
    """
    Check the features and labels of a source string requested from merge.

    Args:
        source_features (Sequence[str]):
            The features the source string covers, in order.
        order (list[str]):
            The source labels in order of precedence.

    Raises:
        TypeError:
            If ``source_features`` is a single string.
        ValueError:
            If ``source_features`` repeats a feature or names
            :data:`SOURCE_STRING_KEY`, or a label has no code in
            :data:`SOURCE_CODES`.
    """
    if isinstance(source_features, str):
        raise TypeError(
            'source_features must be a sequence of feature names, '
            f'not the string {source_features!r}.'
        )
    repeated = sorted(
        (f for f, count in Counter(source_features).items() if count > 1), key=str
    )
    if repeated:
        raise ValueError(f'source_features repeats {repeated}.')
    if SOURCE_STRING_KEY in source_features:
        raise ValueError(
            f'source_features cannot include {SOURCE_STRING_KEY!r}.'
        )
    uncoded = sorted(set(order) - set(SOURCE_CODES))
    if uncoded:
        raise ValueError(
            f'No source code for the labels {uncoded}. A source '
            f'string needs labels from {sorted(SOURCE_CODES)}.'
        )


def _report_first_problems(problems: list[str], separator: str) -> str:
    """
    Join up to MAX_REPORTED_WORLD_PROBLEMS problems, counting the rest.

    Args:
        problems (list[str]):
            One description per problem.
        separator (str):
            The text placed between two descriptions and before the count.

    Returns:
        str:
            The joined descriptions, followed by ``... and N more`` when
            some were left out.
    """
    text = separator.join(problems[:MAX_REPORTED_WORLD_PROBLEMS])
    left_out = len(problems) - MAX_REPORTED_WORLD_PROBLEMS
    if left_out > 0:
        text += f'{separator}... and {left_out} more'
    return text


def is_probabilistic_feature(
    key: str, value: Any, vector_features: Iterable[str]
) -> bool:
    """
    Check whether a feature value holds multiple possible worlds.

    An ordinary feature holds one value per possible world, so a list
    means the feature is probabilistic. A vector feature is list-valued
    already, so a plain list is a single value and only a list of lists
    means the feature is probabilistic.

    Args:
        key (str):
            The name of the feature.
        value (Any):
            The value of the feature.
        vector_features (Iterable[str]):
            Names of the features whose values are vectors.

    Returns:
        bool:
            ``True`` if the value holds one entry per possible world,
            ``False`` if it is a single value shared by every world.

    Examples:
        >>> is_probabilistic_feature('StructureType', ['C1', 'C2'], [])
        True
        >>> is_probabilistic_feature('StructureType', 'C1', [])
        False
        >>> is_probabilistic_feature('HousingUnits', [1, 2], ['HousingUnits'])
        False
        >>> is_probabilistic_feature(
        ...     'HousingUnits', [[1, 2], [3, 4]], ['HousingUnits']
        ... )
        True
    """
    if not isinstance(value, list):
        return False

    if key in vector_features:
        # These are only probabilistic if they are a list of lists.
        return len(value) > 0 and isinstance(value[0], list)

    return True


def _values_agree(first: object, second: object) -> bool:
    """
    Check whether two feature values are equal by the rule of add_world.

    Args:
        first (object):
            A feature value of one world.
        second (object):
            A feature value of another world, compared with ``first``.

    Returns:
        bool:
            ``True`` if the values are equal.
    """
    if first is second:
        return True
    if isinstance(first, float) and isinstance(second, float):
        return first == second or (math.isnan(first) and math.isnan(second))
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return bool(np.array_equal(first, second))
        except (TypeError, ValueError):
            return False
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


class Asset:
    """A spatial asset with geometry coordinates, and attributes.

    To import the :class:`Asset` class, use:

    .. code-block:: python

        from brails.types.asset_inventory import Asset


    Attributes:
        asset_id (str or int):
            Unique identifier for the asset.
        coordinates (list[list[float]] | list[list[list[float]]]):
            The geometry of the asset. Supports:
            - ``[[x, y]]`` (Point)
            - ``[[x, y], [x, y], ...]`` (LineString, Polygon)
            - ``[[[x, y], ...], ...]`` (MultiLineString, MultiPolygon)
        features (dict[str, Any], optional):
            Additional attributes/features of the asset. Defaults to ``None``.
    """

    def __init__(
        self,
        asset_id: str | int,
        coordinates: list[list[float]],
        features: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize an Asset with an asset ID, coordinates, and features.

        Args:
            asset_id (str or int):
                The unique identifier for the asset.
            coordinates (list[list[float]]):
                A two-dimensional list representing the geometry of the asset
                in ``[[lon1, lat1], [lon2, lat2], ..., [lonN, latN]]`` format.
            features (dict[str, Any], optional):
                A dictionary of features. Defaults to an empty dict.
        """
        coords_check, output_msg = InputValidator.validate_coordinates(coordinates)
        if coords_check:
            self.coordinates = coordinates
        else:
            print(f'{output_msg} for {asset_id}; defaulting to an empty list.')
            self.coordinates = []

        self.features = features if features is not None else {}

    def add_features(
        self,
        additional_features: dict[str, Any],
        vector_features: list[str] | None = None,
        *,
        overwrite: bool = True,
    ) -> tuple[bool, int]:
        """
        Update the existing features in the asset.

        Args:
            additional_features (dict[str, Any]):
                New features to merge into the asset's features.
            vector_features (list[str], optional):
                A list of feature names that store their values in vectors.
            overwrite (bool, optional):
                Whether to overwrite existing features. Defaults to ``True``.

        Returns:
            tuple[bool, int]:
                A tuple containing two values:

                - updated (bool): ``True`` if any features were added or
                  updated, ``False`` otherwise.
                - n_pw (int): Number of possible worlds.

        Examples:
            >>> asset = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]]
            ... )
            >>> updated, n_pw = asset.add_features({'roof_type': 'gable'})
            >>> print(updated, n_pw)
            True 1
            >>> print(asset.features)
            {'roof_type': 'gable'}

            >>> updated, n_pw = asset.add_features(
            ...     {'possible_heights': [10, 15, 20], 'roof_type': 'hip'},
            ...     overwrite=True
            ... )
            >>> print(updated, n_pw)
            True 3
            >>> print(asset.features)
            {'roof_type': 'hip', 'possible_heights': [10, 15, 20]}

            >>> # When overwrite=False, existing keys are not updated
            >>> updated, n_pw = asset.add_features(
            ...     {'roof_type': 'flat', 'color': 'red'},
            ...     overwrite=False
            ... )
            >>> print(updated, n_pw)
            True 1
            >>> print(asset.features)
            {'roof_type': 'hip',
            'possible_heights': [10, 15, 20],
            'color': 'red'}
        """
        if vector_features is None:
            vector_features = list(DEFAULT_VECTOR_FEATURES)

        def update_possible_worlds(n_pw: int, key: str, val: Any) -> int:  # noqa: ANN401
            """Check a feature and update the number of possible worlds."""
            if not is_probabilistic_feature(key, val, vector_features):
                # The feature is a single value shared by every world, so
                # it says nothing about how many worlds there are.
                return n_pw

            if (n_pw != 1) and (n_pw != len(val)):
                print(
                    f'WARNING: # possible worlds was {n_pw} but now '
                    f'is {len(val)}. Something went wrong.'
                )
            return len(val)

        n_pw: int = 1

        if overwrite:
            # Overwrite existing features with new ones:
            self.features.update(additional_features)

            for key, val in additional_features.items():
                n_pw = update_possible_worlds(n_pw, key, val)

            updated = True

        else:
            # Only update with new keys, do not overwrite existing keys:
            updated = False
            for key, val in additional_features.items():
                if key not in self.features:
                    # write
                    self.features[key] = val

                    n_pw = update_possible_worlds(n_pw, key, val)

                    updated = True

        return updated, n_pw

    def get_geometry(self) -> BaseGeometry | None:
        """
        Convert coordinates into a robust Shapely geometry object.

        Wraps `GeoTools.list_of_lists_to_geometry` to handle errors gracefully.

        Returns:
            BaseGeometry | None: A valid Shapely geometry, or None if the
            coordinates are malformed or invalid.
        """
        if not self.coordinates:
            return None

        try:
            return GeoTools.list_of_lists_to_geometry(self.coordinates)

        except ValueError as e:
            print(f'WARNING: Asset geometry invalid. {e}')
            return None

    def get_centroid(self) -> list[list[float | None]]:
        """
        Get the centroid of the asset geometry.

        Returns:
            list[list[float]]:
                ``[[x, y]]`` if centroid can be calculated, ``[[None, None]]``
                otherwise.

        Examples:
            >>> asset = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4190, 37.7750]]
            ... )
            >>> asset.get_centroid()
            [[-122.4192, 37.77495]]

            >>> empty_asset = Asset(asset_id='empty', coordinates=[])
            Coordinates input is empty for empty; defaulting to an empty list.
            >>> empty_asset.get_centroid()
            [[None, None]]
        """
        geometry = self.get_geometry()

        if geometry:
            centroid = geometry.centroid
            return [[centroid.x, centroid.y]]

        return [[None, None]]

    def hash_asset(self) -> str:
        """
        Generate a unique hash for this asset based on its coordinates and features.

        This hash can be used to quickly identify duplicate assets by comparing
        geometry and attribute data. Both the coordinates and features are
        serialized as strings and then hashed using MD5.

        Returns:
            str: A hexadecimal string representing the MD5 hash of the asset.

        Example:
            >>> asset1 = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]],
            ...     features={'roof_type': 'gable'}
            ... )
            >>> asset2 = Asset(
            ...     asset_id='124',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]],
            ...     features={'roof_type': 'flat'}
            ... )
            >>> hash1 = asset1.hash_asset()
            >>> print(hash1)
            1e629fc184329ea688c648e7663b439d
            >>> hash2 = asset2.hash_asset()
            >>> print(hash2)
            ac0d104a411f92bffff8cc1398257dd8
            >>> hash1 != hash2
            True
        """
        coord_str = str(self.coordinates)
        feat_str = str(self.features)
        return hashlib.md5((coord_str + feat_str).encode()).hexdigest()

    def remove_features(self, features_to_remove: Iterable[str]) -> bool:
        """
        Remove specified features from the asset.

        Args:
            features_to_remove(Iterable[str]):
                An iterable of feature keys to remove from the Asset features.
                Accepts iterable types such as ``list``, ``tuple``, ``set``, or
                ``dict_keys``.

        Returns:
            bool: ``True`` if at least one feature was removed; ``False``
            otherwise.

        Raises:
            TypeError: If ``features_to_remove`` is not an iterable of strings.

        Example:
            >>> asset = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]]
            ... )
            >>> asset.features = {'color': 'red', 'height': 10}
            >>> asset.remove_features(['color'])
            True
            >>> asset.features
            {'height': 10}
        """
        if not isinstance(features_to_remove, Iterable) or not all(
            isinstance(k, str) for k in features_to_remove
        ):
            raise TypeError('features_to_remove must be an iterable of strings.')

        removed_count = 0
        for feature_key in features_to_remove:
            if feature_key in self.features:
                del self.features[feature_key]
                removed_count += 1

        return removed_count > 0

    def print_info(self) -> None:
        """
        Print the asset's coordinates and feature attributes.

        This method outputs the spatial coordinates and all associated
        feature key-value pairs of the asset to the console.

        Returns:
            None

        Example:
            >>> asset = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]],
            ...     features={'roof_type': 'gable'}
            ... )
            >>> asset.print_info()
            Coordinates:  [[-122.4194, 37.7749], [-122.4180, 37.7755]]
            Features:  {'roof_type': 'gable'}
        """
        print('Coordinates: ', self.coordinates)
        features_json = json.dumps(self.features, indent=2)
        print(f'Features: {features_json}')


class AssetInventory:
    """
    A class representing a collection of Assets managed as an inventory.

    This class provides methods to add, manipulate, join, write and query
    a collection of :`class:Asset` objects.

    To import the :class:`AssetInventory` class, use:

    .. code-block:: python

        from brails.types.asset_inventory import AssetInventory
    """

    def __init__(self, vector_features: list[str] | None = None) -> None:
        """
        Initialize AssetInventory with an empty inventory dictionary.

        Args:
            vector_features (list[str], optional):
                Names of the features whose values are vectors. For
                these, a plain list is a single value, not one entry
                per possible world. Defaults to
                :data:`DEFAULT_VECTOR_FEATURES`.
        """
        self.inventory: dict = {}
        self.housing_unit_inventory: HousingUnitInventory | None = None
        self.n_pw = 1
        self.vector_features: list[str] = (
            list(DEFAULT_VECTOR_FEATURES)
            if vector_features is None
            else list(vector_features)
        )

    def add_asset(self, asset_id: str | int, asset: Asset) -> bool:
        """
        Add an Asset to the inventory.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.
            asset(Asset):
                The asset to be added.

        Returns:
            bool: ``True`` if the asset was added successfully, ``False``
            otherwise.

        Raises:
            TypeError: If ``asset`` is not an instance of :class:`Asset`.

        Examples:
            >>> asset = Asset(
            ...     asset_id='001',
            ...     coordinates=[[-122.4194, 37.7749]],
            ...     features={'type': 'building'}
            ... )
            >>> inventory = AssetInventory()
            >>> success = inventory.add_asset('001', asset)
            >>> print(success)
            True

            >>> # Adding the same asset_id again will fail
            >>> success = inventory.add_asset('001', asset)
            >>> print(success)
            False
        """
        if not isinstance(asset, Asset):
            raise TypeError('Expected an instance of Asset.')

        if asset_id in self.inventory:
            print(f'Asset with id {asset_id} already exists. Asset was not added')
            return False

        self.inventory[asset_id] = asset
        return True

    def add_asset_coordinates(
        self,
        asset_id: str | int,
        coordinates: list[list[float]] | list[list[list[float]]],
    ) -> bool:
        """
        Add an ``Asset`` to the inventory by adding its coordinate information.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.
            coordinates (list[list[float]] | list[list[list[float]]]):
                A nested list of floats representing the geometry.
                Accepts Points, LineStrings, Polygons, and Multi-geometries.

        Returns:
            bool: ``True`` if the asset was added successfully, ``False``
            otherwise.

        Examples:
            >>> inventory = AssetInventory()
            >>> coords = [[-122.42, 37.77], [-122.43, 37.78], [-122.44, 37.79]]
            >>> success = inventory.add_asset_coordinates('A123', coords)
            >>> print(success)
            True

            >>> # Attempt to add the same asset_id again
            >>> success = inventory.add_asset_coordinates('A123', coords)
            >>> print(success)
            False
        """
        existing_asset = self.inventory.get(asset_id, None)

        if existing_asset is not None:
            print(
                f'Asset with id {asset_id} already exists. Coordinates were '
                'not added'
            )
            return False

        # Create asset and add using id as the key:
        asset = Asset(asset_id, coordinates)
        self.inventory[asset_id] = asset

        return True

    def add_asset_features(
        self,
        asset_id: str | int,
        new_features: dict[str, Any],
        *,
        overwrite: bool = True,
    ) -> bool:
        """
        Add new asset features to the Asset with the specified ``asset_id``.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.
            new_features(dict):
                A dictionary of features to add to the asset.
            overwrite(bool):
                Whether to overwrite existing features with the same keys.
                Defaults to ``True``.

        Returns:
            bool:
                ``True`` if features were successfully added, ``False`` if the
                asset does not exist or the operation fails.

        Examples:
            >>> inventory = AssetInventory()
            >>> inventory.add_asset_coordinates(
            ...     'A123',
            ...     [[-122.4194, 37.7749], [-122.4180, 37.7755]]
            ... )
            True
            >>> features = {'height': 10, 'material': 'concrete'}
            >>> success = inventory.add_asset_features('A123', features)
            >>> print(success)
            True

            >>> # Add features without overwriting existing keys
            >>> more_features = {'material': 'steel', 'color': 'red'}
            >>> success = inventory.add_asset_features(
            ...     'A123',
            ...     more_features,
            ...     overwrite=False
            ... )
            >>> print(success)
            True
            >>> asset = inventory.inventory['A123']
            >>> print(asset.features)
            {'height': 10, 'material': 'concrete', 'color': 'red'}
        """
        asset = self.inventory.get(asset_id, None)
        if asset is None:
            print(
                f'No existing Asset with id {asset_id} found. Asset '
                'features not added.'
            )
            return False

        status, n_pw = asset.add_features(
            new_features, vector_features=self.vector_features, overwrite=overwrite
        )
        if n_pw == 1:
            # The new feature is deterministic.
            # This doesn't change the inventory's n_pw, so do nothing.
            pass
        elif self.n_pw == 1:
            # The inventory was deterministic (n_pw=1), but this new
            # feature is probabilistic. Set the inventory's n_pw.
            self.n_pw = n_pw
        elif self.n_pw == n_pw:
            # The new feature's n_pw matches the inventory's n_pw.
            # This is the consistent case, so do nothing.
            pass
        else:
            # This is a mismatch. The inventory had one n_pw (e.g., 5)
            # and the new feature has a different one (e.g., 3).
            print(
                f'WARNING: # possible worlds was {self.n_pw} but is now '
                f'{n_pw}. Something went wrong.'
            )
            # Set to the new value as per the original logic
            self.n_pw = n_pw
        return status

    def add_model_predictions(
        self, predictions: dict[Any, Any], feature_key: str
    ) -> None:
        """
        Add model predictions to the inventory.

        This method goes through the inventory and updates each item by adding
        the corresponding model prediction as a new feature under the specified
        key. Items without a matching prediction are left unchanged.

        Args:
            predictions(dict):
                A dictionary where keys correspond to inventory items and
                values represent the predicted features to be added.
            feature_key(str):
                The key under which the predictions will be stored as a new
                feature in each inventory item.

        Returns:
            None

        Raises:
            TypeError:
                If ``predictions`` is not a dictionary or ``feature_key`` is
                not a string.
            ValueError:
                If none of the keys in ``predictions`` exist in the inventory.

        Example:
            Suppose the inventory contains assets with IDs 1, 3, and 12. To add
            predicted roof types for these assets:

            >>> predictions = {1: 'gable', 3: 'flat', 12: 'hip'}
            >>> inventory.add_model_predictions(
                    predictions,
                    feature_key='roof_type'
                )

            After this call, each asset with an ID in ``predictions`` will have
            a new feature ``roof_type`` set to the corresponding predicted
            value.

        """
        # Validate predictions input:
        if not isinstance(predictions, dict):
            raise TypeError("Expected 'predictions' to be a dictionary.")

        # Ensure at least one key in predictions matches inventory:
        common_keys = set(predictions.keys()) & set(self.inventory.keys())
        if not common_keys:
            raise ValueError(
                "None of the keys in 'predictions' exist in 'self.inventory'."
            )

        # Validate feature_key input:
        if not isinstance(feature_key, str):
            raise TypeError("Expected 'feature_key' to be a string.")

        # Update inventory items with corresponding predictions:
        for asset_id in self.get_asset_ids():
            if asset_id in predictions:
                self.add_asset_features(
                    asset_id, {feature_key: predictions.get(asset_id)}
                )

    def change_feature_names(self, feature_name_mapping: dict[str, str]) -> None:
        """
        Rename feature names in ``AssetInventory`` via user-specified mapping.

        Args:
            feature_name_mapping (dict):
                A dictionary where keys are the original feature names and
                values are the new feature names.

        Raises:
            TypeError:
                If the mapping is not a dictionary or contains invalid
                key-value pairs.

        Example:
            First create an :class:`AssetInventory` with two assets with IDs
            ``asset1`` and ``asset2``.

            >>> inventory = AssetInventory()
            >>> asset1 = Asset(
            ...     asset_id='asset1',
            ...     coordinates=[
            ...         [-122.4194, 37.7749],
            ...         [-122.4194, 37.7849],
            ...         [-122.4094, 37.7849],
            ...         [-122.4094, 37.7749],
            ...         [-122.4194, 37.7749]
            ...     ],
            ...     features={'old_name': 100, 'unchanged': 50}
            ... )
            >>> inventory.add_asset('asset1', asset1)
            >>> asset2 = Asset(
            ...     asset_id='asset2',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4094, 37.7849]],
            ...     features={'old_name': 200}
            ... )
            >>> inventory.add_asset('asset2', asset2)

            Then, change the names of the features of these assets.

            >>> inventory.change_feature_names({'old_name': 'new_name'})
            >>> inventory.inventory['asset1'].features
            {'new_name': 100, 'unchanged': 50}
            >>> inventory.inventory['asset2'].features
            {'new_name': 200}
        """
        # Validate that feature_name_mapping is a dictionary:
        if not isinstance(feature_name_mapping, dict):
            raise TypeError("The 'feature_name_mapping' must be a dictionary.")

        # Validate that all keys and values are strings:
        for original_name, new_name in feature_name_mapping.items():
            if not isinstance(original_name, str) or not isinstance(new_name, str):
                raise TypeError(
                    'Both original and new names of features must be '
                    f'strings. Invalid pair: ({original_name}, '
                    f'{new_name})'
                )

        # Iterate over each asset in the AssetInventory:
        for asset in self.inventory.values():
            # Iterate over the defined name mappings in feature_name_mapping:
            for original_name, new_name in feature_name_mapping.items():
                # If the original_name for a feature name exists in the asset's
                # features:
                if original_name in asset.features:
                    # Rename the feature by popping the old key and adding it
                    # under the new key:
                    asset.features[new_name] = asset.features.pop(original_name)

    def combine(
        self,
        other_inventory: AssetInventory,
        asset_id_map: dict[str | int, str | int] | None = None,
    ) -> dict[str | int, str | int]:
        """
        Add another AssetInventory's assets to this one, handling conflicts.

        This method adds all unique assets from `other_inventory` to the
        current inventory (`self`).

        - **Asset De-duplication:** It automatically skips any incoming
          asset that is an exact duplicate of an existing one (based on
          a hash of its geometry and features).
        - **Asset ID Conflicts:** It resolves all asset ID collisions.
          You can provide an `asset_id_map` to suggest new IDs. If a
          suggested ID (or an original ID) already exists, a new
          sequential numeric ID will be assigned.
        - **Housing Unit Merging:** If both inventories have linked housing unit
          data, this method validates both, then merges the incoming
          `HousingUnitInventory`, re-indexes all new housing unit IDs to
          prevent conflicts, and updates the `HousingUnits` feature
          on all newly-added assets.

        Args:
            other_inventory (AssetInventory):
                The secondary inventory whose assets are added to this one.
            asset_id_map (dict, optional):
                A dictionary mapping {original_asset_id: suggested_asset_id}.
                Use this to rename asset IDs from `other_inventory` as they
                are added.

        Returns:
            dict[str | int, str | int]:
                A `final_id_map` dictionary, which acts as a "receipt"
                mapping {original_asset_id: final_asset_id}. This is
                critical for data traceability, as it reports the
                actual ID assigned after resolving all conflicts.

        Raises:
            ValueError, LookupError, TypeError:
                If either `self` or `other_inventory` has an invalid
                housing unit state (e.g., orphan IDs, invalid data types)
                before the assets are added.
        """
        # --- 1. Asset Merging ---

        # Build a set of existing hashes for O(1) duplicate checking
        existing_hashes = {
            asset.hash_asset(): key for key, asset in self.inventory.items()
        }
        next_id = self._get_next_numeric_id()
        final_id_map = {}

        for orig_key, asset in other_inventory.inventory.items():
            asset_hash = asset.hash_asset()

            # Skip any asset that is a perfect duplicate
            if asset_hash in existing_hashes:
                continue

            # Determine the suggested ID, using the map or the original key
            suggested_id = (
                asset_id_map.get(orig_key, orig_key) if asset_id_map else orig_key
            )
            final_id = suggested_id

            # Ensure uniqueness.
            while final_id in self.inventory:
                final_id = next_id
                next_id += 1

            # Add the asset to the inventory and record its final ID
            self.add_asset(final_id, asset)
            final_id_map[orig_key] = final_id
            existing_hashes[asset_hash] = (
                final_id  # Add hash to prevent duplicates from within other_inventory
            )

        # --- 2. Housing Unit Inventory Merging ---
        hu_id_remap = {}

        # Case 1: We have no inventory, but the incoming one does.
        if (
            self.housing_unit_inventory is None
            and other_inventory.housing_unit_inventory is not None
        ):
            # First, validate the incoming inventory to prevent merging bad data
            print('Validating incoming housing unit inventory...')
            other_inventory.validate_housing_unit_assignments()

            # Use deepcopy to avoid two AssetInventory objects pointing to
            # the same mutable HousingUnitInventory object.
            self.housing_unit_inventory = deepcopy(
                other_inventory.housing_unit_inventory
            )
            # No ID remapping is needed; the new assets' links are already correct.

        # Case 2: Both inventories have housing unit data.
        elif (
            self.housing_unit_inventory is not None
            and other_inventory.housing_unit_inventory is not None
        ):
            # Validate both inventories before attempting to merge.
            print('Validating current housing unit inventory...')
            self.validate_housing_unit_assignments()
            print('Validating incoming housing unit inventory...')
            other_inventory.validate_housing_unit_assignments()

            # Merge the incoming inventory into our own. This returns a
            # map of {old_hu_id: new_hu_id} for all merged housing units.
            print('Merging housing unit inventories... Re-indexing incoming data.')
            hu_id_remap = self.housing_unit_inventory.merge_inventory(
                other_inventory.housing_unit_inventory
            )

        # Case 3 (both are None) -> do nothing.
        # Case 4 (self has one, other is None) -> do nothing.

        # --- 3. Asset Housing-Unit Link Fixing ---

        # If we remapped housing unit IDs (Case 2), we must update the
        # 'HousingUnits' feature on all newly-added assets.
        if hu_id_remap:
            print('Updating housing unit links for merged assets...')

            # We only need to check the assets we *just added*.
            for final_id in final_id_map.values():
                asset = self.inventory[final_id]
                old_hu_id_list = asset.features.get('HousingUnits')

                if old_hu_id_list:
                    # Rebuild the list using the re-map.
                    remapped_hu_id_list = [
                        hu_id_remap[old_hu_id] for old_hu_id in old_hu_id_list
                    ]

                    self.add_asset_features(
                        final_id,
                        {'HousingUnits': remapped_hu_id_list},
                        overwrite=True,
                    )

        return final_id_map

    def merge(
        self,
        others: Mapping[str, AssetInventory],
        precedence: Sequence[str],
        *,
        name: str,
        source_features: Sequence[str] | None = HAZUS_EQ_SOURCE_FEATURES,
    ) -> AssetInventory:
        """
        Merge per-source inventories of this inventory's assets.

        This inventory is the spine, labeled ``name``: its assets define
        the asset set and the geometry of the result. Every other source
        is an inventory keyed by asset ids of this inventory, possibly a
        subset. Each feature of each asset is taken from the first source
        in ``precedence`` that holds a value for it. ``None``, any value
        :func:`pandas.isna` reports as missing, the empty string and
        :data:`MISSING_VALUE_MARKER` are not values. A feature that no
        source holds a value for keeps the missing marker of the first
        source in ``precedence`` that has it.

        Unless ``source_features`` is ``None``, each asset also receives
        the feature :data:`SOURCE_STRING_KEY`: one character per
        feature of ``source_features``, in that order, holding the
        :data:`SOURCE_CODES` code of the source the value came from, or
        :data:`SOURCE_CODE_NONE` where no source holds one.

        Neither this inventory nor any source is modified. The result
        holds deep copies, with assets inserted in ascending order of the
        asset id as a string, so it does not depend on the order of
        ``others`` or of the assets in any inventory.

        Args:
            others (Mapping[str, AssetInventory]):
                The other sources, keyed by their labels.
            precedence (Sequence[str]):
                Every source label, ``name`` included, highest precedence
                first.
            name (str):
                The label of this inventory.
            source_features (Sequence[str] | None, optional):
                The features the source string covers, in order. Defaults
                to :data:`HAZUS_EQ_SOURCE_FEATURES`. ``None`` writes no
                source string.

        Returns:
            AssetInventory:
                A new inventory holding every asset of this one. It
                declares the vector features of all sources, and its
                number of possible worlds is the largest among them.

        Raises:
            TypeError:
                If ``others`` is not a mapping, a label is not a string, a
                source is not an :class:`AssetInventory`, or
                ``precedence`` or ``source_features`` is a single string.
            ValueError:
                If ``name`` labels one of ``others``; if ``precedence``
                omits a label, repeats one or names a label that is not a
                source; if ``source_features`` repeats a feature or names
                :data:`SOURCE_STRING_KEY`; if a source string is
                requested and a label has no code in
                :data:`SOURCE_CODES`; if any source already carries
                :data:`SOURCE_STRING_KEY`; if a source holds an asset
                id that this inventory lacks; if two sources hold
                different numbers of possible worlds above one; or if a
                source holds a housing unit inventory.
            BrailsError:
                If a source with one possible world holds a list, which
                the result would read as realizations, while another
                source holds several worlds, or if the result fails
                :meth:`validate_possible_worlds`.

        Examples:
            >>> spine = AssetInventory()
            >>> _ = spine.add_asset_coordinates('A1', [[-122.4, 37.8]])
            >>> _ = spine.add_asset_features('A1', {'NumberOfStories': 'NA'})
            >>> nsi = AssetInventory()
            >>> _ = nsi.add_asset_coordinates('A1', [[-122.4, 37.8]])
            >>> _ = nsi.add_asset_features(
            ...     'A1', {'NumberOfStories': 2, 'OccupancyClass': 'RES1'}
            ... )
            >>> merged = spine.merge(
            ...     {'nsi': nsi}, ['footprint', 'nsi'], name='footprint'
            ... )
            >>> features = merged.get_asset_features('A1')[1]
            >>> features['NumberOfStories'], features['SourceString']
            (2, 'N-N-----')
        """
        if source_features is not None and not isinstance(source_features, str):
            # Read once, so that the checks do not consume an iterator.
            source_features = list(source_features)
        sources, order, n_pw = self._check_merge_sources(
            others, precedence, name, source_features
        )

        vector_features = list(
            dict.fromkeys(
                feature
                for label in order
                for feature in sources[label].vector_features
            )
        )
        result = type(self)(vector_features=vector_features)
        result.n_pw = n_pw

        for asset_id in sorted(self.inventory, key=_merge_id_order):
            keys: list[str] = []
            values: dict[str, Any] = {}
            winners: dict[str, str] = {}
            for label in order:
                asset = sources[label].inventory.get(asset_id)
                if asset is None:
                    continue
                for key, value in asset.features.items():
                    if key not in values:
                        keys.append(key)
                        values[key] = value
                    if key not in winners and not _is_missing_value(value):
                        winners[key] = label
                        values[key] = value

            features = {key: deepcopy(values[key]) for key in keys}
            if source_features is not None:
                features[SOURCE_STRING_KEY] = ''.join(
                    SOURCE_CODES[winners[feature]]
                    if feature in winners
                    else SOURCE_CODE_NONE
                    for feature in source_features
                )

            merged = copy(self.inventory[asset_id])
            merged.coordinates = deepcopy(merged.coordinates)
            merged.features = features
            result.add_asset(asset_id, merged)

        result.validate_possible_worlds()
        return result

    def _check_merge_labels(
        self,
        others: Mapping[str, AssetInventory],
        precedence: Sequence[str],
        name: str,
    ) -> tuple[dict[str, AssetInventory], list[str]]:
        """
        Check the source labels and the precedence given to :meth:`merge`.

        Args:
            others (Mapping[str, AssetInventory]):
                The other sources, keyed by their labels.
            precedence (Sequence[str]):
                Every source label, highest precedence first.
            name (str):
                The label of this inventory.

        Returns:
            tuple[dict[str, AssetInventory], list[str]]:
                Every source keyed by its label, this inventory included,
                and the labels in order of precedence.

        Raises:
            TypeError:
                As documented in :meth:`merge`.
            ValueError:
                As documented in :meth:`merge`.
        """
        if not isinstance(name, str):
            raise TypeError(f'The source label must be a string, not {name!r}.')
        if not isinstance(others, Mapping):
            raise TypeError(
                'others must map source labels to inventories, not '
                f'{type(others).__name__}.'
            )
        for label, inventory in others.items():
            if not isinstance(label, str):
                raise TypeError(f'Source labels must be strings, not {label!r}.')
            if not isinstance(inventory, AssetInventory):
                raise TypeError(
                    f'Source {label!r} is not an AssetInventory '
                    f'(type {type(inventory).__name__}).'
                )
        if name in others:
            raise ValueError(
                f'The label {name!r} names this inventory and one of others.'
            )
        sources: dict[str, AssetInventory] = {name: self, **others}

        if isinstance(precedence, str):
            raise TypeError(
                'precedence must be a sequence of labels, not the string '
                f'{precedence!r}.'
            )
        order = list(precedence)
        repeated = sorted(
            (lbl for lbl, count in Counter(order).items() if count > 1), key=str
        )
        omitted = sorted(set(sources) - set(order))
        unknown = sorted({str(lbl) for lbl in order if lbl not in sources})
        if repeated or omitted or unknown:
            raise ValueError(
                'precedence must list every source label once. '
                f'Repeated: {repeated}; omitted: {omitted}; '
                f'not a source: {unknown}.'
            )
        return sources, order

    def _check_merge_sources(
        self,
        others: Mapping[str, AssetInventory],
        precedence: Sequence[str],
        name: str,
        source_features: Sequence[str] | None,
    ) -> tuple[dict[str, AssetInventory], list[str], int]:
        """
        Check the arguments of :meth:`merge` and collect its sources.

        Args:
            others (Mapping[str, AssetInventory]):
                The other sources, keyed by their labels.
            precedence (Sequence[str]):
                Every source label, highest precedence first.
            name (str):
                The label of this inventory.
            source_features (Sequence[str] | None):
                The features the source string covers, or ``None``.

        Returns:
            tuple[dict[str, AssetInventory], list[str], int]:
                Every source keyed by its label, this inventory included;
                the labels in order of precedence; and the number of
                possible worlds of the result.

        Raises:
            TypeError:
                As documented in :meth:`merge`.
            ValueError:
                As documented in :meth:`merge`.
        """
        sources, order = self._check_merge_labels(others, precedence, name)
        if source_features is not None:
            _check_source_features(source_features, order)

        carrying = [
            label
            for label in order
            if any(
                SOURCE_STRING_KEY in asset.features
                for asset in sources[label].inventory.values()
            )
        ]
        if carrying:
            raise ValueError(
                f'Source(s) {carrying} already carry '
                f'{SOURCE_STRING_KEY!r}, which describes an earlier '
                'merge. Merge the per-source inventories in one call, or '
                'remove the feature first.'
            )

        with_housing_units = [
            label
            for label in order
            if sources[label].housing_unit_inventory is not None
        ]
        if with_housing_units:
            raise ValueError(
                f'Source(s) {with_housing_units} hold a housing unit '
                'inventory. merge does not combine housing unit inventories, '
                'while the combine method does.'
            )

        world_counts = {
            label: sources[label].n_pw for label in order if sources[label].n_pw > 1
        }
        if len(set(world_counts.values())) > 1:
            raise ValueError(
                'The sources hold different numbers of possible worlds: '
                f'{world_counts}.'
            )
        n_pw = max(sources[label].n_pw for label in order)
        if n_pw > 1:
            self._check_one_world_lists(sources, order, n_pw)

        foreign = [
            f'{asset_id!r} ({label})'
            for label in order
            for asset_id in sorted(
                sources[label].inventory.keys() - self.inventory.keys(),
                key=_merge_id_order,
            )
        ]
        if foreign:
            raise ValueError(
                f'{len(foreign)} asset id(s) held by a source are absent '
                f'from {name!r}: {_report_first_problems(foreign, ", ")}.'
            )

        return sources, order, n_pw

    @staticmethod
    def _check_one_world_lists(
        sources: dict[str, AssetInventory], order: list[str], n_pw: int
    ) -> None:
        """
        Refuse a one-world source holding lists in a merge of several worlds.

        Args:
            sources (dict[str, AssetInventory]):
                Every source keyed by its label.
            order (list[str]):
                The labels in order of precedence.
            n_pw (int):
                The number of possible worlds of the result, above one.

        Raises:
            BrailsError:
                If a source with one possible world holds a list for a
                feature that is not a vector feature of the result. The
                message names the first such source and up to
                :data:`MAX_REPORTED_WORLD_PROBLEMS` of its values.
        """
        vector_features = {
            feature for label in order for feature in sources[label].vector_features
        }
        for label in order:
            source = sources[label]
            if source.n_pw != 1:
                continue
            lists = [
                f'asset {asset_id!r}, feature {key!r}'
                for asset_id in sorted(source.inventory, key=_merge_id_order)
                for key, value in source.inventory[asset_id].features.items()
                if is_probabilistic_feature(key, value, vector_features)
            ]
            if lists:
                raise BrailsError(
                    f'Source {label!r} has one possible world but holds '
                    f'{len(lists)} list value(s), which a merge with '
                    f'{n_pw} possible worlds would read as realizations: '
                    f'{_report_first_problems(lists, "; ")}.'
                )

    @staticmethod
    def has_area(coordinates: object) -> bool:
        """
        Return whether the exterior rings of a footprint enclose any area.

        A footprint without area receives no geohash asset id
        (:func:`brails.utils.asset_id.assign_asset_ids`), so callers drop
        such footprints with this test first.

        Args:
            coordinates (object):
                A BRAILS coordinate list (a list of ``[lon, lat]`` for a
                polygon, a list of such rings for a multipolygon), a GeoJSON
                mapping of type ``Polygon`` or ``MultiPolygon``, or an object
                with ``__geo_interface__``, such as a Shapely geometry, in
                EPSG:4326.

        Returns:
            bool:
                True when the absolute areas of the exterior rings sum to
                more than zero.

        Raises:
            ValueError:
                If the coordinates are not one of the accepted forms, or a
                vertex is not a finite coordinate pair within EPSG:4326
                ranges.
        """
        return _has_area(coordinates)

    @staticmethod
    def centroid_and_plan_area(coordinates: object) -> tuple[float, float, float]:
        """
        Return a footprint's centroid and plan area, as its asset id reads them.

        The centroid is the planar centroid of the exterior rings in
        EPSG:4326, the point whose geohash is the footprint's id. It ignores
        interior rings, so for a geometry with holes it differs from the
        Shapely centroid that :meth:`Asset.get_centroid` returns. The plan
        area is the exterior rings' shoelace area in square
        degrees scaled to a sphere of radius
        :data:`brails.constants.EARTH_RADIUS_M` at the centroid's
        latitude. Neither depends on a ring's starting vertex, orientation
        or closing vertex, or on the order of the parts.

        Args:
            coordinates (object):
                A footprint in any form :meth:`has_area` accepts.

        Returns:
            tuple[float, float, float]:
                The centroid's longitude and latitude, and the plan area in
                square meters, unrounded.

        Raises:
            ValueError:
                In the cases :meth:`has_area` raises it, or if the exterior
                rings enclose no area.
        """
        return _centroid_and_plan_area(coordinates)

    def convert_polygons_to_centroids(self) -> None:
        """
        Convert geometries in the inventory to their centroid points.

        Iterates through the asset inventory and replaces the coordinates of
        each geometry with the coordinates of its centroid using the
        logic defined in `Asset.get_centroid()`.

        Notes:
            - Points are left unchanged.
            - Polygons, LineStrings, and Multi-geometries are converted to their centroid.
            - Invalid or malformed geometries will be replaced with ``[[None, None]]``
              and a warning will be printed by the Asset class.

        Modifies:
            self.inventory (dict):
                Updates the ``coordinates`` field of each asset in-place.

        Example:
            >>> inventory = AssetInventory()
            >>> # 1. Polygon
            >>> inventory.add_asset_coordinates(
            ...     'asset1',
            ...     [
            ...         [-122.4194, 37.7749],
            ...         [-122.4194, 37.7849],
            ...         [-122.4094, 37.7849],
            ...         [-122.4094, 37.7749],
            ...         [-122.4194, 37.7749]
            ...     ]
            ... )
            >>> # 2. LineString
            >>> inventory.add_asset_coordinates(
            ...     'asset2',
            ...     [
            ...         [-122.4194, 37.7749],
            ...         [-122.4094, 37.7849]
            ...     ]
            ... )
            >>> # 3. MultiPolygon
            >>> inventory.add_asset_coordinates(
            ...     'asset3',
            ...     [
            ...         [
            ...             [-122.4194, 37.7749], [-122.4194, 37.7849],
            ...             [-122.4094, 37.7849], [-122.4094, 37.7749],
            ...             [-122.4194, 37.7749]
            ...         ],
            ...         [
            ...             [-122.3000, 37.8000], [-122.3000, 37.8100],
            ...             [-122.2900, 37.8100], [-122.2900, 37.8000],
            ...             [-122.3000, 37.8000]
            ...         ]
            ...     ]
            ... )
            >>> inventory.convert_polygons_to_centroids()
            >>> inventory.get_asset_coordinates('asset1')
            [[-122.4144, 37.7799]]
            >>> inventory.get_asset_coordinates('asset2')
            [[-122.4144, 37.7799]]
        """
        for asset in self.inventory.values():
            asset.coordinates = asset.get_centroid()

    def get_asset_coordinates(self, asset_id: str | int) -> tuple[bool, list]:
        """
        Get the coordinates of a particular asset.

        Args:
            asset_id (str or int):
                The unique identifier for the asset.

        Returns:
            tuple[bool, list]:
                - **bool** – Indicates whether the asset was found.
                - **list** – A list of coordinate pairs in the format
                  ``[[lon1, lat1], [lon2, lat2], ..., [lonN, latN]]`` if
                  found, or an empty list if the asset does not exist.

        Example:
            >>> inventory = AssetInventory({
            ...     "A101": Asset(
            ...     coordinates=[[30.123, -97.456], [30.124, -97.457]]
            ...     ),
            ...     "B202": Asset(
            ...     coordinates=[[40.789, -74.123], [40.790, -74.124]]
            ...     )
            ... })
            >>> inventory.get_asset_coordinates("A101")
            (True, [[30.123, -97.456], [30.124, -97.457]])
            >>> inventory.get_asset_coordinates("Z999")
            (False, [])
        """
        asset = self.inventory.get(asset_id, None)
        if asset is None:
            return False, []

        return True, asset.coordinates

    def get_asset_features(self, asset_id: str | int) -> tuple[bool, dict[str, Any]]:
        """
        Get features of a particular asset.

        Args:
            asset_id (str or int):
                The unique identifier for the asset.

        Returns:
            tuple[bool, dict]:
                A tuple where the first element is a boolean indicating whether
                the asset was found, and the second element is a dictionary
                containing the asset's features if the asset is present.
                Returns an empty dictionary if the asset is not found.

        Examples:
            >>> inventory = AssetInventory()
            >>> asset = Asset(
            ...     asset_id='001',
            ...     coordinates=[[-122.4194, 37.7749]],
            ...     features={'height': 10, 'material': 'concrete'}
            ... )
            >>> inventory.add_asset('001', asset)
            True
            >>> found, features = inventory.get_asset_features('001')
            >>> found
            True
            >>> features
            {'height': 10, 'material': 'concrete'}

            >>> found, features = inventory.get_asset_features('nonexistent')
            >>> found
            False
            >>> features
            {}
        """
        asset = self.inventory.get(asset_id, None)
        if asset is None:
            return False, {}

        return True, asset.features

    def get_all_asset_features(self) -> set[str]:
        """
        Retrieves a set of unique feature keys present across all assets.

        Iterates through every asset and collects the keys from their 'features'
        dictionaries. This operation handles deduplication automatically.

        Returns:
            set[str]: A collection of unique feature names found in the inventory.
        """
        return {
            feature
            for asset in self.inventory.values()
            for feature in asset.features
        }

    def get_asset_ids(self) -> list[str | int]:
        """
        Retrieve the IDs of all assets in the inventory.

        Returns:
            list[str or int]:
                A list of asset IDs, which may be strings or
                integers.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset1',
            ...     [[-122.4194, 37.7749], [-122.4180, 37.7750]]
            ... )
            >>> _ = inventory.add_asset_coordinates(
            ...     2,
            ...     [[-74.0060, 40.7128], [-74.0055, 40.7130]]
            ... )
            >>> inventory.get_asset_ids()
            ['asset1', 2]
        """
        return list(self.inventory.keys())

    def get_assets_intersecting_polygon(self, bpoly: BaseGeometry):
        """
        Get assets with geometries intersecting a bounding polygon.

        This method performs a spatial intersection check between each asset's
        geometry in the inventory and a provided bounding polygon (or
        multipolygon). Assets that intersect the polygon are identified and
        retained. All non-intersecting assets are removed from the inventory.

        Args:
            bpoly (shapely.geometry.base.BaseGeometry):
                The bounding polygon or multipolygon used to determine spatial
                intersections.

        Raises:
            TypeError:
                If ``bpoly`` is not a ``Polygon`` or ``MultiPolygon``.

        Example:
            >>> from shapely.geometry import Polygon
            >>> inventory = AssetInventory()
            >>> # A LineString in Dallas, TX (will intersect the Dallas bpoly):
            >>> _ = inventory.add_asset_coordinates(
            ...     'bridge_A',
            ...     [[-96.8003, 32.7767], [-96.7998, 32.7770]]
            ... )
            >>> # A Polygon in Los Angeles, CA (will NOT intersect the bpoly):
            >>> _ = inventory.add_asset_coordinates(
            ...     'tower_B',
            ...     [
            ...         [-118.2450, 34.0537],
            ...         [-118.2450, 34.0540],
            ...         [-118.2445, 34.0540],
            ...         [-118.2445, 34.0537],
            ...         [-118.2450, 34.0537],
            ...     ]
            ... )
            >>> # A bounding polygon roughly around downtown Dallas:
            >>> bpoly = Polygon([
            ...     (-96.81, 32.77),
            ...     (-96.81, 32.78),
            ...     (-96.79, 32.78),
            ...     (-96.79, 32.77),
            ...     (-96.81, 32.77)
            ... ])
            >>>
            >>> inventory.get_assets_intersecting_polygon(bpoly)
            >>> 'bridge_A' in inventory.inventory
            True
            >>> 'tower_B' in inventory.inventory
            False
        """
        # Validate bounding polygon type:
        if bpoly.geom_type not in ['Polygon', 'MultiPolygon']:
            raise TypeError(
                f'Invalid bounding polygon type: {bpoly.geom_type}. '
                "Expected 'Polygon' or 'MultiPolygon'."
            )

        # Fix the bounding polygon if it is not well-formed:
        if not bpoly.is_valid:
            bpoly = bpoly.buffer(0)

        valid_geometries = {}
        keys_to_remove = set()

        for key, asset in self.inventory.items():
            geom = asset.get_geometry()

            if geom and not geom.is_empty and geom.is_valid:
                valid_geometries[key] = geom
            else:
                keys_to_remove.add(key)

        if valid_geometries:
            non_intersecting_keys = GeoTools.compare_geometries(
                bpoly, valid_geometries, 'intersects'
            )
            keys_to_remove.update(non_intersecting_keys)

        for key in keys_to_remove:
            self.remove_asset(key)

    def get_coordinates(
        self,
    ) -> tuple[list[list[list[float]] | list[list[list[float]]]], list[str | int]]:
        """
        Get geometry(coordinates) and keys of all assets in the inventory.

        Returns:
            tuple:
                A tuple containing:
                - A list of coordinates for each asset. Depending on the geometry
                  type, this will be:
                  - Depth 2: ``[[lon, lat], ...]`` (Point, LineString, Polygon)
                  - Depth 3: ``[[[lon, lat], ...], ...]`` (MultiLineString, MultiPolygon)
                - A list of asset keys corresponding to each asset.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset1',
            ...     [[-122.4194, 37.7749], [-122.4180, 37.7750]]
            ... )
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset2',
            ...     [[-74.0060, 40.7128], [-74.0055, 40.7130]]
            ... )
            >>> coords, keys = inventory.get_coordinates()
            >>> coords
            [[[-122.4194, 37.7749], [-122.4180, 37.7750]],
             [[-74.0060, 40.7128], [-74.0055, 40.7130]]]
            >>> keys
            ['asset1', 'asset2']
        """
        coordinates = [asset.coordinates for asset in self.inventory.values()]
        asset_ids = list(self.inventory.keys())

        return coordinates, asset_ids

    def get_random_sample(
        self,
        nsamples: int,
        seed: int | float | str | bytes | bytearray | None = None,
    ) -> AssetInventory:
        """
        Generate a smaller asset inventory with a random selection of assets.

        This method randomly selects ``nsamples`` assets from the existing
        inventory and returns a new :class:`AssetInventory` instance containing
        only these sampled assets. The randomness can be controlled using an
        optional ``seed`` for reproducibility.

        Args:
            nsamples (int):
                The number of assets to randomly sample from the inventory.
                Must be a positive integer not exceeding the total number of
                assets.
            seed (int or float or str or bytes or bytearray or None, optional):
                A seed value for the random generator to ensure
                reproducibility. If None, the system default (current system
                time) is used.

        Returns:
            AssetInventory:
                A new :class:`AssetInventory` instance containing the randomly
                selected subset of assets.

        Raises:
            ValueError:
                If ``nsamples`` is not a positive integer or exceeds the number
                of assets in the inventory.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset('asset1', Asset(
            ...     asset_id='asset1',
            ...     coordinates=[[-122.4194, 37.7749]],
            ...     features={'type': 'building'}
            ... ))
            >>> _ = inventory.add_asset('asset2', Asset(
            ...     asset_id='asset2',
            ...     coordinates=[[-74.0060, 40.7128]],
            ...     features={'type': 'bridge'}
            ... ))
            >>> _ = inventory.add_asset('asset3', Asset(
            ...     asset_id='asset3',
            ...     coordinates=[[2.3522, 48.8566]],
            ...     features={'type': 'tower'}
            ... ))
            >>> sample_inventory = inventory.get_random_sample(
            ...     nsamples=2,
            ...     seed=42
            ... )
            >>> sorted(sample_inventory.get_asset_ids())
            ['asset1', 'asset3']
        """
        if not isinstance(nsamples, int) or nsamples <= 0:
            raise ValueError('Number of samples must be a positive integer.')

        if nsamples > len(self.inventory):
            raise ValueError(
                'Number of samples cannot exceed the number of '
                'assets in the inventory'
            )

        if seed is not None:
            random.seed(seed)

        random_keys = random.sample(list(self.inventory.keys()), nsamples)

        result = AssetInventory()
        for key in random_keys:
            result.add_asset(key, self.inventory[key])

        return result

    def get_extent(self, buffer: str | list[float] = 'default') -> box:
        """
        Calculate the geographical extent of the inventory.

        Args:
            buffer (str or list[float]): A string or a list of 4 floats.

                - ``'default'`` applies preset buffer values.
                - ``'none'`` applies zero buffer values.
                - A list of 4 floats defines custom buffer values for each
                  edge of the bounding box in the order:
                  [minlon buffer, minlat buffer, maxlon buffer, maxlat buffer].

        Returns:
            shapely.geometry.box:
                A Shapely polygon representing the extent of the inventory,
                with buffer applied.

        Raises:
            ValueError: If the ``buffer`` input is invalid.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset('asset1', Asset(
            ...     asset_id='asset1',
            ...     coordinates=[[-122.4194, 37.7749]],
            ...     features={'type': 'building'}
            ... ))
            >>> _ = inventory.add_asset('asset2', Asset(
            ...     asset_id='asset2',
            ...     coordinates=[[-74.0060, 40.7128]],
            ...     features={'type': 'bridge'}
            ... ))

            >>> # Get extent with default buffer:
            >>> extent_default = inventory.get_extent(buffer='default')
            >>> print(extent_default.bounds)
            (-122.4196, 37.7748, -74.0058, 40.712900000000005)

            >>> # Get extent with no buffer:
            >>> extent_none = inventory.get_extent(buffer='none')
            >>> print(extent_none.bounds)
            (-122.4194, 37.7749, -74.006, 40.7128)

            >>> # Get extent with a custom buffer:
            >>> extent_custom = inventory.get_extent(
            ...     buffer=[0.1, 0.1, 0.1, 0.1]
            ... )
            >>> print(extent_custom.bounds)
            (-122.51939999999999, 37.6749, -73.906, 40.8128)
        """
        # Check buffer input:
        error_msg = (
            'Invalid buffer input. Valid options for the buffer input'
            "are 'default', 'none', or a list of 4 integers."
        )

        if isinstance(buffer, str):
            if buffer.lower() == 'default':
                buffer_levels = [0.0002, 0.0001, 0.0002, 0.0001]
            elif buffer.lower() == 'none':
                buffer_levels = [0, 0, 0, 0]
            else:
                raise ValueError(error_msg)
        elif (
            isinstance(buffer, list)
            and len(buffer) == 4
            and all(isinstance(x, (int, float)) for x in buffer)
        ):
            buffer_levels = buffer.copy()
        else:
            raise ValueError(error_msg)

        # Determine the geographical extent of the inventory
        valid_geoms = [asset.get_geometry() for asset in self.inventory.values()]
        valid_geoms = [g for g in valid_geoms if g and not g.is_empty]

        if not valid_geoms:
            return box(0, 0, 0, 0)

        min_x, min_y, max_x, max_y = GeometryCollection(valid_geoms).bounds

        # Return a Shapely polygon of the bounding box:
        return box(
            min_x - buffer_levels[0],
            min_y - buffer_levels[1],
            max_x + buffer_levels[2],
            max_y + buffer_levels[3],
        )

    def get_geojson(self) -> dict[str, Any]:
        """
        Generate a GeoJSON representation of the assets in the inventory.

        The function constructs a valid GeoJSON ``FeatureCollection``, where
        each asset is represented as a ``Feature``. Each feature contains:

        - A ``geometry`` field defining a ``Point``, ``LineString``, or
          ``Polygon`` based on the asset's coordinates.
        - A ``properties`` field containing asset-specific metadata.

        Additionally, the GeoJSON output includes:

        - A timestamp indicating when the data was created.
        - The ``BRAILS`` package version (if available).
        - A Coordinate Reference System (``crs``) definition set to ``CRS84``.

        Returns:
            dict:
                A dictionary formatted as a GeoJSON ``FeatureCollection``
                containing all assets in the inventory.

        Note:
            Assets without geometry are excluded from the generated GeoJSON
            representation.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset1',
            ...     coordinates=[[-122.4194, 37.7749]]
            ... )
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset2',
            ...     coordinates=[[-74.0060, 40.7128], [-73.935242, 40.730610]]
            ... )
            >>> inventory_geojson = inventory.get_geojson()
            >>> print(inventory_geojson)
            {'type': 'FeatureCollection',
             'generated': '2025-08-11 02:49:47.520953',
             'brails_version': '4.0',
             'crs': {'type': 'name',
                     'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
             'features': [{'type': 'Feature', 'properties': {},
                           'geometry': {'type': 'Point',
                                        'coordinates': [-122.4194, 37.7749]}},
                          {'type': 'Feature', 'properties': {},
                           'geometry': {
                               'type': 'LineString',
                               'coordinates': [[-74.006, 40.7128],
                                               [-73.935242, 40.73061]]
                               }
                           }
                          ]
             }
        """
        try:
            brails_version = version('BRAILS')
        except PackageNotFoundError:
            brails_version = 'NA'

        geojson = {
            'type': 'FeatureCollection',
            'generated': str(datetime.now(timezone.utc)),
            'brails_version': brails_version,
            'crs': {
                'type': 'name',
                'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
            },
            'features': [],
        }

        # Initialize list to track assets with invalid geometries
        failed_ids = []

        for key, asset in self.inventory.items():
            geometry_obj = asset.get_geometry()

            # Only store valid geometries
            if geometry_obj and not geometry_obj.is_empty:
                properties = asset.features | {'id': key}

                geojson['features'].append(
                    {
                        'type': 'Feature',
                        'properties': properties,
                        'geometry': mapping(geometry_obj),
                    }
                )
            else:
                # Track failures
                failed_ids.append(str(key))

        # Report Failures
        if failed_ids:
            print(
                f'\nWarning: {len(failed_ids)} assets were skipped due to invalid or unknown geometry.'
            )
            print(f'Skipped IDs: {", ".join(failed_ids)}')

        return geojson

    def join(
        self,
        inventory_to_join: AssetInventory,
        method: str = 'GetPointsInPolygons',
    ) -> None:
        """
        Join with another AssetInventory using specified spatial join method.

        Args:
            inventory_to_join (AssetInventory):
                The inventory to be joined with the current one.
            method (str):
                The spatial join method to use. Defaults to
                ``GetPointsInPolygons``. The ``method`` defines how the join
                operation is executed between inventories.

        Raises:
            TypeError: If ``inventory_to_join`` is not an instance of
                :class:`AssetInventory` or if ``method`` is not a string.

        Returns:
            None:
                This method modifies the :class:`AssetInventory` instance in
                place.

        Example:
            >>> polygon1_asset = Asset('polygon1', [
            ...     [-122.40, 37.75],
            ...     [-122.40, 37.76],
            ...     [-122.39, 37.76],
            ...     [-122.39, 37.75],
            ...     [-122.40, 37.75]
            ... ])
            >>> polygon2_asset = Asset('polygon2', [
            ...     [-122.38, 37.77],
            ...     [-122.38, 37.78],
            ...     [-122.37, 37.78],
            ...     [-122.37, 37.77],
            ...     [-122.38, 37.77]
            ... ])
            >>> poly_inventory = AssetInventory()
            >>> _ = poly_inventory.add_asset(
            ...     asset_id='polygon1',
            ...     asset=polygon1_asset
            ... )
            >>> _ = poly_inventory.add_asset(
            ...     asset_id='polygon2',
            ...     asset=polygon2_asset
            ... )
            >>> poly_inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  polygon1 Asset:
                Coordinates:  [[-122.4, 37.75], [-122.4, 37.76],
                               [-122.39, 37.76], [-122.39, 37.75],
                               [-122.4, 37.75]]
                Features:  {}
            Key:  polygon2 Asset:
                Coordinates:  [[-122.38, 37.77], [-122.38, 37.78],
                               [-122.37, 37.78], [-122.37, 37.77],
                               [-122.38, 37.77]]
                Features:  {}
            >>> # This point lies within polygon1's boundaries:
            >>> point_inventory = AssetInventory()
            >>> _ = point_inventory.add_asset(
            ...     asset_id = 'point1',
            ...     asset = Asset(
            ...         'point1',
            ...         [[-122.395, 37.755]],
            ...         features={'FFE':6.8}
            ...     )
            ... )
            >>> poly_inventory.join(point_inventory)
            Joining inventories using GetPointsInPolygons method...
            Identified a total of 1 matched points.
            Inventories successfully joined.
            >>> poly_inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  polygon1 Asset:
                Coordinates:  [[-122.4, 37.75], [-122.4, 37.76],
                               [-122.39, 37.76], [-122.39, 37.75],
                               [-122.4, 37.75]]
                Features:  {'FFE': 6.8}
            Key:  polygon2 Asset:
                Coordinates:  [[-122.38, 37.77], [-122.38, 37.78],
                               [-122.37, 37.78], [-122.37, 37.77],
                               [-122.38, 37.77]]
                Features:  {}
        """
        # Ensure inventory_to_join is of type AssetInventory:
        if not isinstance(inventory_to_join, AssetInventory):
            raise TypeError(
                'Inventory input specified for join needs to be an AssetInventory'
            )

        # Ensure method is a valid string:
        if not isinstance(method, str):
            raise TypeError('Join method should be a valid string')

        # Perform the spatial join using the specified method:
        self = SpatialJoinMethods.execute(method, self, inventory_to_join)

    def print_info(self) -> None:
        """
        Print summary information about the AssetInventory.

        This method outputs the name of the class , the type of data structure
        used to store the inventory, and basic information about each asset
        in the inventory, including its ``key`` and ``features``.

        Returns:
            None

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(
            ...     asset_id='building1',
            ...     asset=Asset(
            ...         'building1',
            ...         [[-122.40, 37.75], [-122.40, 37.76],
            ...          [-122.39, 37.76], [-122.39, 37.75],
            ...          [-122.40, 37.75]],
            ...         features={'height': 12.5}
            ...     )
            ... )
            >>> _ = inventory.add_asset(
            ...     asset_id='building2',
            ...     asset=Asset(
            ...         'building2',
            ...         [[-122.38, 37.77], [-122.38, 37.78],
            ...          [-122.37, 37.78], [-122.37, 37.77],
            ...          [-122.38, 37.77]],
            ...         features={'height': 8.0}
            ...     )
            ... )
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  building1 Asset:
                Coordinates:  [[-122.4, 37.75], [-122.4, 37.76],
                               [-122.39, 37.76], [-122.39, 37.75],
                               [-122.4, 37.75]]
                Features:  {'height': 12.5}
            Key:  building2 Asset:
                Coordinates:  [[-122.38, 37.77], [-122.38, 37.78],
                               [-122.37, 37.78], [-122.37, 37.77],
                               [-122.38, 37.77]]
                Features:  {'height': 8.0}
        """
        print(self.__class__.__name__)
        print('Inventory stored in: ', self.inventory.__class__.__name__)
        for key, asset in self.inventory.items():
            print('Key: ', key, 'Asset:')
            asset.print_info()

    def read_from_geojson(  # noqa: C901
        self,
        file_path: str,
        asset_type: str = 'building',
        id_column: str | None = None,
        *,
        key_by_unique_id: bool = False,
    ) -> bool:
        """
        Reads a GeoJSON file and imports assets into the asset inventory.

        This method loads a GeoJSON FeatureCollection, validates its
        structure, checks geometries, and converts each asset_data into a
        BRAILS `Asset` object.

        It also supports mapping asset IDs from the GeoJSON file's properties
        (using `id_column`), auto-generating new numeric IDs, or keying assets
        by the `unique_id` property (using `key_by_unique_id`).

        Args:
            file_path (str):
                Path to the GeoJSON file to be read. Must represent a valid
                GeoJSON FeatureCollection.
            asset_type (str, optional):
                The 'type' assigned to all imported assets, or with
                `key_by_unique_id` only to those without a 'type'.
                Defaults to `'building'`.
            id_column (str, optional):
                The name of the asset_data-level or properties key containing
                the unique asset ID. If `None` (default), IDs are
                auto-generated sequentially.
            key_by_unique_id (bool, optional):
                If `True`, key each asset by its `unique_id` property, which
                reads back a file written by `write_to_geojson` with
                `integer_ids=True`. The keys are ints when every `unique_id`
                is ASCII digits without a leading zero (`'0'` included), the
                rule under which that export orders numerically, and strings
                otherwise. The property is not kept as a feature,
                and each asset keeps its own `type`. Cannot be combined with
                `id_column`. Defaults to `False`.

        Returns:
            bool:
                ``True`` if the GeoJSON file is successfully read and assets
                are added to the inventory.

        Raises:
            FileNotFoundError:
                If the specified `file_path` does not exist or is not a file.
            json.JSONDecodeError:
                If the file is not valid JSON.
            ValueError:
                If the file is invalid JSON, not a valid GeoJSON
                FeatureCollection, or contains no features. With
                `key_by_unique_id`, also if `id_column` is given, or if a
                feature's `unique_id` value is missing or empty, is shared
                with another feature, or already keys an asset in the
                inventory.

        Example:
            Example GeoJSON file (`seattle_buildings.geojson`):

            .. code-block:: json

                {
                  "type": "FeatureCollection",
                  "features": [
                    {
                      "type": "Feature",
                      "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                          [
                            [-122.3355, 47.6080],
                            [-122.3350, 47.6080],
                            [-122.3350, 47.6085],
                            [-122.3355, 47.6085],
                            [-122.3355, 47.6080]
                          ]
                        ]
                      },
                      "properties": {
                        "name": "Building A",
                        "height_m": 18.7
                      }
                    },
                    {
                      "type": "Feature",
                      "geometry": {
                        "type": "Point",
                        "coordinates": [-122.3321, 47.6062]
                      },
                      "properties": {
                        "name": "Building B",
                        "height_m": 12.4
                      }
                    }
                  ]
                }

            Example usage:

                >>> inv = AssetInventory()
                >>> inv.read_from_geojson(
                ...     'seattle_buildings.geojson',
                ...     asset_type='building',
                ...     id_column='name'
                ... )
                True
                >>> inv.print_info()
                AssetInventory
                Inventory stored in:  dict
                Key:  Building A Asset:
                     Coordinates:  [[-122.3355, 47.608], [-122.335, 47.608],
                 [-122.335, 47.6085], [-122.3355, 47.6085],
                 [-122.3355, 47.608]]
                     Features:  {'name': 'Building A', 'height_m': 18.7,
                 'type': 'building'}
                Key:  Building B Asset:
                     Coordinates:  [[-122.3321, 47.6062]]
                     Features:  {'name': 'Building B', 'height_m': 12.4,
                 'type': 'building'}

        Note:
            All geometries are expected to follow the GeoJSON standard
            ``[longitude, latitude]`` order and use the WGS-84 geographic
            coordinate reference system (EPSG:4326).
        """
        # Path checks:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f'File not found: {file_path}')

        # JSON parsing:
        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON format in {file_path}.') from e

        # GeoJSON structure validation:
        if not isinstance(data, dict) or data.get('type') != 'FeatureCollection':
            raise ValueError('Input file is not a valid GeoJSON FeatureCollection.')

        features = data.get('features', [])
        if not isinstance(features, list) or not features:
            raise ValueError('GeoJSON FeatureCollection contains no valid features.')

        if key_by_unique_id:
            if id_column is not None:
                raise ValueError(
                    'key_by_unique_id cannot be combined with '
                    f'id_column={id_column!r}.'
                )
            # Check every feature before any asset is added, so a missing,
            # empty or repeated unique_id, or one that already keys an asset,
            # leaves the inventory as it was.
            seen_unique_ids = set()
            unique_ids = []
            for i, asset_data in enumerate(features):
                if (
                    not isinstance(asset_data, dict)
                    or asset_data.get('type') != 'Feature'
                ):
                    continue
                feature_properties = asset_data.get('properties')
                unique_id = (
                    feature_properties.get('unique_id')
                    if isinstance(feature_properties, dict)
                    else None
                )
                if unique_id is None or str(unique_id) == '':
                    raise ValueError(
                        f"Feature at index {i} has no 'unique_id' value."
                    )
                unique_id = str(unique_id)
                if unique_id in seen_unique_ids:
                    raise ValueError(
                        f"'unique_id' value {unique_id!r} occurs "
                        'in more than one feature.'
                    )
                seen_unique_ids.add(unique_id)
                unique_ids.append(unique_id)
            integer_keys = all(
                _is_plain_digit_string(unique_id) for unique_id in unique_ids
            )
            for unique_id in unique_ids:
                if (int(unique_id) if integer_keys else unique_id) in self.inventory:
                    raise ValueError(
                        f"'unique_id' value {unique_id!r} already "
                        'keys an asset in the inventory.'
                    )

        id_counter = 1
        for i, asset_data in enumerate(features):
            # Validate asset_data structure
            if not isinstance(asset_data, dict):
                print(
                    f'Warning: Skipping invalid asset data at index {i} '
                    f'(not a dictionary)'
                )
                continue

            if asset_data.get('type') != 'Feature':
                print(
                    f'Warning: Skipping asset data at index {i} '
                    f"(type is not 'Feature')"
                )
                continue

            # Parse Geometry
            try:
                geometry = GeoTools.parse_geojson_geometry(
                    asset_data.get('geometry')
                )
            except (TypeError, ValueError, NotImplementedError, IndexError) as e:
                # Catch-all for any geometry parsing error
                print(
                    f'Warning: Skipping asset data at index {i} '
                    f'(invalid geometry structure: {e})'
                )
                continue

            if not geometry:
                print(
                    f'Warning: Skipping asset data at index {i} '
                    f'(unsupported or empty geometry)'
                )
                continue

            # Get properties and add asset_type
            properties = asset_data.get('properties', {})
            if not isinstance(properties, dict):
                print(
                    f'Warning: Asset data at index {i} has invalid properties '
                    f'(not a dictionary), using empty properties'
                )
                properties = {}

            # With key_by_unique_id, each feature keeps its own type:
            if not key_by_unique_id or _is_blank_type(properties.get('type')):
                properties['type'] = asset_type

            # Determine asset ID
            if key_by_unique_id:
                asset_id = str(properties.pop('unique_id'))
                if integer_keys:
                    asset_id = int(asset_id)
            else:
                # Use id_column if provided, otherwise default to "id"
                id_key = id_column if id_column else 'id'

                if id_key in asset_data:
                    asset_id = asset_data[id_key]
                elif id_key in properties:
                    asset_id = properties[id_key]
                else:
                    # Use auto-generated ID
                    asset_id = id_counter
                    id_counter += 1

                # Make sure numeric IDs are represented as ints to avoid
                # duplicates
                if isinstance(asset_id, str) and asset_id.isdigit():
                    asset_id = int(asset_id)

            # Make sure the asset_id is hashable
            if not isinstance(asset_id, Hashable):
                print(
                    f'Warning: Skipping asset data at index {i}. '
                    f"The asset ID '{asset_id}' is not a hashable type "
                    '(e.g., it might be a list or dict).'
                )
                continue

            # Create and add the asset
            asset = Asset(asset_id, geometry, properties)
            success = self.add_asset(asset_id, asset)
            if not success:
                print(
                    f'Warning: Asset with ID {asset_id} already exists, '
                    f'skipping asset data at index {i}'
                )

        return True

    def remove_asset(self, asset_id: str | int) -> bool:
        """
        Remove an asset from the inventory.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.

        Returns:
            bool: ``True`` if asset was removed, ``False`` otherwise.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(
            ...     asset_id='building1',
            ...     asset=Asset(
            ...         'building1',
            ...         [[-122.40, 37.75], [-122.40, 37.76],
            ...          [-122.39, 37.76], [-122.39, 37.75],
            ...          [-122.40, 37.75]],
            ...         features={'height': 12.5}
            ...     )
            ... )
            >>> inventory.remove_asset('building1')
            True
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
        """
        if asset_id in self.inventory:
            del self.inventory[asset_id]
            return True
        return False

    def remove_features(self, features_to_remove: Iterable[str]) -> bool:
        """
        Remove specified features from all assets in the inventory.

        Args:
            features_to_remove(Iterable[str]):
                An iterable of feature keys to remove from each :class:`Asset`.
                Accepts types such as ``list``, ``tuple``, ``dict_keys``, etc.

        Returns:
            bool:
                ``True`` if at least one feature was removed from any asset,
                ``False`` otherwise.

        Raises:
            TypeError: If ``features_to_remove`` is not an iterable of strings.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(
            ...     asset_id='building1',
            ...     asset=Asset(
            ...         'building1',
            ...         [[-122.40, 37.75], [-122.40, 37.76],
            ...          [-122.39, 37.76], [-122.39, 37.75],
            ...          [-122.40, 37.75]],
            ...         features={'height': 12.5, 'floors': 3}
            ...     )
            ... )
            >>> _ = inventory.add_asset(
            ...     asset_id='building2',
            ...     asset=Asset(
            ...         'building2',
            ...         [[-122.38, 37.77], [-122.38, 37.78],
            ...          [-122.37, 37.78], [-122.37, 37.77],
            ...          [-122.38, 37.77]],
            ...         features={'height': 8.0, 'floors': 2}
            ...     )
            ... )
            >>> inventory.remove_features(['floors'])
            True
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  building1 Asset:
                Coordinates:  [[-122.4, 37.75], [-122.4, 37.76],
                               [-122.39, 37.76], [-122.39, 37.75],
                               [-122.4, 37.75]]
                Features:  {'height': 12.5}
            Key:  building2 Asset:
                Coordinates:  [[-122.38, 37.77], [-122.38, 37.78],
                               [-122.37, 37.78], [-122.37, 37.77],
                               [-122.38, 37.77]]
                Features:  {'height': 8.0}
        """
        if not isinstance(features_to_remove, Iterable) or not all(
            isinstance(f, str) for f in features_to_remove
        ):
            raise TypeError('features_to_remove must be an iterable of strings.')

        results = [
            asset.remove_features(features_to_remove)
            for asset in self.inventory.values()
        ]
        return any(results)

    def remove_nonmatching_assets(self, image_set, verbose=False) -> None:
        """
        Remove assets that do not have corresponding entries in image set.

        This method compares asset keys in the inventory with those in the
        provided ImageSet and removes any asset whose key does not exist in
        the ImageSet.

        Args:
            image_set(ImageSet):
                The image set containing valid image keys.
            verbose(bool, optional):
                If ``True``, prints a summary of removed keys. Default is
                ``False``.

        Modifies:
            self.inventory(dict):
                Removes nonmatching asset entries directly from the inventory.
                The object is updated in-place.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(
            ...     asset_id='house_A',
            ...     asset=Asset(
            ...         'house_A',
            ...         [[-118.5123, 34.0451], [-118.5120, 34.0454],
            ...          [-118.5117, 34.0452], [-118.5121, 34.0449],
            ...          [-118.5123, 34.0451]],
            ...         features={'type': 'residential', 'floors': 2}
            ...     )
            ... )
            >>> _ = inventory.add_asset(
            ...     asset_id='warehouse_B',
            ...     asset=Asset(
            ...         'warehouse_B',
            ...         [[-118.5105, 34.0467], [-118.5102, 34.0471],
            ...          [-118.5099, 34.0468], [-118.5103, 34.0464],
            ...          [-118.5105, 34.0467]],
            ...         features={'type': 'commercial', 'floors': 1}
            ...     )
            ... )
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  house_A Asset:
                     Coordinates:  [[-118.5123, 34.0451], [-118.512, 34.0454],
            [-118.5117, 34.0452], [-118.5121, 34.0449], [-118.5123, 34.0451]]
                     Features:  {'type': 'residential', 'floors': 2}
            Key:  warehouse_B Asset:
                     Coordinates:  [[-118.5105, 34.0467], [-118.5102, 34.0471],
            [-118.5099, 34.0468], [-118.5103, 34.0464], [-118.5105, 34.0467]]
                     Features:  {'type': 'commercial', 'floors': 1}
            >>> img_set = ImageSet()
            >>> img1 = Image('bldg1.jpg')
            >>> img2 = Image('bldg2.jpg')
            >>> _ = img_set.add_image('house_A', img1)
            >>> _ = img_set.add_image('house_B', img2)
            >>> inventory.remove_nonmatching_assets(img_set, verbose=True)
            Removed 1 nonmatching assets: ['warehouse_B']
        """
        # Collect keys from both sets
        footprint_keys = set(self.inventory.keys())
        imageset_keys = set(image_set.images.keys())

        # Identify keys missing in image set
        missing_keys = footprint_keys - imageset_keys

        # Remove unmatched assets (in-place modification)
        for key in missing_keys:
            self.remove_asset(key)

        # Optionally, print a summary of removed assets
        if verbose:
            print(
                f'Removed {len(missing_keys)} nonmatching assets: '
                f'{sorted(missing_keys)}'
            )

    def write_to_geojson(
        self,
        output_file: str = '',
        *,
        integer_ids: bool = False,
        asset_type: str | None = None,
    ) -> dict:
        """
        Write inventory to a GeoJSON file and return the GeoJSON dictionary.

        This method generates a GeoJSON representation of the asset inventory,
        writes it to the specified file path (if provided), and returns the
        GeoJSON object.

        Args:
            output_file(str, optional):
                Path to the output GeoJSON file. If empty, no file is written.
                The file is written under a temporary name in the same
                directory, which then replaces the output file, so a failed
                write leaves an existing file as it was. A NaN is written as
                ``null``, except inside a tuple, such as a coordinate pair,
                whose values are written as stored (see ``clean_floats``).
            integer_ids (bool, optional):
                If ``True``, write the form the R2D tool stack reads, which
                requires integer ids. The features are numbered from 1 within
                each ``type``, and each carries its number as the
                feature-level ``id`` and its asset id, as a string, in the
                property ``unique_id``. They are ordered by ``type`` and then
                by asset id, numerically when every asset id, as a string, is
                ASCII digits without a leading zero (``'0'`` included), and
                as strings otherwise. Assets without geometry are skipped.
                Defaults to ``False``.
            asset_type (str, optional):
                With ``integer_ids``, the ``type`` written for an asset whose
                ``type`` is missing, ``None``, NaN or an empty string, the
                rule ``read_from_geojson`` follows for ``asset_type`` with
                ``key_by_unique_id``. An asset with a ``type`` keeps it, and
                the inventory is not changed. Defaults to ``None``, under
                which such an asset is refused. It is refused without
                ``integer_ids``, whose features are written as the inventory
                holds them, and when it is blank or names a missing value
                (``'nan'``, ``'none'`` or ``'null'``, in any case).

        Returns:
            dict:
                A dictionary containing the GeoJSON ``FeatureCollection``.

        Raises:
            ValueError:
                If ``output_file`` is given and the inventory holds an
                infinite value outside a tuple, in which case no file is
                opened. With ``integer_ids``, also if a written asset already
                carries a ``unique_id`` or ``AIM_id`` feature, has no
                ``type`` or a ``type`` that is ``None``, NaN or an empty
                string while ``asset_type`` is ``None``, or if two asset ids
                have the same string form. Also if ``asset_type`` is given
                without ``integer_ids``, or is not a string that names a type.

        Examples:
            Define an AssetInventory consisting of a single asset.

            >>> inventory = AssetInventory()
            >>> inventory.add_asset(
            ...     asset_id='asset1',
            ...     asset=Asset(
            ...         'asset1',
            ...         [
            ...             [-122.40, 37.75],
            ...             [-122.40, 37.76],
            ...             [-122.39, 37.76],
            ...             [-122.39, 37.75],
            ...             [-122.40, 37.75]
            ...         ],
            ...         features={'name': 'Building A', 'roofHeight': 22.6}
            ...     )
            ... )

            Write the AssetInventory data into a GeoJSON dictionary.

            >>> geojson_dict = inventory.write_to_geojson()
            >>> print(geojson_dict['features'])
            [{'type': 'Feature', 'properties': {'name': 'Building A',
            'roofHeight': 22.6}, 'geometry': {'type': 'Polygon', 'coordinates':
            [[[-122.4, 37.75], [-122.4, 37.76], [-122.39, 37.76],
            [-122.39, 37.75], [-122.4, 37.75]]]}, 'id': '0'}]

            Write the AssetInventory data in a GeoJSON dictionary and a file
            named  ``output.geojson``.

            >>> geojson_written = inventory.write_to_geojson('output.geojson')
            Wrote 1 asset to {../output.geojson}
        """
        if asset_type is not None:
            if not integer_ids:
                raise ValueError(
                    'asset_type applies only with integer_ids=True. Without '
                    'it, set the type of the assets in the inventory.'
                )
            if not isinstance(asset_type, str) or asset_type.strip().lower() in (
                '',
                'nan',
                'none',
                'null',
            ):
                raise ValueError(
                    'asset_type must be a string that names a type, such as '
                    f"'building', not {asset_type!r}."
                )

        geojson = self.get_geojson()

        if integer_ids:
            if asset_type is not None:
                # The properties are copies made by get_geojson, so the
                # inventory keeps its own values.
                for feature in geojson['features']:
                    if _is_blank_type(feature['properties'].get('type')):
                        feature['properties']['type'] = asset_type
            geojson['features'] = _number_features_by_type(geojson['features'])
        else:
            # Ensure each feature has its asset id, as a string, as its
            # top-level 'id'.
            for index, feature in enumerate(geojson['features']):
                feature_id = feature['properties'].pop('id', str(index))
                feature['id'] = str(feature_id)

        # If a file name is provided, write the created GeoJSON dictionary into
        # a GeoJSON file:
        if output_file:
            _write_geojson_output_file(output_file, geojson)
            num_elements = len(geojson['features'])
            element = 'asset' if num_elements == 1 else 'assets'
            print(
                f'Wrote {num_elements} {element} to {os.path.abspath(output_file)}'
            )
        return geojson

    def read_from_csv(
        self,
        file_path: str,
        keep_existing: bool,
        str_type: str = 'building',
        id_column: str | None = None,
    ) -> bool:
        """
        Read inventory data from a CSV file and add it to the inventory.

        Args:
            file_path(str):
                  The path to the CSV file
            keep_existing(bool):
                  If ``False``, the inventory will be initialized
            str_type(str):
                  ``building`` or ``bridge``
            id_column(str):
                  The name of column that contains id values. If ``None``, new
                  indices will be assigned

        Returns:
            bool:
                  ``True`` if assets were added, ``False`` otherwise.
        """

        def is_float(element: Any) -> bool:
            # If you expect None to be passed:
            if element is None:
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False

        if keep_existing:
            if len(self.inventory) == 0:
                print(
                    'No existing inventory found. Reading in the new '
                    'inventory from the file.'
                )
                id_counter = 1
            else:
                # we don't want a duplicate the id
                id_counter = max(self.inventory.keys()) + 1
        else:
            self.inventory = {}
            id_counter = 1

        # Attempt to open the file
        try:
            with open(file_path) as csvfile:
                csv_reader = csv.DictReader(csvfile)
                rows = list(csv_reader)
        except FileNotFoundError:
            raise Exception(f'The file {file_path} does not exist.')

        # Check if latitude/longitude exist
        lat = ['latitude', 'lat']
        lon = ['longitude', 'lon', 'long']
        key_names = csv_reader.fieldnames
        lat_id = [i for i, y in enumerate(key_names) if y.lower() in lat]
        lon_id = [i for i, x in enumerate(key_names) if x.lower() in lon]
        if len(lat_id) == 0:
            raise Exception(
                "The key 'Latitude' or 'Lat' (case insensitive) not found. "
                'Please specify the building coordinate.'
            )
        if len(lon_id) == 0:
            raise Exception(
                "The key 'Longitude' or 'Lon' (case insensitive) not found. "
                'Please specify the building coordinate.'
            )
        lat_key = key_names[lat_id[0]]
        lon_key = key_names[lon_id[0]]

        for bldg_features in rows:
            for i, key in enumerate(bldg_features):
                # converting to a number
                val = bldg_features[key]
                if val.isdigit():
                    bldg_features[key] = int(val)
                elif is_float(val):
                    bldg_features[key] = float(val)

            # coordinates = [[bldg_features[lat_key], bldg_features[lon_key]]]
            coordinates = [[bldg_features[lon_key], bldg_features[lat_key]]]

            bldg_features.pop(lat_key)
            bldg_features.pop(lon_key)

            # TODO: Avoid hardcoding types here; consider using dynamic type
            # handling or type hints instead.
            if 'type' in bldg_features.keys():
                if bldg_features['type'] not in ['building', 'bridge']:
                    raise Exception(
                        f"The csv file {file_path} cannot have a column named 'type'"
                    )
            else:
                bldg_features['type'] = str_type

            # is the id provided by user?
            if id_column is None:
                # if not we assin the id
                id = id_counter
            else:
                if id_column not in bldg_features.keys():
                    raise Exception(
                        f"The key '{id_column}' not found in {file_path}"
                    )
                id = bldg_features[id_column]

            asset = Asset(id, coordinates, bldg_features)
            self.add_asset(id, asset)
            id_counter += 1

        return True

    def add_asset_features_from_csv(
        self, file_path: str, id_column: str | None
    ) -> bool:
        """
        Read inventory data from a CSV file and add it to the inventory.

        Args:
            file_path(str):
                  The path to the CSV file
            id_column(str):
                  The name of column that contains id values. If ``None``, new
                  indices will be assigned

        Returns:
            bool: ``True`` if assets were added, ``False`` otherwise.
        """
        try:  # Attempt to open the file
            with open(file_path) as csvfile:
                csv_reader = csv.dictReader(csvfile)
                rows = list(csv_reader)
        except FileNotFoundError:
            raise Exception(f'The file {csvfile} does not exist.')

        for bldg_features in rows:
            for i, key in enumerate(bldg_features):
                # converting to number
                val = bldg_features[key]
                if val.isdigit():
                    bldg_features[key] = int(val)
                elif InputValidator.is_float(val):
                    bldg_features[key] = float(val)

            if id_column not in bldg_features.keys():
                raise Exception(f"The key '{id_column}' not found in {file_path}")
            id = bldg_features[id_column]

            self.add_asset_features(id, bldg_features)

        return True

    def get_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Convert the asset inventory into two structured DataFrames and count.

        This method processes the internal asset inventory and returns:

        - A ``DataFrame`` containing the features of each asset, with support
          for multiple possible worlds (realizations).
        - A ``DataFrame`` containing centroid coordinates (longitude and
          latitude) for spatial operations.
        - The total number of assets in the inventory.

        The method flattens feature lists into separate columns if multiple
        possible worlds exist. It also derives centroid coordinates from the
        geometry for each asset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, int]:

                - Asset feature ``DataFrame`` (indexed by asset ID).
                - Asset geometry ``DataFrame`` with 'Lat' and 'Lon' columns.
                - Total number of assets (int).

        Raises:
            ValueError:
                If a feature list's length does not match the expected number
                of possible worlds.
        """
        n_possible_worlds = self.get_n_pw()

        # Lists to hold data for DataFrame construction
        ids = []
        lats = []
        lons = []
        feature_rows = []

        # Single pass through the inventory to collect all data
        for asset_id, asset in self.inventory.items():
            ids.append(asset_id)

            # Geometry: Reuse the robust logic in Asset.get_centroid()
            centroid_data = asset.get_centroid()[0]
            lons.append(centroid_data[0])
            lats.append(centroid_data[1])

            # Features: Collect for processing below
            features = asset.features.copy()
            features['index'] = asset_id
            feature_rows.append(features)

        nbldg = len(ids)

        # Process Features considering Possible Worlds
        if n_possible_worlds == 1:
            bldg_properties_df = pd.DataFrame(feature_rows)
        else:
            # Identify columns that are lists (vector columns) in ANY asset
            # These are the ones with multiple worlds
            vector_columns = set()
            for entry in feature_rows:
                vector_columns.update(
                    key for key, value in entry.items() if isinstance(value, list)
                )

            flat_data = []
            for entry in feature_rows:
                # Base row with scalar values
                row = {
                    key: value
                    for key, value in entry.items()
                    if key not in vector_columns
                }

                # Expand vector columns
                for col in vector_columns:
                    val = entry.get(col)

                    # Safety Check: Handle mixed types.
                    # If this asset has a scalar for a column that is a vector
                    # elsewhere, we replicate that scalar across all worlds.
                    if not isinstance(val, list):
                        for i in range(n_possible_worlds):
                            row[f'{col}_{i + 1}'] = val
                        continue

                    # Validate length
                    if len(val) != n_possible_worlds:
                        raise ValueError(
                            f'Asset {entry["index"]}: The specified # of '
                            f'possible worlds is {n_possible_worlds} but '
                            f"'{col}' contains {len(val)} realizations."
                        )

                    # Flatten
                    for i in range(n_possible_worlds):
                        row[f'{col}_{i + 1}'] = val[i]

                flat_data.append(row)

            bldg_properties_df = pd.DataFrame(flat_data)

        # Cleanup Feature DataFrame
        if 'type' in bldg_properties_df.columns:
            bldg_properties_df = bldg_properties_df.drop(columns=['type'])

        bldg_properties_df = bldg_properties_df.set_index('index')

        # --- Construct Geometry DataFrame ---
        bldg_geometries_df = pd.DataFrame({'Lat': lats, 'Lon': lons, 'index': ids})
        bldg_geometries_df = bldg_geometries_df.set_index('index')

        return bldg_properties_df, bldg_geometries_df, nbldg

    def get_world_realization(self, id: int = 0) -> AssetInventory:
        """
        Extract a single realization (possible world) from an inventory.

        This method builds a new inventory of the same class with
        ``n_pw`` 1 in which every feature holding multiple possible worlds
        is reduced to its value in world ``id``. Other features are carried
        over. A list or dict value is copied one level deep. Any other
        value, and any container nested inside the realization's list or
        dict value, is shared with this inventory. Coordinates and the
        housing-unit inventory are copied.

        Args:
            id(int, default=0):
                The index of the realization to extract. Must be within the
                range of available possible worlds(0-based indexing). When
                ``n_pw`` is 1 and no feature holds more than one
                realization, every index returns world 0.

        Returns:
            AssetInventory:
                A new inventory object representing the selected realization.

        Raises:
            Exception: If ``id`` is not 0, ``n_pw`` is 1 and a feature
                       holds more than one realization.
            Exception: If any feature has fewer realizations than the specified
                       ``id``.
        """
        single = self.n_pw == 1
        world = 0 if single else id

        realization = type(self)(vector_features=self.vector_features)
        realization.housing_unit_inventory = deepcopy(self.housing_unit_inventory)

        for i, asset in self.inventory.items():
            features = {}
            for key, val in asset.features.items():
                if not is_probabilistic_feature(key, val, self.vector_features):
                    value = val
                elif single and id != 0 and len(val) > 1:
                    errmsg = (
                        f'Cannot retrieve world {id}: the inventory has n_pw '
                        f'1, but asset id {i}, feature {key} holds {len(val)} '
                        'realizations.'
                    )
                    raise Exception(errmsg)
                elif len(val) > world:
                    value = val[world]
                elif len(val) == world:
                    errmsg = (
                        f'The world index {world} should be smaller than the '
                        f'existing number of worlds {len(val)}, as the '
                        'index starts from zero.'
                    )
                    raise Exception(errmsg)
                else:
                    errmsg = (
                        f'The world index {world} should be smaller than the '
                        f'existing number of worlds, e.g. asset id {i}, '
                        f'feature {key} contains only {len(val)} '
                        'realizations.'
                    )
                    raise Exception(errmsg)
                features[key] = (
                    copy(value) if isinstance(value, (list, dict)) else value
                )

            new_asset = copy(asset)
            new_asset.coordinates = deepcopy(asset.coordinates)
            new_asset.features = features
            realization.inventory[i] = new_asset

        return realization

    def add_world(self, realization: AssetInventory) -> None:  # noqa: C901
        """
        Add a world to the inventory from an inventory of that world alone.

        Called on an inventory without assets, ``add_world`` takes the
        realization as world 0. It copies each of the realization's
        :class:`Asset` objects, with a feature dict of its own, in ascending
        order of the asset id as a string, and takes the realization's
        vector features. Each later realization becomes the next world, and
        :attr:`n_pw` counts the worlds added. A later realization must hold
        the same asset ids at the same coordinates as world 0. A feature
        keeps one value while every world agrees and becomes a list of one
        value per world where one differs, and a key a realization lacks
        counts as ``None`` there. The inventory shares coordinates and
        feature values with the realizations, so a caller must not change a
        realization after adding it.

        Values are compared with ``==``. Two float NaN are equal, arrays are
        compared with ``numpy.array_equal``, and values ``==`` cannot compare
        are unequal. Since the inventory holds one value per world as a list, a
        realization may hold a list only as the value of one of the world's
        vector features, and there not a list whose first item is a list.
        World 0's vector features are those its realization declares, and a
        later world's are the inventory's and its realization's. Realizations
        without assets leave the inventory without assets and at one world.

        Args:
            realization (AssetInventory):
                The world to add, an inventory of one possible world.

        Raises:
            TypeError:
                If ``realization`` is not an :class:`AssetInventory`.
            ValueError:
                If the realization has more than one possible world; if it
                holds a list as the value of a feature that is not one of the
                world's vector features, or a list whose first item is a list
                as the value of one that is; if its asset ids or coordinates
                differ from those of world 0; or if a vector feature, of the
                inventory or of the realization, whose world 0 value is not a
                list takes a different value in the realization or is already
                held as one value per world. The message names up to
                :data:`MAX_REPORTED_WORLD_PROBLEMS` assets. A refused
                realization changes nothing.
        """
        if not isinstance(realization, AssetInventory):
            msg = (
                'A realization must be an AssetInventory, not '
                f'{type(realization).__name__}.'
            )
            raise TypeError(msg)
        if realization.n_pw != 1:
            msg = f'A realization must have one world, not {realization.n_pw}.'
            raise ValueError(msg)
        vector = list(
            dict.fromkeys([*self.vector_features, *realization.vector_features])
        )
        # World 0 keeps the realization's vector features alone, so its lists
        # are checked against those.
        checked = vector if self.inventory else list(realization.vector_features)
        listing = sorted(
            (
                (asset_id, key)
                for asset_id, asset in realization.inventory.items()
                for key, value in asset.features.items()
                if is_probabilistic_feature(key, value, checked)
            ),
            key=lambda pair: (_merge_id_order(pair[0]), pair[1]),
        )
        if listing:
            listing = [
                f'asset {asset_id!r}, feature {key!r}' for asset_id, key in listing
            ]
            msg = (
                f'A realization holds a list as the value of {len(listing)} '
                'feature(s) that are not vector features, or a list of lists '
                'as the value of a vector feature: '
                f'{_report_first_problems(listing, "; ")}.'
            )
            raise ValueError(msg)
        if not self.inventory:
            for asset_id in sorted(realization.inventory, key=_merge_id_order):
                record = copy(realization.inventory[asset_id])
                record.features = dict(record.features)
                self.inventory[asset_id] = record
            self.n_pw = 1
            self.vector_features = list(realization.vector_features)
            return

        world = self.n_pw
        held, given = self.inventory.keys(), realization.inventory.keys()
        missing = sorted(held - given, key=_merge_id_order)
        extra = sorted(given - held, key=_merge_id_order)
        if missing or extra:
            problems = [f'{asset_id!r} missing' for asset_id in missing] + [
                f'{asset_id!r} not in world 0' for asset_id in extra
            ]
            msg = (
                f'World {world} holds assets other than those of world 0: '
                f'{_report_first_problems(problems, ", ")}.'
            )
            raise ValueError(msg)
        moved = [
            asset_id
            for asset_id in held
            if realization.inventory[asset_id].coordinates
            != self.inventory[asset_id].coordinates
        ]
        if moved:
            moved = [repr(i) for i in sorted(moved, key=_merge_id_order)]
            msg = (
                f'World {world} holds {len(moved)} asset(s) with coordinates '
                f'other than in world 0: {_report_first_problems(moved, ", ")}.'
            )
            raise ValueError(msg)

        def listed(key: str, value: object) -> bool:
            """
            Say whether a held value is a list of one value per world added.

            Args:
                key (str):
                    The feature the value belongs to.
                value (object):
                    The value the inventory holds for the feature.

            Returns:
                bool:
                    ``True`` if the value holds one value for each world
                    added so far.
            """
            return (
                world > 1
                and is_probabilistic_feature(key, value, self.vector_features)
                and len(value) == world
            )

        # A vector feature held per world must hold a list in world 0, since
        # a list of per-world values is distinguished from a vector value by
        # its first item. A feature already held per world before a
        # realization declares it a vector is checked through its world 0
        # entry.
        ambiguous = []
        for asset_id in given:
            features = self.inventory[asset_id].features
            for key in vector:
                value = features.get(key)
                per_world = listed(key, value)
                first = value[0] if per_world else value
                if not isinstance(first, list) and (
                    per_world
                    or not _values_agree(
                        first, realization.inventory[asset_id].features.get(key)
                    )
                ):
                    ambiguous.append((asset_id, key))
        if ambiguous:
            ambiguous = [
                f'asset {asset_id!r}, feature {key!r}'
                for asset_id, key in sorted(
                    ambiguous,
                    key=lambda pair: (_merge_id_order(pair[0]), pair[1]),
                )
            ]
            msg = (
                f'World {world} gives a new value to {len(ambiguous)} vector '
                'feature(s) whose world 0 value is not a list, so their values '
                'per world could not be told apart from one vector value: '
                f'{_report_first_problems(ambiguous, "; ")}.'
            )
            raise ValueError(msg)

        for asset_id, asset in realization.inventory.items():
            features = asset.features
            held_features = self.inventory[asset_id].features
            for key in [
                *held_features,
                *(key for key in features if key not in held_features),
            ]:
                value = features.get(key)
                current = held_features.get(key)
                # A new list, so that no list the caller holds is extended.
                if listed(key, current):
                    held_features[key] = [*current, value]
                elif not _values_agree(current, value):
                    held_features[key] = [current] * world + [value]
        self.vector_features = vector
        self.n_pw = world + 1

    def update_world_realization(
        self, id: int, world_realization: AssetInventory
    ) -> None:
        """
        Update the current AssetInventory with a specific world realization.

        This method integrates feature values from a single-world
        ``world_realization`` inventory into the current multi-world inventory
        by updating the realization at the specified index ``id``.

        Args:
            id(int):
                The index of the world(realization) to update. Must be less
                than ``self.n_pw``.
            world_realization(AssetInventory):
                An :class:`AssetInventory` instance representing a single
                realization of the world. All features in this inventory must
                be scalar (i.e., not lists).

        Raises:
            Exception:
                - If the specified ``id`` is not within the valid range of
                  realizations.
                - If ``world_realization`` contains features with multiple
                  realizations.
        """
        if self.n_pw == id:
            errmsg = (
                f'The world index {id} should be smaller than the existing '
                f'number of worlds {self.n_pw}, and the index starts from '
                'zero.'
            )
            raise Exception(errmsg)

        if self.n_pw < id:
            errmsg = (
                f'The world index {id} should be smaller than the existing '
                f'number of worlds {self.n_pw}.'
            )
            raise Exception(errmsg)

        for i in world_realization.get_asset_ids():
            flag, features = self.get_asset_features(i)
            flag_new, features_new = world_realization.get_asset_features(i)

            for key, val in features_new.items():
                if isinstance(val, list):
                    errmsg = (
                        'world_realization should not contain multiple '
                        'possible worlds.'
                    )
                    raise Exception(errmsg)

                try:
                    original_value = self.get_asset_features(i)[1][key]
                except KeyError:
                    # create a new key-value pair if it did not exist.
                    original_value = val

                # initialize
                if isinstance(original_value, list):
                    new_value = original_value
                else:
                    new_value = [original_value] * self.n_pw

                # update
                new_value[id] = val

                # if identical, shrink it
                if len(set(new_value)) == 1:
                    new_value = new_value[0]

                # overwrite existing ones.
                self.add_asset_features(i, {key: new_value}, overwrite=True)

    def validate_possible_worlds(self) -> None:
        """
        Check every feature against the inventory's number of worlds.

        A feature value must be either a scalar, meaning one value shared
        by every possible world, or a list holding exactly
        :attr:`n_pw` realizations. Vector features are exempt unless they are
        lists of lists, since a plain list is a single value for those.

        Raises:
            BrailsError:
                If any feature holds a number of realizations other than
                :attr:`n_pw`. The message names the offending assets and
                features, up to
                :data:`MAX_REPORTED_WORLD_PROBLEMS` of them.

        Examples:
            >>> inventory = AssetInventory()
            >>> inventory.add_asset_coordinates('A1', [[-122.4, 37.8]])
            True
            >>> _ = inventory.add_asset_features(
            ...     'A1', {'StructureType': ['C1', 'C2']}
            ... )
            >>> inventory.validate_possible_worlds()

            A feature with too few realizations is rejected:

            >>> _ = inventory.add_asset_features('A1', {'DesignLevel': ['H']})
            >>> inventory.validate_possible_worlds()
            Traceback (most recent call last):
            brails.exceptions.BrailsError: ...
        """
        problems = []

        for asset_id in self.get_asset_ids():
            flag, features = self.get_asset_features(asset_id)

            for key, value in features.items():
                if not is_probabilistic_feature(
                    key, value, self.vector_features
                ):
                    continue
                if len(value) != self.n_pw:
                    problems.append((asset_id, key, len(value)))

        if not problems:
            return

        details = '  ' + _report_first_problems(
            [
                f'asset {asset_id}, feature {key!r}: {count} realizations'
                for asset_id, key, count in problems
            ],
            '\n  ',
        )
        if len(problems) > MAX_REPORTED_WORLD_PROBLEMS:
            details += '.'

        raise BrailsError(
            f'The inventory has {self.n_pw} possible worlds, but '
            f'{len(problems)} feature value(s) provide a different '
            'number of realizations. Every feature must be a scalar, '
            f'or a list of exactly {self.n_pw} realizations.\n{details}'
        )

    def get_n_pw(self) -> int:  # move to Asset
        """
        Get the number of possible worlds (realizations) in the inventory.

        Returns:
            int:
                The number of possible worlds stored in the inventory.
        """
        return self.n_pw

    def get_multi_keys(self) -> tuple[list[str], list[str]]:  # move to Asset
        """
        Identify features that contain multiple realizations across assets.

        Iterates through all assets and returns two lists:

        - Keys associated with multi-valued features(i.e., lists).
        - All unique feature keys present in the inventory.

        Returns:
            tuple[list[str], list[str]]:
                - A list of keys corresponding to multi-realization features.
                - A complete list of all unique feature keys in the inventory.
        """
        multi_keys = []
        all_keys = []

        for i in self.get_asset_ids():
            flag, features = self.get_asset_features(i)

            for key, val in features.items():
                if (
                    key not in multi_keys
                    and is_probabilistic_feature(
                        key, val, self.vector_features
                    )
                ):
                    multi_keys.append(key)
                if key not in all_keys:
                    all_keys.append(key)

        return multi_keys, all_keys

    def _get_next_numeric_id(self) -> int:
        """
        Compute the next available numeric asset ID in the inventory.

        Returns:
            int:
                The next available numeric ID (max numeric key + 1).
                Returns 0 if the inventory contains no numeric keys.

        Notes:
            - Non-numeric keys (e.g., 'A101') are ignored.
            - This function is typically used to generate sequential
              numeric identifiers for new assets.
        """
        numeric_ids = [int(k) for k in self.inventory if str(k).isdigit()]
        return (max(numeric_ids) + 1) if numeric_ids else 0

    def set_housing_unit_inventory(
        self,
        hu_inventory: HousingUnitInventory,
        hu_assignment: dict[str | int, list] | None = None,
        *,
        validate: bool = True,
    ) -> None:
        """
        Set the housing unit inventory and optionally assign housing units to assets.

        This method links a HousingUnitInventory object to the AssetInventory.
        It can also be used to assign housing unit ID lists to individual assets
        using the `hu_assignment` dictionary.

        This enables two primary workflows:
        1.  Assigning new housing units: Pass both `hu_inventory` and
            `hu_assignment`. The function will link the inventory and add
            the 'HousingUnits' feature to each asset in the assignment dict.
        2.  Linking a loaded inventory: After loading an AssetInventory
            (e.g., from GeoJSON) that already has 'HousingUnits' features,
            pass only the `hu_inventory` object. The function will link it
            and can optionally validate the existing assignments.

        Args:
            hu_inventory (HousingUnitInventory):
                The `HousingUnitInventory` object to link to this
                `AssetInventory`.
            hu_assignment (dict[str | int, list], optional):
                A dictionary mapping `asset_id` to a list of housing unit IDs.
                If provided, this list will be added as the 'HousingUnits'
                feature for each corresponding asset, overwriting any
                existing 'HousingUnits' feature. Defaults to `None`.
            validate (bool, optional):
                If `True`, run `validate_housing_unit_assignments()` after
                linking to check for mismatches. Defaults to `True`.

        Raises:
            TypeError: If `hu_inventory` is not an instance of
                       `HousingUnitInventory` or if `hu_assignment` is
                       provided and is not a `dict`.
        """
        if not isinstance(hu_inventory, HousingUnitInventory):
            raise TypeError(
                f'hu_inventory must be an instance of HousingUnitInventory, '
                f'not {type(hu_inventory)}.'
            )

        self.housing_unit_inventory = hu_inventory

        if hu_assignment is not None:
            if not isinstance(hu_assignment, dict):
                raise TypeError(
                    f'hu_assignment must be a dictionary or None, '
                    f'not {type(hu_assignment)}.'
                )

            print(f'Assigning housing units to {len(hu_assignment)} assets...')
            assets_not_found = 0
            for asset_id, hu_list in hu_assignment.items():
                success = self.add_asset_features(
                    asset_id, {'HousingUnits': hu_list}, overwrite=True
                )
                if not success:
                    assets_not_found += 1

            if assets_not_found > 0:
                print(
                    f'Warning: {assets_not_found} asset(s) listed in '
                    f'hu_assignment were not found in the inventory.'
                )

        if validate:
            print('Validating housing unit assignments...')
            self.validate_housing_unit_assignments()
            print('Validation successful.')

    def validate_housing_unit_assignments(self) -> bool:
        """
        Check the integrity of housing unit assignments.

        This function performs two main checks:
        1.  If any assets have a 'HousingUnits' feature, it checks that
            `self.housing_unit_inventory` is not `None`.
        2.  If `self.housing_unit_inventory` is set, it checks that every
            housing unit ID in every asset's 'HousingUnits' list exists in the
            `self.housing_unit_inventory`.

        Returns:
            bool:
                `True` if all assignments are valid or if there are
                no assignments.

        Raises:
            TypeError:
                If an asset's 'HousingUnits' feature is not a `list`.
            LookupError:
                If assets have 'HousingUnits' assignments but
                `self.housing_unit_inventory` is `None`.
            ValueError:
                If any housing unit IDs assigned to assets are not found
                in the `self.housing_unit_inventory`.
        """
        all_asset_hu_ids = set()
        found_assignments = False
        assets_with_bad_data = []

        # First, collect all housing unit IDs from assets
        for asset_id, asset in self.inventory.items():
            hu_data = asset.features.get('HousingUnits')
            if hu_data is not None:
                found_assignments = True
                if not isinstance(hu_data, list):
                    assets_with_bad_data.append(asset_id)
                else:
                    all_asset_hu_ids.update(hu_data)

        # Check for non-list 'HousingUnits' features
        if assets_with_bad_data:
            raise TypeError(
                f"The 'HousingUnits' feature must be a list. Found invalid "
                f'data in assets: {assets_with_bad_data}'
            )

        # If no assets have assignments, we are in a valid state.
        if not found_assignments:
            return True

        # If we have assignments, we MUST have an inventory linked.
        if self.housing_unit_inventory is None:
            raise LookupError(
                'Assets have housing unit assignments, but no '
                'HousingUnitInventory is linked. Call '
                'set_housing_unit_inventory() first.'
            )

        # Check for orphan IDs
        valid_hu_ids = set(self.housing_unit_inventory.get_housing_unit_ids())
        missing_ids = all_asset_hu_ids - valid_hu_ids

        if missing_ids:
            sample = list(missing_ids)[:10]
            raise ValueError(
                f'Found {len(missing_ids)} orphan housing unit IDs assigned to '
                f'assets that are not in the HousingUnitInventory. '
                f'Example orphans: {sample}'
            )

        # All checks passed
        return True

    def remove_housing_unit_inventory(
        self, *, clear_assignments: bool = True
    ) -> None:
        """
        Remove the linked housing unit inventory and clear all assignments.

        This method removes the `HousingUnitInventory` object from the
        `AssetInventory` by setting `self.housing_unit_inventory` to `None`.

        By default, it also removes the 'HousingUnits' feature from all
        assets in the inventory, as the assignments are specific to the
        inventory being removed. This ensures the `AssetInventory`
        remains in a valid state.

        Args:
            clear_assignments (bool, optional):
                If `True` (the default), this method will iterate through
                all assets and remove the 'HousingUnits' feature from them.
                If set to `False`, only the main `housing_unit_inventory`
                attribute will be removed, which will likely result
                in an invalid state that fails validation.
        """
        self.housing_unit_inventory = None
        print('HousingUnitInventory removed.')

        if clear_assignments:
            print("Removing 'HousingUnits' feature from all assets...")
            # Use the existing remove_features method
            self.remove_features(['HousingUnits'])
            print('Asset assignments cleared.')
        else:
            print(
                "Warning: Asset 'HousingUnits' features were not cleared. "
                'The inventory is likely in an invalid state.'
            )
