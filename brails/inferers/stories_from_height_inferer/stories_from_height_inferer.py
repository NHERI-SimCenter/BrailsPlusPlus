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

"""
Infer story counts from footprint heights by the scale of their source.

.. autosummary::

      StoriesFromHeightInferer
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from brails.constants import (
    DEFAULT_UNITS,
    FOOTPRINT_PROVIDER_FEATURE,
    STORIES_FROM_HEIGHT_KEY,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
    diagnosis_frame,
    preview_failures,
)
from brails.inferers.inference_engine import InferenceEngine
from brails.types.asset_inventory import _is_missing_value, is_probabilistic_feature
from brails.utils.unit_converter import UNIT_TO_METER, UnitConverter

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    from brails.types.asset_inventory import AssetInventory

# The feature naming a footprint's provider, which keys the scales.
PROVIDER_KEY = FOOTPRINT_PROVIDER_FEATURE

# The one provider whose scales may be given per source of the height: an
# Overture footprint names the source of its height in SOURCES_KEY, as the
# dataset of the entry whose ``property`` is HEIGHT_PROPERTY, else of the
# first entry without a ``property``.
OVERTURE_PROVIDER = 'overture'
SOURCES_KEY = 'sources'
HEIGHT_PROPERTY = '/properties/height'

# Height scales shipped with the package, described in the README beside it.
HEIGHT_SCALES_FILE = Path(__file__).with_name('height_scales.json')

# A ``key: `` pair of the Overture sources string.
_SOURCES_PAIR = re.compile(r'(?:^|, )([A-Za-z_]+): ')


class StoriesFromHeightInferer(InferenceEngine):
    """
    Infer each asset's story count from its footprint height.

    The count is ``max(1, floor(h / s + 0.5))`` for the height ``h`` and
    the scale ``s`` that applies to it, in one length unit, rounded half up
    and at least one. The scales are keyed by the footprint's provider, its
    :data:`PROVIDER_KEY` value. A provider's entry holds one scale for all
    its heights, or one scale per source of the height, a form only the
    :data:`OVERTURE_PROVIDER` entry may take. For a provider in that form
    the source is the ``dataset`` of the :data:`SOURCES_KEY` entry whose
    ``property`` is :data:`HEIGHT_PROPERTY`, else of the first entry without
    a ``property``. A footprint without a provider, or whose sources name no
    height source, gives no count. Neither does one whose provider or source
    the scales do not list, nor a count above the trusted maximum of its
    scale.

    The packaged scales calibrate the three sources of Overture's heights.
    Another provider's heights take an entry of their own in ``scales``.

    The inferer writes the count under ``stories_key`` for every asset and
    never reads what an asset holds there. Nothing is drawn.

    Every constructor argument except ``scales`` is stored as an attribute
    of the same name, read-only by convention.
    """

    def __init__(
        self,
        height_key: str = 'height',
        stories_key: str = STORIES_FROM_HEIGHT_KEY,
        length_unit: str = DEFAULT_UNITS['length'],
        scales: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    ) -> None:
        """
        Set up the inferer.

        Args:
            height_key (str, optional):
                Feature name the footprint height is read from.
            stories_key (str, optional):
                Feature name the height's count is written to. Defaults to
                :data:`STORIES_FROM_HEIGHT_KEY`.
            length_unit (str, optional):
                The unit of the height. Defaults to the length unit of
                :data:`brails.constants.DEFAULT_UNITS`.
            scales (Mapping[str, Any] | str | os.PathLike[str] | None, optional):
                ``None`` for :data:`HEIGHT_SCALES_FILE`, the path of a file
                of that form, or a mapping of that form: exactly the keys
                ``length_unit``, the unit of the scales, and ``providers``,
                a mapping from :data:`PROVIDER_KEY` value to an entry. An
                entry holds either ``scale`` and ``max_stories``, or
                ``sources``, a mapping from source to
                ``{'scale': ..., 'max_stories': ...}``, which only the
                :data:`OVERTURE_PROVIDER` entry may hold. ``max_stories``
                is ``None`` for no trusted maximum. Read once, here.
                Passing ``scales`` replaces the packaged file, and
                footprints of two providers whose heights differ in key or
                unit take two inferers.

        Raises:
            ValueError:
                If ``length_unit`` or the scales' ``length_unit`` is not a
                length unit, the scales hold other keys than those listed,
                an entry holds both forms or neither, ``sources`` is not a
                mapping or sits in an entry other than Overture's, a scale
                is not a positive finite number, or a trusted maximum is
                neither a positive integer nor ``None``.
        """
        unit = UnitConverter.normalize_unit(length_unit)
        if unit not in UNIT_TO_METER:
            msg = (
                f'{length_unit!r} is not a length unit; use one of '
                f'{sorted(UNIT_TO_METER)}.'
            )
            raise ValueError(msg)
        self.height_key = height_key
        self.stories_key = stories_key
        self.length_unit = length_unit
        self._meters_per_unit = UNIT_TO_METER[unit]
        self._scales = _load_scales(scales)

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Diagnose every asset without writing anything.

        An asset is ``OK`` when its height gives a count. A missing or
        unusable height, a missing or non-string provider, and sources that
        name no height source where the provider's scales are per source
        are invalid. A per-world height, provider or sources is
        unsupported, by the inventory's rule for list values: a list under
        :data:`SOURCES_KEY` holds one value per world unless the inventory
        declares that key a vector feature. A provider or source without a
        scale and a count above the trusted maximum are infeasible, with
        nothing to relax. What the asset holds under ``stories_key`` is not
        read.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.DataFrame:
                One row per asset id, with the columns of
                ``DIAGNOSIS_COLUMNS``.
        """
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            diagnosis, detail, _ = self._from_height(
                features, input_inventory.vector_features
            )
            rows[asset_id] = (diagnosis, detail, ())
        return diagnosis_frame(rows)

    def draw(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Compute the count of every asset, assuming all pass :meth:`check`.

        Args:
            input_inventory (AssetInventory):
                An inventory whose assets all passed :meth:`check`.

        Returns:
            pd.DataFrame:
                One row per asset id with the one column ``stories_key``,
                the height's count. Object dtype.

        Raises:
            ValueError:
                If an asset fails :meth:`check`, which means it was
                skipped.
        """
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            diagnosis, detail, count = self._from_height(
                features, input_inventory.vector_features
            )
            if diagnosis != DIAGNOSIS_OK:
                msg = (
                    f'Cannot infer {self.stories_key} for asset {asset_id}: '
                    f'{detail}. Screen the inventory with check() before '
                    'calling draw().'
                )
                raise ValueError(msg)
            rows[asset_id] = (count,)
        columns = [self.stories_key]
        if not rows:
            return pd.DataFrame(columns=columns, dtype=object)
        return pd.DataFrame.from_dict(
            rows, orient='index', columns=columns, dtype=object
        )

    def infer(self, input_inventory: AssetInventory) -> AssetInventory:
        """
        Write the height's count of every asset under ``stories_key``.

        Args:
            input_inventory (AssetInventory):
                The inventory to process. Every asset must pass
                :meth:`check`; to write only the assets that have a count,
                call :meth:`check`, then :meth:`draw` on those that pass.

        Returns:
            AssetInventory:
                A copy of the input with the counts of :meth:`draw` written,
                replacing any value held under ``stories_key``.

        Raises:
            ValueError:
                If any asset fails :meth:`check`, naming the assets and
                their diagnoses.
        """
        verdicts = self.check(input_inventory)
        failed = verdicts[verdicts['diagnosis'] != DIAGNOSIS_OK]
        if not failed.empty:
            reasons = preview_failures(failed['detail'])
            msg = (
                f'Cannot infer {self.stories_key} for {len(failed)} of '
                f'{len(verdicts)} assets: {reasons}. Remove them or supply '
                'the missing data, then call infer() again.'
            )
            raise ValueError(msg)
        counts = self.draw(input_inventory)
        output_inventory = deepcopy(input_inventory)
        for asset_id, count in counts[self.stories_key].items():
            output_inventory.add_asset_features(asset_id, {self.stories_key: count})
        return output_inventory

    def stories_of(self, features: Mapping[str, Any]) -> int | None:
        """
        Return the count one asset's height gives.

        Args:
            features (Mapping[str, Any]):
                The asset's features in one world, so a list under
                :data:`SOURCES_KEY` is read as that world's entries.

        Returns:
            int | None:
                The count, or ``None`` wherever :meth:`check` would not
                give ``OK``.
        """
        return self._from_height(features, (SOURCES_KEY,))[2]

    def _from_height(
        self, features: Mapping[str, Any], vector_features: Iterable[str]
    ) -> tuple[str, str, int | None]:
        """
        Apply the rule to an asset's height.

        Args:
            features (Mapping[str, Any]):
                The asset's features.
            vector_features (Iterable[str]):
                Names of the features whose values are vectors.

        Returns:
            tuple[str, str, int | None]:
                The diagnosis, what happened in words, and the count,
                ``None`` unless the diagnosis is ``OK``.
        """
        height = features.get(self.height_key)
        if is_probabilistic_feature(self.height_key, height, vector_features):
            return (
                DIAGNOSIS_UNSUPPORTED,
                f'{self.height_key} holds one value per world',
                None,
            )
        if (
            _is_missing_value(height)
            or isinstance(height, bool)
            or not isinstance(height, (int, float, np.integer, np.floating))
            or not math.isfinite(height)
            or height <= 0
        ):
            return DIAGNOSIS_INVALID, f'no usable {self.height_key}', None
        diagnosis, detail, scaled = self._scale_of(features, vector_features)
        if scaled is None:
            return diagnosis, detail, None
        scale, maximum, name = scaled
        # The provenance of the scales: README.rst beside this module, "The
        # packaged scales".
        count = max(1, math.floor(height * self._meters_per_unit / scale + 0.5))
        if maximum is not None and count > maximum:
            return (
                DIAGNOSIS_INFEASIBLE,
                f'{count} stories exceed the trusted maximum of {name!r}',
                None,
            )
        return DIAGNOSIS_OK, '', count

    def _scale_of(
        self, features: Mapping[str, Any], vector_features: Iterable[str]
    ) -> tuple[str, str, tuple[float, int | None, str] | None]:
        """
        Find the scale that applies to an asset's height.

        Args:
            features (Mapping[str, Any]):
                The asset's features.
            vector_features (Iterable[str]):
                Names of the features whose values are vectors.

        Returns:
            tuple[str, str, tuple[float, int | None, str] | None]:
                The diagnosis, what happened in words, and the scale in
                meters, its trusted maximum and the name of its provider or
                source, ``None`` unless the diagnosis is ``OK``.
        """
        provider = features.get(PROVIDER_KEY)
        if is_probabilistic_feature(PROVIDER_KEY, provider, vector_features):
            return (
                DIAGNOSIS_UNSUPPORTED,
                f'{PROVIDER_KEY} holds one value per world',
                None,
            )
        if _is_missing_value(provider):
            return DIAGNOSIS_INVALID, f'no {PROVIDER_KEY}', None
        if not isinstance(provider, str):
            return DIAGNOSIS_INVALID, f'{PROVIDER_KEY} is not a string', None
        if provider not in self._scales:
            return (
                DIAGNOSIS_INFEASIBLE,
                f'no scale for the {PROVIDER_KEY} {provider!r}',
                None,
            )
        by_source = self._scales[provider]
        if None in by_source:
            return DIAGNOSIS_OK, '', (*by_source[None], provider)
        value = features.get(SOURCES_KEY)
        if is_probabilistic_feature(SOURCES_KEY, value, vector_features):
            return (
                DIAGNOSIS_UNSUPPORTED,
                f'{SOURCES_KEY} holds one value per world',
                None,
            )
        # The dataset of the Overture sources entry that names the height
        # property, else of the first entry without a property.
        entries = _parse_sources(value)
        chosen = next(
            (e for e in entries if e.get('property') == HEIGHT_PROPERTY), None
        ) or next((e for e in entries if not e.get('property')), {})
        source = chosen.get('dataset')
        if not isinstance(source, str) or not source:
            return DIAGNOSIS_INVALID, f'no height source in {SOURCES_KEY}', None
        if source not in by_source:
            return (
                DIAGNOSIS_INFEASIBLE,
                f'no scale for the height source {source!r} of {provider!r}',
                None,
            )
        return DIAGNOSIS_OK, '', (*by_source[source], source)


def _load_scales(  # noqa: C901
    scales: Mapping[str, Any] | str | os.PathLike[str] | None,
) -> dict[str, dict[str | None, tuple[float, int | None]]]:
    """
    Read and check height scales keyed by provider.

    Args:
        scales (Mapping[str, Any] | str | os.PathLike[str] | None):
            The scales in any form the inferer's constructor accepts.

    Returns:
        dict[str, dict[str | None, tuple[float, int | None]]]:
            Provider to source to its scale converted to meters and its
            trusted maximum, with the one source ``None`` for a provider
            whose entry gives its scale directly.

    Raises:
        ValueError:
            In the scale cases the inferer's constructor lists.
    """
    if not isinstance(scales, Mapping):
        with Path(scales or HEIGHT_SCALES_FILE).open(encoding='utf-8') as file:
            scales = json.load(file)
    if set(scales) != {'length_unit', 'providers'} or not isinstance(
        scales['providers'], Mapping
    ):
        msg = (
            'The height scales must hold exactly length_unit and a providers '
            f'mapping, not the keys {sorted(scales)}.'
        )
        raise ValueError(msg)
    unit = scales['length_unit']
    unit = UnitConverter.normalize_unit(unit) if isinstance(unit, str) else unit
    if unit not in UNIT_TO_METER:
        msg = (
            f'The height scales give {scales["length_unit"]!r} as length_unit, '
            f'which is not one of {sorted(UNIT_TO_METER)}.'
        )
        raise ValueError(msg)
    direct = {'scale', 'max_stories'}
    loaded = {}
    for provider, entry in scales['providers'].items():
        keys = set(entry) if isinstance(entry, Mapping) else set()
        if 'sources' in keys and provider != OVERTURE_PROVIDER:
            msg = (
                f'Only the {OVERTURE_PROVIDER!r} entry may hold sources; no reader '
                f'names the source of the heights under {provider!r}.'
            )
            raise ValueError(msg)
        if 'sources' in keys and keys & direct:
            msg = (
                f'The scales entry of {provider!r} holds both sources and a '
                'scale; give one form.'
            )
            raise ValueError(msg)
        if keys not in (direct, {'sources'}):
            forms = ' or sources' if provider == OVERTURE_PROVIDER else ''
            msg = (
                f'The scales entry of {provider!r} must be a mapping holding '
                f'scale and max_stories{forms}.'
            )
            raise ValueError(msg)
        if 'sources' in keys and not isinstance(entry['sources'], Mapping):
            msg = (
                f'The sources of {provider!r} must be a mapping from source to '
                'scale and max_stories.'
            )
            raise ValueError(msg)
        loaded[str(provider)] = {}
        for source, pair in entry.get('sources', {None: entry}).items():
            name = (
                f'{provider!r}' if source is None else f'{source!r} of {provider!r}'
            )
            if not isinstance(pair, Mapping) or set(pair) != direct:
                msg = f'The scales entry of {name} must hold scale and max_stories.'
                raise ValueError(msg)
            scale, maximum = pair['scale'], pair['max_stories']
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(scale)
                or scale <= 0
            ):
                msg = f'The height scale of {name} must be a positive finite number.'
                raise ValueError(msg)
            if maximum is not None and (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum < 1
            ):
                msg = f'The trusted maximum of {name} must be a positive integer.'
                raise ValueError(msg)
            loaded[str(provider)][None if source is None else str(source)] = (
                float(scale) * UNIT_TO_METER[unit],
                maximum,
            )
    return loaded


def _parse_sources(value: object) -> list[dict[str, Any]]:
    """
    Read an Overture ``sources`` value into a list of entries.

    Args:
        value (object):
            The string the Overture scraper writes (``key: value`` pairs
            joined by ``, ``, entries joined by ``; ``), or a list or array
            of dictionaries.

    Returns:
        list[dict[str, Any]]:
            One dictionary per entry; empty for any other value, a missing
            one included.
    """
    if isinstance(value, str):
        if _is_missing_value(value):
            return []
        entries = []
        for text in value.split('; '):
            starts = list(_SOURCES_PAIR.finditer(text))
            entry = {}
            for position, match in enumerate(starts):
                end = (
                    starts[position + 1].start()
                    if position + 1 < len(starts)
                    else len(text)
                )
                entry[match.group(1)] = text[match.end() : end]
            entries.append(entry)
        return entries
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [entry for entry in value if isinstance(entry, Mapping)]
    return []
