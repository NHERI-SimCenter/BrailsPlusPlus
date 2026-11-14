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
Freeze the output of each pipeline stage and record its digest.

.. autosummary::

      StageFreeze
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from brails.types.asset_inventory import AssetInventory, _merge_id_order

if TYPE_CHECKING:
    import os

DIGEST_ALGORITHM = 'blake2b-128'
"""Name of the digest algorithm."""

_DIGEST_SIZE = 16


class StageFreeze:
    """
    Hold the frozen output of each stage and the order the stages were frozen in.

    A stage is frozen once, under a name. The freeze holds the output by
    reference and :meth:`read` returns it, so a stage must not change a
    frozen output. A stage that needs to change one copies it first.
    :meth:`manifest` recomputes each held output's digest and raises if
    one no longer matches the digest recorded at freeze. Frozen outputs
    are held in memory only.

    Example:
        >>> from brails.types.asset_inventory import Asset, AssetInventory
        >>> observed = AssetInventory()
        >>> _ = observed.add_asset('a', Asset('a', [[-122.4, 37.7]]))
        >>> freeze = StageFreeze()
        >>> digest = freeze.freeze('observed', observed)
        >>> freeze.read('observed') is observed
        True
        >>> freeze.order
        ('observed',)
    """

    def __init__(self) -> None:
        """Create an empty freeze with no stage frozen."""
        self._outputs: dict[str, AssetInventory] = {}
        self._digests: dict[str, str] = {}
        self._order: list[str] = []

    @property
    def order(self) -> tuple[str, ...]:
        """Names of the frozen stages, in the order they were frozen."""
        return tuple(self._order)

    def freeze(self, stage: str, output: AssetInventory) -> str:
        """
        Freeze the output of a stage and return its digest.

        The output is held by reference (see :class:`StageFreeze`).

        Args:
            stage (str):
                Name of the stage. It must not be frozen already.
            output (AssetInventory):
                The stage's output.

        Returns:
            str:
                The hexadecimal digest of the frozen output.

        Raises:
            TypeError:
                If ``stage`` is not a string, ``output`` is not an
                :class:`AssetInventory`, or the output holds a value the
                digest cannot represent.
            ValueError:
                If ``stage`` is empty or already frozen.
        """
        if not isinstance(stage, str):
            msg = f'A stage name must be a string, not {type(stage).__name__}.'
            raise TypeError(msg)
        if not stage:
            msg = 'A stage name must not be empty.'
            raise ValueError(msg)
        if stage in self._outputs:
            msg = f'Stage {stage!r} is already frozen.'
            raise ValueError(msg)
        if not isinstance(output, AssetInventory):
            msg = (
                'A stage output must be an AssetInventory, not '
                f'{type(output).__name__}.'
            )
            raise TypeError(msg)

        digest = _inventory_digest(output)

        self._outputs[stage] = output
        self._digests[stage] = digest
        self._order.append(stage)
        return digest

    def read(self, stage: str) -> AssetInventory:
        """
        Return a frozen stage's output.

        Args:
            stage (str):
                Name of a frozen stage.

        Returns:
            AssetInventory:
                The held output itself (see :class:`StageFreeze`).

        Raises:
            LookupError:
                If the stage is not frozen.
        """
        self._require_frozen(stage)
        return self._outputs[stage]

    def digest(self, stage: str) -> str:
        """
        Return the digest recorded when a stage was frozen.

        Args:
            stage (str):
                Name of a frozen stage.

        Returns:
            str:
                The hexadecimal digest.

        Raises:
            LookupError:
                If the stage is not frozen.
        """
        self._require_frozen(stage)
        return self._digests[stage]

    def manifest(self) -> dict[str, Any]:
        """
        Return the recorded order and each digest.

        Each call recomputes the digest of every held output and builds a
        new dict.

        Returns:
            dict[str, Any]:
                ``{'digest_algorithm': ..., 'stages': [...]}``, where each
                entry of ``stages`` is a dict with keys ``stage`` and
                ``digest``, and the list is in the order the stages were
                frozen.

        Raises:
            RuntimeError:
                If a held output changed after it was frozen. The message
                names the first such stage.
        """
        for name in self._order:
            self._check_unchanged(name)
        return {
            'digest_algorithm': DIGEST_ALGORITHM,
            'stages': [
                {'stage': name, 'digest': self._digests[name]}
                for name in self._order
            ],
        }

    def save_manifest(self, path: str | os.PathLike[str]) -> None:
        """
        Write the manifest to a JSON file.

        Args:
            path (str or os.PathLike):
                File to write. An existing file is replaced.

        Raises:
            RuntimeError:
                If a held output changed after it was frozen, in which case
                nothing is written.
        """
        text = json.dumps(self.manifest(), indent=2)
        Path(path).write_text(text + '\n', encoding='utf-8')

    @staticmethod
    def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
        """
        Read a manifest written by :meth:`save_manifest`.

        Args:
            path (str or os.PathLike):
                File to read.

        Returns:
            dict[str, Any]:
                The manifest, in the form :meth:`manifest` returns.

        Raises:
            ValueError:
                If the file's ``digest_algorithm`` is not
                :data:`DIGEST_ALGORITHM`, it has no ``stages`` list, an entry
                lacks a string ``stage`` or a string ``digest``, or a stage
                appears twice.
        """
        manifest = json.loads(Path(path).read_text(encoding='utf-8'))
        _check_manifest(manifest, str(path))
        return manifest

    def _require_frozen(self, stage: str) -> None:
        """Raise ``LookupError`` unless ``stage`` is frozen."""
        if stage not in self._outputs:
            msg = f'Stage {stage!r} is not frozen. Frozen so far: {self._order}.'
            raise LookupError(msg)

    def _check_unchanged(self, stage: str) -> None:
        """
        Raise ``RuntimeError`` if a held output no longer has its digest.

        Args:
            stage (str):
                The frozen stage whose output is checked.

        Raises:
            RuntimeError:
                If the output can no longer be digested, or its digest
                differs from the one recorded at freeze.
        """
        try:
            current = _inventory_digest(self._outputs[stage])
        except TypeError as error:
            msg = (
                f'Stage {stage!r} changed after it was frozen: its output '
                f'can no longer be digested ({error})'
            )
            raise RuntimeError(msg) from error
        if current != self._digests[stage]:
            msg = (
                f'Stage {stage!r} changed after it was frozen: its digest '
                f'is {current}, not the {self._digests[stage]} recorded at '
                'freeze.'
            )
            raise RuntimeError(msg)


def _check_manifest(manifest: Any, source: str) -> None:  # noqa: ANN401
    """
    Raise ``ValueError`` unless ``manifest`` is a well-formed manifest.

    The checks are those :meth:`StageFreeze.load_manifest` lists.

    Args:
        manifest (Any):
            The parsed JSON.
        source (str):
            Where it came from, for the message.

    Raises:
        ValueError:
            If any of those checks fails.
    """
    # A malformed file is a bad value, as json's own decode error is, so it
    # raises ValueError where ruff's TRY004 asks for TypeError.
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get('stages'), list
    ):
        msg = f'{source} does not hold a stage manifest.'
        raise ValueError(msg)  # noqa: TRY004
    algorithm = manifest.get('digest_algorithm')
    if algorithm != DIGEST_ALGORITHM:
        msg = (
            f'{source} records the digest algorithm {algorithm!r}, not '
            f'{DIGEST_ALGORITHM!r}.'
        )
        raise ValueError(msg)
    seen: set[str] = set()
    for entry in manifest['stages']:
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get('stage'), str)
            and isinstance(entry.get('digest'), str)
        ):
            msg = f'{source} holds a malformed stage entry: {entry!r}.'
            raise ValueError(msg)  # noqa: TRY004
        name = entry['stage']
        if name in seen:
            msg = f'{source} records stage {name!r} more than once.'
            raise ValueError(msg)
        seen.add(name)


def _inventory_digest(inventory: AssetInventory) -> str:
    """
    Return the digest of an inventory's canonical JSON.

    Assets and keys are sorted, so the digest does not depend on insertion
    order.

    Args:
        inventory (AssetInventory):
            The inventory to digest.

    Returns:
        str:
            The hexadecimal digest.

    Raises:
        TypeError:
            If the inventory holds a value :func:`_canonical` refuses.
    """
    document: dict[str, Any] = {
        'assets': [
            _entry(asset_id, inventory.inventory[asset_id], 'asset')
            for asset_id in sorted(inventory.inventory, key=_merge_id_order)
        ],
        'n_pw': _canonical(inventory.n_pw, 'n_pw'),
        'vector_features': sorted(
            _canonical(list(inventory.vector_features), 'vector_features')
        ),
    }
    housing_units = inventory.housing_unit_inventory
    if housing_units is not None:
        document['housing_units'] = [
            _entry(unit_id, housing_units.inventory[unit_id], 'housing unit')
            for unit_id in sorted(housing_units.inventory, key=_merge_id_order)
        ]

    text = json.dumps(
        document, sort_keys=True, separators=(',', ':'), allow_nan=True
    )
    return hashlib.blake2b(
        text.encode('utf-8'), digest_size=_DIGEST_SIZE
    ).hexdigest()


def _entry(item_id: Any, item: Any, kind: str) -> dict[str, Any]:  # noqa: ANN401
    """
    Return the canonical entry of one asset or housing unit.

    Args:
        item_id (Any):
            The id it is keyed by.
        item (Any):
            The :class:`Asset` or :class:`HousingUnit`.
        kind (str):
            ``'asset'`` or ``'housing unit'``, for the error message.

    Returns:
        dict[str, Any]:
            ``id`` and ``features``, plus ``coordinates`` for an asset.
    """
    key = _canonical(item_id, f'a {kind} id')
    entry = {
        'id': key,
        'features': _canonical(item.features, f'the features of {kind} {key!r}'),
    }
    if kind == 'asset':
        entry['coordinates'] = _canonical(
            item.coordinates, f'the coordinates of {kind} {key!r}'
        )
    return entry


def _canonical(value: Any, where: str) -> Any:  # noqa: ANN401
    """
    Convert a value to the plain JSON types the digest serializes.

    Floats keep their value, so a NaN stays NaN and a float32 keeps its
    float32 rounding. An integer and a float of equal value stay distinct.

    Args:
        value (Any):
            The value to convert.
        where (str):
            What the value is, for the error message.

    Returns:
        Any:
            The value as ``None``, ``bool``, ``int``, ``float``, ``str``,
            ``list`` or ``dict`` with string keys.

    Raises:
        TypeError:
            If the value, or anything inside it, is of another type, or a
            mapping has a key that is not a string.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical(item, where) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = (
                    f'Cannot digest {where}: the key {key!r} has type '
                    f'{type(key).__name__}, not str.'
                )
                raise TypeError(msg)
            out[key] = _canonical(item, where)
        return out
    msg = (
        f'Cannot digest {where}: a value of type {type(value).__name__} '
        f'({value!r}) has no canonical form.'
    )
    raise TypeError(msg)
