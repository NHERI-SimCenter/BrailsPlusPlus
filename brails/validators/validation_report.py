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
This module defines the report returned by inventory validators.

.. autosummary::

    ValidationReport
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

# Columns of the frame produced by ValidationReport.to_dataframe.
REPORT_COLUMNS = ('asset_id', 'world_index', 'path', 'message')


class ValidationReport:
    """
    The violations found when validating an AssetInventory.

    A report stores one list of messages per asset per possible world.
    Callers that want a simpler shape can use one of the views:
    :meth:`by_asset` collapses the worlds, :meth:`by_world` groups
    by world, and :meth:`to_dataframe` returns one row per message.

    An empty report means every asset satisfied the schema. The report
    follows container conventions, so it is **truthy when violations were
    found**. Use :attr:`is_valid` to ask whether validation passed.

    To import the :class:`ValidationReport` class, use:

    .. code-block:: python

        from brails.validators import ValidationReport
    """

    def __init__(
        self,
        dataset_id: str,
        n_pw: int = 1,
        n_assets: int = 0,
    ) -> None:
        """
        Initialize an empty report.

        Args:
            dataset_id (str):
                The DLML dataset the inventory was validated against.
            n_pw (int):
                The number of possible worlds that were validated.
            n_assets (int):
                The number of assets that were validated, whether or not
                they had violations.
        """
        self.dataset_id = dataset_id
        self.n_pw = n_pw
        self.n_assets = n_assets
        self._violations: dict[str | int, dict[int, list[str]]] = {}

    def add(
        self,
        asset_id: str | int,
        world_index: int,
        messages: Iterable[str],
    ) -> None:
        """
        Record the violations found for one asset in one world.

        Args:
            asset_id (str or int):
                The identifier of the offending asset.
            world_index (int):
                The index of the possible world the violations were found
                in.
            messages (Iterable[str]):
                The violation messages. An empty iterable is ignored, so
                that only offending assets appear in the report.
        """
        messages = list(messages)
        if not messages:
            return

        self._violations.setdefault(asset_id, {})[world_index] = messages

    @property
    def is_valid(self) -> bool:
        """
        Check whether the inventory satisfied the schema.

        Returns:
            bool:
                ``True`` if no violations were recorded.
        """
        return not self._violations

    def assets_with_violations(self) -> list[str | int]:
        """
        List the assets that had at least one violation.

        Returns:
            list[str | int]:
                The offending asset identifiers, in the order they were
                recorded.
        """
        return list(self._violations)

    def by_asset(self) -> dict[str | int, list[str]]:
        """
        Group the violations by asset, collapsing the possible worlds.

        Messages that repeat across worlds are reported once, in the
        order they were first seen. With a single world this is the shape
        returned by ``dlml.validate_assets``.

        Returns:
            dict[str | int, list[str]]:
                The violation messages of each offending asset.
        """
        collapsed: dict[str | int, list[str]] = {}

        for asset_id, worlds in self._violations.items():
            seen: list[str] = []
            for messages in worlds.values():
                seen.extend(
                    message for message in messages if message not in seen
                )
            collapsed[asset_id] = seen

        return collapsed

    def by_world(self) -> dict[int, dict[str | int, list[str]]]:
        """
        Group the violations by possible world.

        Returns:
            dict[int, dict[str | int, list[str]]]:
                For each world that had violations, the messages of each
                offending asset.
        """
        grouped: dict[int, dict[str | int, list[str]]] = {}

        for asset_id, worlds in self._violations.items():
            for world_index, messages in worlds.items():
                grouped.setdefault(world_index, {})[asset_id] = messages

        return grouped

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the violations as one row per message.

        The ``path`` column holds the location the violation was reported
        at, such as ``$.DesignLevel`` for a property whose value is wrong
        or ``$`` for a missing required property. This long form joins
        back to the inventory on ``asset_id``.

        Returns:
            pd.DataFrame:
                A frame with the columns ``asset_id``, ``world_index``,
                ``path`` and ``message``. Empty when the inventory is
                valid.
        """
        rows = [
            (asset_id, world_index, *_split_message(message))
            for asset_id, worlds in self._violations.items()
            for world_index, messages in worlds.items()
            for message in messages
        ]

        return pd.DataFrame(rows, columns=list(REPORT_COLUMNS))

    def __len__(self) -> int:
        """
        Count the assets that had at least one violation.

        Returns:
            int:
                The number of offending assets.
        """
        return len(self._violations)

    def __bool__(self) -> bool:
        """
        Check whether any violations were recorded.

        Returns:
            bool:
                ``True`` if the report is **not** empty. See
                :attr:`is_valid` for the opposite question.
        """
        return bool(self._violations)

    def __repr__(self) -> str:
        """
        Summarize the report.

        Returns:
            str:
                A short description of the outcome.
        """
        if self.is_valid:
            outcome = f'valid, {self.n_assets} assets'
        else:
            outcome = f'{len(self._violations)} of {self.n_assets} assets invalid'

        return (
            f'<ValidationReport {self.dataset_id!r}: {outcome}, '
            f'{self.n_pw} possible world(s)>'
        )


def _split_message(message: str) -> tuple[str, str]:
    """
    Split a violation message into its location and its text.

    DLML formats a violation as ``'<json_path>: <message>'``. A message
    without that prefix is returned unchanged, with an empty location.

    Args:
        message (str):
            The violation message.

    Returns:
        tuple[str, str]:
            The location the violation was reported at, and its text.
    """
    head, separator, tail = message.partition(': ')

    if not separator:
        return '', message

    return head, tail
