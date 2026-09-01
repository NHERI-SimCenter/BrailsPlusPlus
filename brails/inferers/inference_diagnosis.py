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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.


"""
Report what an inferer can and cannot do with an asset.

An inferer that fails on individual assets says so per asset rather than
raising on the first one, so a caller can screen an inventory, keep the
reasons, and act on the assets that pass. These are the pieces of that
vocabulary that no peril owns.

.. autosummary::

      diagnosis_frame
      preview_failures
      preview_ids
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

# The verdict for an asset the inferer can process.
DIAGNOSIS_OK = 'OK'

# A driver is absent, or present with a value the inferer cannot use. The
# caller obtains the data, corrects it, or imputes a replacement.
DIAGNOSIS_INVALID = 'invalid'

# A driver is present and real, in a shape the inferer does not handle yet.
# The caller excludes the asset. Imputing over it would replace data the
# inventory holds with data the inferer preferred.
DIAGNOSIS_UNSUPPORTED = 'unsupported'

# Every driver is usable and the inferer still has nothing to produce. The
# caller relaxes one of the features in ``resolvable_by``.
DIAGNOSIS_INFEASIBLE = 'infeasible'

# The columns of a diagnosis frame. ``diagnosis`` is one of the four
# verdicts above, ``detail`` says what happened in words, and
# ``resolvable_by`` names the features that would each, alone, let the
# inferer produce a value. Only an infeasible asset has any.
DIAGNOSIS_COLUMNS = ('diagnosis', 'detail', 'resolvable_by')

# How many failing assets an error message names before counting the rest.
# An inventory-wide failure would otherwise print an inventory-wide message.
PREVIEW_LIMIT = 10


def diagnosis_frame(rows: dict) -> pd.DataFrame:
    """
    Build a diagnosis frame from per-asset verdicts.

    Args:
        rows (dict):
            Asset id to its diagnosis, detail, and resolvable features.

    Returns:
        pd.DataFrame:
            One row per asset id, with the columns of
            :data:`DIAGNOSIS_COLUMNS`.
    """
    if not rows:
        return pd.DataFrame(columns=list(DIAGNOSIS_COLUMNS), dtype=object)
    return pd.DataFrame.from_dict(
        rows, orient='index', columns=list(DIAGNOSIS_COLUMNS)
    )


def _truncate(entries: list[str], total: int) -> str:
    """
    Join entries for an error message, counting whatever is left out.

    Args:
        entries (list[str]):
            The entries to show, already cut to the limit.
        total (int):
            How many there were before the cut.

    Returns:
        str:
            A comma-separated preview with a count of any remainder.
    """
    shown = ', '.join(entries)
    extra = total - len(entries)
    if extra > 0:
        shown += f' and {extra} more'
    return shown


def preview_ids(asset_ids: Iterable) -> str:
    """
    Format failing asset ids for an error message.

    Args:
        asset_ids (Iterable):
            The ids to show.

    Returns:
        str:
            A comma-separated preview with a count of any remainder.
    """
    ids = list(asset_ids)
    return _truncate([str(asset_id) for asset_id in ids[:PREVIEW_LIMIT]],
                     len(ids))


def preview_failures(failures: pd.Series) -> str:
    """
    Format failing assets and their diagnoses for an error message.

    Args:
        failures (pd.Series):
            Diagnoses indexed by asset id.

    Returns:
        str:
            A comma-separated preview with a count of any remainder.
    """
    entries = [
        f'{asset_id} ({diagnosis})'
        for asset_id, diagnosis in list(failures.items())[:PREVIEW_LIMIT]
    ]
    return _truncate(entries, len(failures))
