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
Format asset ids for an error message.

An error about many assets names the first :data:`PREVIEW_LIMIT` and counts the rest,
so an inventory-wide failure does not print an inventory-wide message.

.. autosummary::

      join_preview
      preview_ids
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# How many assets an error message names before counting the rest.
PREVIEW_LIMIT = 10


def join_preview(entries: list[str], total: int) -> str:
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
    Format asset ids for an error message.

    Args:
        asset_ids (Iterable):
            The ids to show.

    Returns:
        str:
            A comma-separated preview of the first :data:`PREVIEW_LIMIT` with a count
            of any remainder.
    """
    ids = list(asset_ids)
    return join_preview(
        [str(asset_id) for asset_id in ids[:PREVIEW_LIMIT]], len(ids)
    )
