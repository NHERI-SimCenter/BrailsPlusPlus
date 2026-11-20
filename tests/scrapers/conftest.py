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


"""Helpers shared by the scraper tests: a region stand-in and a canned NSI API."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
import requests
from shapely.geometry import box

if TYPE_CHECKING:
    from collections.abc import Callable


class Region:
    """A minimal stand-in for RegionBoundary."""

    def __init__(self, polygon: object = None) -> None:
        """
        Hold the boundary polygon.

        Args:
            polygon (object, optional):
                The shapely polygon `get_boundary` returns. Defaults to a box from
                -0.01 to 0.01 in both coordinates.
        """
        self.polygon = box(-0.01, -0.01, 0.01, 0.01) if polygon is None else polygon

    def get_boundary(self) -> tuple:
        """Return the boundary polygon as RegionBoundary does."""
        return self.polygon, None, None


class FakeResponse:
    """A canned NSI API response."""

    def __init__(self, payload: dict) -> None:
        """
        Hold the payload the fake request returns.

        Args:
            payload (dict):
                The GeoJSON body `json()` serves.
        """
        self._payload = payload
        self.headers = {'Content-Type': 'application/json'}
        self.text = ''

    def raise_for_status(self) -> None:
        """Accept the response, as a 200 would."""

    def json(self) -> dict:
        """Return the canned payload."""
        return self._payload


def serve_features(monkeypatch: pytest.MonkeyPatch, features: list[dict]) -> None:
    """
    Serve these GeoJSON features from the NSI endpoint.

    The download itself, ``NSI_Parser._get_nsi_data``, still runs, so the records are
    keyed as in a real pull.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            The patcher.
        features (list[dict]):
            GeoJSON features, as the API serves them.
    """
    payload = {'features': features}
    monkeypatch.setattr(
        requests.Session,
        'get',
        lambda *_args, **_kwargs: FakeResponse(payload),
    )


def canned_response(monkeypatch: pytest.MonkeyPatch, records: list[dict]) -> None:
    """
    Serve these property records from the NSI endpoint.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            The patcher.
        records (list[dict]):
            Raw NSI property records, each placed at its ``x`` and ``y``.
    """
    serve_features(
        monkeypatch,
        [
            {
                'geometry': {
                    'type': 'Point',
                    'coordinates': [record['x'], record['y']],
                },
                'properties': record,
            }
            for record in records
        ],
    )


@pytest.fixture
def canned_fetch(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[dict]], None]:
    """
    Return a function that serves property records from the NSI endpoint.

    Returns:
        Callable[[list[dict]], None]:
            :func:`canned_response` with the test's patcher bound.
    """
    return functools.partial(canned_response, monkeypatch)
