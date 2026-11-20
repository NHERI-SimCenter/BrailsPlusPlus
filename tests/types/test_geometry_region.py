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

"""Tests for GeometryRegion in brails.types.geometry_region."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from brails.scrapers.nbi_scraper import nbi_scraper
from brails.scrapers.nti_scraper import nti_scraper
from brails.scrapers.usa_footprint_scraper import usa_footprint_scraper
from brails.types import GeometryRegion as ExportedRegion
from brails.types.geometry_region import GeometryRegion
from brails.utils import Importer

# Two squares near (-122.3, 37.9); the first has a square hole.
WITH_HOLE = Polygon(
    box(-122.30, 37.90, -122.28, 37.92).exterior.coords,
    [box(-122.295, 37.905, -122.285, 37.915).exterior.coords],
)
MULTIPOLYGON = MultiPolygon([WITH_HOLE, box(-122.26, 37.90, -122.25, 37.91)])


class PastTheCheck(Exception):  # noqa: N818
    """Raised where a scraper builds its ArcGIS helper, after its region check."""


pytestmark = pytest.mark.usefixtures('no_network')


def test_boundary_is_the_geometry_given() -> None:
    """get_boundary returns the geometry itself, the name and None."""
    geometry, name, osm_id = GeometryRegion(MULTIPOLYGON).get_boundary()
    assert geometry is MULTIPOLYGON
    assert name == 'the region'
    assert osm_id is None


def test_name_is_returned() -> None:
    """A name given is returned beside the geometry."""
    region = GeometryRegion(WITH_HOLE, name='Berkeley')
    assert region.get_boundary(print_progress=False)[1] == 'Berkeley'


@pytest.mark.parametrize(
    'geometry',
    [
        Point(-122.3, 37.9),
        LineString([(-122.3, 37.9), (-122.2, 37.8)]),
        (1, 2, 3, 4),
    ],
)
def test_geometry_that_is_not_polygonal_is_refused(geometry: object) -> None:
    """A point, a line or a tuple of coordinates raises TypeError."""
    with pytest.raises(TypeError, match='Polygon or MultiPolygon'):
        GeometryRegion(geometry)


@pytest.mark.parametrize(
    ('scraper', 'method'),
    [
        (nbi_scraper.NBIScraper(), 'get_assets'),
        (nti_scraper.NTIScraper(), 'get_assets'),
        (usa_footprint_scraper.USA_FootprintScraper({}), 'get_footprints'),
    ],
)
def test_scrapers_that_check_the_region_type_accept_it(
    monkeypatch: pytest.MonkeyPatch, scraper: object, method: str
) -> None:
    """The NBI, NTI and USA Structures scrapers take the region past their check."""

    def stop(*_args: object, **_kwargs: object) -> None:
        raise PastTheCheck

    # Each scraper builds its ArcGIS helper after the check.
    monkeypatch.setattr(f'{type(scraper).__module__}.ArcgisAPIServiceHelper', stop)
    with pytest.raises(PastTheCheck):
        getattr(scraper, method)(GeometryRegion(WITH_HOLE))


def test_class_is_exported_and_found_by_the_importer() -> None:
    """brails.types exports the class, and the Importer finds it by its name."""
    assert ExportedRegion is GeometryRegion
    assert Importer().get_class('GeometryRegion') is GeometryRegion
