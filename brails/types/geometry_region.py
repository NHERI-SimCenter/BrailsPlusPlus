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
A region given by a Shapely geometry.

.. autosummary::

      GeometryRegion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapely.geometry import MultiPolygon, Polygon

from brails.types.region_boundary import RegionBoundary

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


class GeometryRegion(RegionBoundary):
    """
    A region whose boundary is a polygon or multipolygon given in EPSG:4326.

    :meth:`get_boundary` returns the geometry as given, with its holes and all its
    parts, and looks nothing up.

    Args:
        geometry (shapely.geometry.Polygon or shapely.geometry.MultiPolygon):
            The boundary, in longitude and latitude.
        name (str, optional):
            The name scrapers print for the region. Defaults to ``'the region'``.

    Raises:
        TypeError:
            If ``geometry`` is neither a Polygon nor a MultiPolygon.
    """

    def __init__(
        self, geometry: Polygon | MultiPolygon, name: str = 'the region'
    ) -> None:
        """Keep the geometry and the region's name."""
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            msg = (
                'geometry must be a Shapely Polygon or MultiPolygon, not '
                f'{type(geometry).__name__}.'
            )
            raise TypeError(msg)
        self.geometry = geometry
        self.name = name

    def get_boundary(
        self,
        print_progress: bool = True,  # noqa: ARG002, FBT001, FBT002
    ) -> tuple[BaseGeometry, str, None]:
        """
        Return the geometry and the name of the region.

        Args:
            print_progress (bool, optional):
                Accepted for the signature of
                :meth:`~brails.types.region_boundary.RegionBoundary.get_boundary` and
                ignored: nothing is looked up, so nothing is printed.

        Returns:
            tuple[BaseGeometry, str, None]:
                The geometry as given, the name, and ``None`` in the place of the
                OpenStreetMap id a looked-up region returns.
        """
        return self.geometry, self.name, None
