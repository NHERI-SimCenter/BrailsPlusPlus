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


"""Tests for the reading of projected CRSs in meters."""

from __future__ import annotations

import pytest

from brails.utils.metric_crs import meter_crs


class TestMeterCrs:
    """Only projected CRSs in ground meters are accepted."""

    def test_utm_accepted(self) -> None:
        """A UTM zone is returned as a CRS."""
        assert meter_crs('EPSG:32610').to_epsg() == 32610

    def test_geographic_refused(self) -> None:
        """A CRS in degrees raises."""
        with pytest.raises(ValueError, match='projected CRS in meters'):
            meter_crs('EPSG:4326')

    def test_web_mercator_refused(self) -> None:
        """Web Mercator raises."""
        with pytest.raises(ValueError, match='ground distances'):
            meter_crs('EPSG:3857')

    def test_not_a_crs(self) -> None:
        """A value pyproj cannot read raises."""
        with pytest.raises(ValueError, match='not a CRS'):
            meter_crs('not a crs')
