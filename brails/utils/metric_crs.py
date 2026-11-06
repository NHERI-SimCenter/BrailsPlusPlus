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
Read projected coordinate reference systems that measure ground meters.

.. autosummary::

      meter_crs
"""

from __future__ import annotations

from pyproj import CRS
from pyproj.exceptions import CRSError


def meter_crs(value: object) -> CRS:
    """
    Read a projected CRS whose axes are in meters.

    Args:
        value (object):
            A CRS in any form ``pyproj`` accepts.

    Returns:
        pyproj.CRS:
            The CRS.

    Raises:
        ValueError:
            If the value is not a CRS, is not projected, has an axis in a
            unit other than the meter, or is Web Mercator (EPSG:3857).
    """
    try:
        crs = CRS.from_user_input(value)
    except CRSError as error:
        msg = f'projected_crs is not a CRS: {value!r}'
        raise ValueError(msg) from error
    units = {axis.unit_name for axis in crs.axis_info}
    if not crs.is_projected or not units <= {'metre', 'meter'}:
        msg = (
            'projected_crs must be a projected CRS in meters, not '
            f'{crs.name} ({crs.to_string()}), whose axes are in '
            f'{", ".join(sorted(units)) or "no unit"}.'
        )
        raise ValueError(msg)
    operation = crs.coordinate_operation
    if operation is not None and 'Pseudo Mercator' in operation.method_name:
        msg = (
            'projected_crs must measure ground distances, not '
            f'{crs.name} ({crs.to_string()}), a Web Mercator CRS whose '
            'meters are not ground meters away from the equator.'
        )
        raise ValueError(msg)
    return crs
