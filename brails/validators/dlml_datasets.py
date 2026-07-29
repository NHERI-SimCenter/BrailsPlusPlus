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
This module maps friendly method names to DLML dataset identifiers.

.. autosummary::

    resolve_dataset_id
"""

from __future__ import annotations

import dlml

from brails.exceptions import BrailsError

# Friendly names for the DLML datasets, so that a caller can write
# 'Hazus Earthquake - Buildings' instead of
# 'seismic/building/portfolio/Hazus v6.1'.
#
# These strings are copied verbatim from pelicun's
# `pelicun/resources/dlml_resource_paths.json`, so that a user moving
# between BRAILS and pelicun meets the same vocabulary. Only
# the datasets that ship an input schema are listed: an alias for a
# dataset without one would resolve to an identifier no validator can
# use. Extend this as DLML gains schemas.
METHOD_ALIASES = {
    'Hazus Earthquake - Buildings': 'seismic/building/portfolio/Hazus v6.1',
    'Hazus Hurricane': 'hurricane/building/portfolio/Hazus v5.1 coupled',
    'Hazus Hurricane Wind': 'hurricane/building/portfolio/Hazus v5.1 coupled',
    'Hazus Hurricane Wind - Buildings': (
        'hurricane/building/portfolio/Hazus v5.1 coupled'
    ),
    'Hazus Hurricane Wind - Buildings - Coupled': (
        'hurricane/building/portfolio/Hazus v5.1 coupled'
    ),
}


def resolve_dataset_id(name: str) -> str:
    """
    Resolve a method alias or dataset identifier to a dataset identifier.

    Args:
        name (str):
            A friendly method name listed in :data:`METHOD_ALIASES`, or a
            DLML dataset identifier such as
            ``'seismic/building/portfolio/Hazus v6.1'``.

    Returns:
        str:
            The DLML dataset identifier.

    Raises:
        BrailsError:
            If ``name`` is neither a known alias nor a DLML dataset
            identifier.

    Examples:
        >>> resolve_dataset_id('Hazus Earthquake - Buildings')
        'seismic/building/portfolio/Hazus v6.1'

        >>> resolve_dataset_id('seismic/building/portfolio/Hazus v6.1')
        'seismic/building/portfolio/Hazus v6.1'
    """
    if name in METHOD_ALIASES:
        return METHOD_ALIASES[name]

    if name in dlml.list_datasets():
        return name

    aliases = ', '.join(repr(alias) for alias in sorted(METHOD_ALIASES))
    raise BrailsError(
        f'{name!r} is not a method alias or a DLML dataset identifier. '
        f'Valid method aliases: {aliases}. Valid dataset identifiers are '
        'those returned by dlml.list_datasets().'
    )
