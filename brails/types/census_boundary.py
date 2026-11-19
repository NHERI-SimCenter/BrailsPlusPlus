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
A region whose boundary is a census place or county.

.. autosummary::

      CensusBoundary
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import TYPE_CHECKING

import pandas as pd

from brails.types.region_boundary import RegionBoundary
from brails.utils.console import print_ascii_fallback
from brails.utils.tiger_files import (
    DEFAULT_TIGER_VINTAGE,
    read_tiger_file,
    tiger_file_url,
)

if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry

STATES: tuple[tuple[str, str, str], ...] = (
    ('01', 'AL', 'Alabama'),
    ('02', 'AK', 'Alaska'),
    ('04', 'AZ', 'Arizona'),
    ('05', 'AR', 'Arkansas'),
    ('06', 'CA', 'California'),
    ('08', 'CO', 'Colorado'),
    ('09', 'CT', 'Connecticut'),
    ('10', 'DE', 'Delaware'),
    ('11', 'DC', 'District of Columbia'),
    ('12', 'FL', 'Florida'),
    ('13', 'GA', 'Georgia'),
    ('15', 'HI', 'Hawaii'),
    ('16', 'ID', 'Idaho'),
    ('17', 'IL', 'Illinois'),
    ('18', 'IN', 'Indiana'),
    ('19', 'IA', 'Iowa'),
    ('20', 'KS', 'Kansas'),
    ('21', 'KY', 'Kentucky'),
    ('22', 'LA', 'Louisiana'),
    ('23', 'ME', 'Maine'),
    ('24', 'MD', 'Maryland'),
    ('25', 'MA', 'Massachusetts'),
    ('26', 'MI', 'Michigan'),
    ('27', 'MN', 'Minnesota'),
    ('28', 'MS', 'Mississippi'),
    ('29', 'MO', 'Missouri'),
    ('30', 'MT', 'Montana'),
    ('31', 'NE', 'Nebraska'),
    ('32', 'NV', 'Nevada'),
    ('33', 'NH', 'New Hampshire'),
    ('34', 'NJ', 'New Jersey'),
    ('35', 'NM', 'New Mexico'),
    ('36', 'NY', 'New York'),
    ('37', 'NC', 'North Carolina'),
    ('38', 'ND', 'North Dakota'),
    ('39', 'OH', 'Ohio'),
    ('40', 'OK', 'Oklahoma'),
    ('41', 'OR', 'Oregon'),
    ('42', 'PA', 'Pennsylvania'),
    ('44', 'RI', 'Rhode Island'),
    ('45', 'SC', 'South Carolina'),
    ('46', 'SD', 'South Dakota'),
    ('47', 'TN', 'Tennessee'),
    ('48', 'TX', 'Texas'),
    ('49', 'UT', 'Utah'),
    ('50', 'VT', 'Vermont'),
    ('51', 'VA', 'Virginia'),
    ('53', 'WA', 'Washington'),
    ('54', 'WV', 'West Virginia'),
    ('55', 'WI', 'Wisconsin'),
    ('56', 'WY', 'Wyoming'),
    ('60', 'AS', 'American Samoa'),
    ('66', 'GU', 'Guam'),
    ('69', 'MP', 'Commonwealth of the Northern Mariana Islands'),
    ('72', 'PR', 'Puerto Rico'),
    ('78', 'VI', 'United States Virgin Islands'),
)
"""FIPS code, USPS code and name of each row of the TIGER/Line state file."""

COUNTY_DESCRIPTORS = (
    'county',
    'parish',
    'borough',
    'area',
    'region',
    'municipality',
    'municipio',
    'island',
    'district',
)
"""
The last words of a county's ``NAMELSAD``.

A name that ends in one of them is also looked up in the national county file.
"""

COUNTRY_NAMES = ('usa', 'us', 'united states', 'united states of america')
"""Trailing parts of a place's text that name the country and are dropped."""

NONFUNCTIONING_STATUSES = ('I', 'N')
"""
``FUNCSTAT`` codes of inactive and nonfunctioning places.

The match by ``NAME`` and the near matches leave such places out.
"""

PLACE_GEOID_LENGTH = 7
"""The number of digits of a census place's GEOID."""

MAX_NEAR_MATCHES = 5
"""The largest number of near matches listed when no place or county matches."""

NEAR_MATCH_CUTOFF = 0.6
"""The ``cutoff`` of :func:`difflib.get_close_matches` for the near matches."""

_COUNTY_ADVICE = (
    'A county is written with the word that ends its census name, such as County '
    "or Parish, as in 'Marin County, CA'."
)


class CensusBoundary(RegionBoundary):
    """
    A region whose boundary is a census place or county of the TIGER/Line files.

    The place is written with its state, as a USPS code or a full name, as in
    ``'Berkeley, CA'`` or ``'Marin County, California'``. A territory may be written
    without the leading words of its census name, as ``'Virgin Islands'`` for the
    United States Virgin Islands. A trailing ``'USA'`` or ``'United States'`` is
    dropped, and accents, case and periods are ignored, so ``'St Helena, CA'``
    matches St. Helena and ``'Washington, D.C.'`` names the District of Columbia.
    A place's GEOID, such as ``'0606000'``, is taken alone or with its state.

    A place's ``NAMELSAD`` is its census name with its kind, as ``'Berkeley city'``,
    and its ``NAME`` the name alone, as ``'Berkeley'``. :meth:`get_boundary` reads
    the state's place file and takes the first of these that matches the name:

    1. a place whose ``NAMELSAD`` is the name;
    2. for a name whose last word is one of :data:`COUNTY_DESCRIPTORS`, a county of
       the state whose ``NAMELSAD`` is the name, as ``'Marin County'``, read from the
       national county file;
    3. a place whose ``NAME`` is the name, leaving out the places whose ``FUNCSTAT``
       is one of :data:`NONFUNCTIONING_STATUSES`. Of several places of that name,
       the one incorporated place (``CLASSFP`` starting with ``C``) is taken over
       the census-designated places, and a printed line names them all.

    A county is matched only by its name with its descriptor, so ``'Marin, CA'``
    finds no county. The boundary keeps its holes, such as Piedmont inside
    Oakland. The files are read with
    :func:`~brails.utils.tiger_files.read_tiger_file`, which downloads each once per
    process.

    Attributes:
        place (str):
            The text the boundary was made from.
        vintage (int):
            The year of the TIGER/Line release read.
        record (dict[str, str | int]):
            Empty until :meth:`get_boundary` fills it with the boundary's
            ``provider`` (``'census'``), ``kind`` (``'place'`` or ``'county'``),
            ``geoid``, ``name`` (its ``NAMELSAD``), ``state``, ``vintage`` and the
            ``url`` of the file it came from.
    """

    def __init__(self, place: str, *, vintage: int = DEFAULT_TIGER_VINTAGE) -> None:
        """
        Check a place's text and find its state.

        Args:
            place (str):
                The place's name and state, as in ``'Berkeley, CA'``, or a place's
                GEOID.
            vintage (int, optional):
                The year of the TIGER/Line release to read. Defaults to
                :data:`~brails.utils.tiger_files.DEFAULT_TIGER_VINTAGE`.

        Raises:
            TypeError:
                If ``place`` is not a string.
            ValueError:
                If the text names no state, names a state that is not one of
                :data:`STATES`, or is a number other than the GEOID of a place of
                its state, of :data:`PLACE_GEOID_LENGTH` digits that start with the
                state's FIPS code.
        """
        if not isinstance(place, str):
            msg = f'place must be a string, not {type(place).__name__}.'
            raise TypeError(msg)
        self.place = place
        self.vintage = vintage
        self.record: dict[str, str | int] = {}
        countries = {_state_key(name) for name in COUNTRY_NAMES}
        parts = [part.strip() for part in place.split(',') if part.strip()]
        while len(parts) > 1 and _state_key(parts[-1]) in countries:
            parts.pop()
        if len(parts) == 1 and parts[0].isdigit():
            self._name = parts[0]
            self._state = next(
                (row for row in STATES if row[0] == parts[0][:2]), None
            )
        elif len(parts) <= 1:
            msg = (
                f"'{place}' names no state. Add the place's state, as in "
                "'Berkeley, CA', or pass source='osm' to filter_to_region for a "
                'place outside the US.'
            )
            raise ValueError(msg)
        else:
            key = _state_key(parts[-1])
            found = [
                row
                for row in STATES
                if key == row[1].casefold()
                or key == _state_key(row[2])
                or (len(key.split()) > 1 and _state_key(row[2]).endswith(' ' + key))
            ]
            if not found:
                msg = (
                    f"'{parts[-1]}' is not a US state, DC or territory. Write the "
                    'state as its two-letter code or its full name, as in '
                    "'Berkeley, CA' or 'Berkeley, California'. For a place outside "
                    "the United States, pass source='osm' to filter_to_region."
                )
                raise ValueError(msg)
            self._state = found[0]
            self._name = ', '.join(parts[:-1])
        self._geoid = self._name if self._name.isdigit() else None
        if self._geoid and (
            len(self._geoid) != PLACE_GEOID_LENGTH
            or self._state is None
            or self._geoid[:2] != self._state[0]
        ):
            msg = (
                f"'{self._name}' is not the GEOID of a census place"
                + (f' in {self._state[2]}' if len(parts) > 1 else '')
                + f". A place's GEOID has {PLACE_GEOID_LENGTH} digits, of which the "
                "first two are its state's FIPS code. Write a place or county by "
                "its name and state instead, as in 'Berkeley, CA' or 'Marin "
                "County, CA'."
            )
            raise ValueError(msg)

    def get_boundary(
        self,
        print_progress: bool = True,  # noqa: FBT001, FBT002
    ) -> tuple[BaseGeometry, str, str]:
        """
        Return the boundary of the census place or county the text names.

        Args:
            print_progress (bool, optional):
                If ``True``, print the boundary found and a line before the
                county file is downloaded. The line naming the places of one name
                when the incorporated one is taken is printed either way. Defaults
                to ``True``.

        Returns:
            tuple[BaseGeometry, str, str]:
                The boundary in EPSG:4326, the name found with its state, as in
                ``'Berkeley city, California'``, and its GEOID.

        Raises:
            ValueError:
                If no place or county matches the name, listing up to
                :data:`MAX_NEAR_MATCHES` near matches. If two places or counties
                match in one step, listing them. If no place has the GEOID, or the
                server has no file of the vintage.
            requests.RequestException:
                If a download fails otherwise.
        """
        fips, _, state = self._state
        places = read_tiger_file('PLACE', fips, self.vintage)
        if self._geoid:
            hits = places[places['GEOID'] == self._geoid]
            kind, candidates = 'place', places
            if hits.empty:
                msg = (
                    f'No census place in {state} has the GEOID {self._geoid} '
                    f'(TIGER/Line {self.vintage}). Check the GEOID against the '
                    'TIGER/Line release given as vintage, or write the place by '
                    "its name and state, as in 'Berkeley, CA'."
                )
                raise ValueError(msg)
        else:
            hits, kind, candidates = self._match(
                places, print_progress=print_progress
            )
        if hits.empty:
            names = candidates['NAMELSAD'].map(_normalize).tolist()
            words = set(re.findall(r'\w+', _normalize(self._name)))
            close = difflib.get_close_matches(
                _normalize(self._name),
                names,
                n=MAX_NEAR_MATCHES,
                cutoff=NEAR_MATCH_CUTOFF,
            )
            # Every word of the name, as a whole word, and then similarity.
            rows = [
                i
                for i, n in enumerate(names)
                if words and words <= set(re.findall(r'\w+', n))
            ]
            rows += [i for name in close for i, n in enumerate(names) if n == name]
            near = candidates.iloc[list(dict.fromkeys(rows))[:MAX_NEAR_MATCHES]]
            msg = (
                f'No census place or county in {state} is named '
                f"'{self._name}' (TIGER/Line {self.vintage})."
                + (
                    ' Near matches: '
                    + ', '.join(
                        f'{row.NAMELSAD} ({row.GEOID})' for row in near.itertuples()
                    )
                    + '. Pass one of these names or its GEOID,'
                    if not near.empty
                    else " Pass the place's name as the census writes it, as in "
                    "'Berkeley city, CA', or its "
                    f"{PLACE_GEOID_LENGTH}-digit GEOID alone, as in '0606000',"
                )
                + " or pass source='osm' to filter_to_region to look the name up in "
                f'OpenStreetMap. {_COUNTY_ADVICE}'
            )
            raise ValueError(msg)
        if len(hits) > 1:
            listed = ', '.join(
                f'{row.NAMELSAD} ({row.GEOID}, {row.ALAND / 1e6:.1f} km² of land)'
                for row in hits.itertuples()
            )
            kinds = 'places' if kind == 'place' else 'counties'
            msg = (
                f"'{self.place}' matches {len(hits)} census {kinds} in {state}: "
                f'{listed}. Pass the GEOID of the one you mean.'
            )
            raise ValueError(msg)
        row = hits.to_crs(4326).iloc[0]
        area = fips if kind == 'place' else 'us'
        self.record = {
            'provider': 'census',
            'kind': kind,
            'geoid': row.GEOID,
            'name': row.NAMELSAD,
            'state': state,
            'vintage': self.vintage,
            'url': tiger_file_url(kind.upper(), area, self.vintage),
        }
        name = f'{row.NAMELSAD}, {state}'
        if print_progress:
            print_ascii_fallback(
                f'Found {name}, census {kind} {row.GEOID}, '
                f'TIGER/Line {self.vintage}.'
            )
        return row.geometry, name, row.GEOID

    def _match(
        self, places: gpd.GeoDataFrame, *, print_progress: bool
    ) -> tuple[gpd.GeoDataFrame, str, gpd.GeoDataFrame]:
        """
        Match the name against the state's places and counties, in order.

        Args:
            places (gpd.GeoDataFrame):
                The state's place file.
            print_progress (bool):
                Whether to print a line before the county file is downloaded.

        Returns:
            tuple[gpd.GeoDataFrame, str, gpd.GeoDataFrame]:
                The rows of the first step that matches, empty if none does;
                ``'place'`` or ``'county'``; and the rows the near matches of an
                unmatched name are drawn from: the state's counties and then its
                functioning places for a name that ends in a county descriptor,
                and its functioning places otherwise.
        """
        fips, _, state = self._state
        key = _normalize(self._name)
        hits = places[places['NAMELSAD'].map(_normalize) == key]
        if not hits.empty:
            return hits, 'place', places
        functioning = places[~places['FUNCSTAT'].isin(NONFUNCTIONING_STATUSES)]
        candidates = functioning
        if key and key.split()[-1] in COUNTY_DESCRIPTORS:
            notice = (
                f'Downloading the TIGER/Line {self.vintage} county file of the '
                'United States, about 84 MB...'
            )
            counties = read_tiger_file(
                'COUNTY', 'us', self.vintage, notice=notice if print_progress else ''
            )
            in_state = counties[counties['STATEFP'] == fips]
            hits = in_state[in_state['NAMELSAD'].map(_normalize) == key]
            if not hits.empty:
                return hits, 'county', in_state
            candidates = pd.concat([in_state, functioning], ignore_index=True)
        hits = functioning[functioning['NAME'].map(_normalize) == key]
        incorporated = hits[hits['CLASSFP'].str.startswith('C')]
        if len(hits) > 1 and len(incorporated) == 1:
            listed = [f'{r.NAMELSAD} ({r.GEOID})' for r in hits.itertuples()]
            print_ascii_fallback(
                f"'{self.place}' names {', '.join(listed[:-1])} and {listed[-1]} "
                f'in {state}. {incorporated.NAMELSAD.iloc[0]}, the incorporated '
                'place, is taken. For another, pass its GEOID.'
            )
            return incorporated, 'place', candidates
        return hits, 'place', candidates


def _normalize(text: str) -> str:
    """
    Return a name as names are compared: without accents, periods or case.

    Args:
        text (str):
            A name from the user's text or from a TIGER/Line column.

    Returns:
        str:
            The name in NFKD form without combining marks and periods, case
            folded, with runs of whitespace replaced by one space.
    """
    decomposed = unicodedata.normalize('NFKD', text)
    bare = ''.join(c for c in decomposed if not unicodedata.combining(c))
    return ' '.join(bare.casefold().replace('.', '').split())


def _state_key(text: str) -> str:
    """
    Return a state's or a country's name as they are compared.

    Args:
        text (str):
            The last part of the user's text, or a name of :data:`STATES` or
            :data:`COUNTRY_NAMES`.

    Returns:
        str:
            The name as :func:`_normalize` gives it, with ``'united states'``
            written ``'us'``.
    """
    return _normalize(text).replace('united states', 'us')
