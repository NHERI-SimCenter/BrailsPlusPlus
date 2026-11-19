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


"""Tests for a region from a census place or county name."""

from __future__ import annotations

import re
import socket

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from brails.types import CensusBoundary, census_boundary
from brails.types.region_boundary import RegionBoundary
from brails.utils import Importer

# Oakland's square, with Piedmont as a hole.
OAKLAND = Polygon(
    box(-122.3, 37.7, -122.1, 37.9).exterior.coords,
    [box(-122.24, 37.81, -122.2, 37.84).exterior.coords],
)

# Places of California: Berkeley, Alameda and Oakland as cities, a city and a CDP
# of one name, two CDPs of one name, a name with an accent, three names that only
# resemble others, an invented nonfunctioning city beside its metro government's
# balance, as Louisville has in Kentucky's file, a name with a period and a name
# that ends in a county descriptor.
PLACES = gpd.GeoDataFrame(
    [
        ('0606000', 'Berkeley', 'Berkeley city', 'C1', 'A', 45e6),
        ('0600562', 'Alameda', 'Alameda city', 'C1', 'A', 27e6),
        ('0653000', 'Oakland', 'Oakland city', 'C1', 'A', 144e6),
        ('0621796', 'El Cerrito', 'El Cerrito city', 'C1', 'A', 9.6e6),
        ('0621810', 'El Cerrito', 'El Cerrito CDP', 'U1', 'S', 3e6),
        ('0604478', 'Bayview', 'Bayview CDP', 'U1', 'S', 1.94e6),
        ('0604470', 'Bayview', 'Bayview CDP', 'U1', 'S', 0.81e6),
        (
            '0639003',
            'La Cañada Flintridge',
            'La Cañada Flintridge city',
            'C1',
            'A',
            22e6,
        ),
        ('0645820', 'Marin City', 'Marin City CDP', 'U1', 'S', 1.6e6),
        ('0664140', 'St. Helena', 'St. Helena city', 'C1', 'A', 12.8e6),
        ('0606210', 'Bethel Island', 'Bethel Island CDP', 'U1', 'S', 13.6e6),
        ('0645778', 'Marina', 'Marina city', 'C1', 'A', 23e6),
        (
            '0665042',
            'San Buenaventura (Ventura)',
            'San Buenaventura (Ventura) city',
            'C1',
            'A',
            56e6,
        ),
        ('0699001', 'Riverbend', 'Riverbend city', 'C9', 'N', 150e6),
        (
            '0699002',
            'Riverbend metro government (balance)',
            'Riverbend metro government (balance)',
            'C8',
            'F',
            600e6,
        ),
    ],
    columns=['GEOID', 'NAME', 'NAMELSAD', 'CLASSFP', 'FUNCSTAT', 'ALAND'],
    geometry=[box(-122.3, 37.8, -122.2, 37.9), box(-122.3, 37.7, -122.2, 37.8)]
    + [OAKLAND]
    + [box(-121.0 + i, 36.0, -120.9 + i, 36.1) for i in range(12)],
    crs='EPSG:4269',
)

# Counties: two of California and one of Tennessee.
COUNTIES = gpd.GeoDataFrame(
    [
        ('06', '06001', 'Alameda County', 1.9e9),
        ('06', '06041', 'Marin County', 1.3e9),
        ('47', '47037', 'Davidson County', 1.3e9),
    ],
    columns=['STATEFP', 'GEOID', 'NAMELSAD', 'ALAND'],
    geometry=[
        box(-122.4, 37.4, -121.4, 37.9),
        box(-123.0, 37.8, -122.4, 38.3),
        box(-87.0, 36.0, -86.5, 36.4),
    ],
    crs='EPSG:4269',
)

COUNTY_NOTICE = (
    'Downloading the TIGER/Line {} county file of the United States, about 84 MB...'
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every connection and name lookup, so no test reaches a service."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = 'A test tried to open a network connection.'
        raise OSError(msg)

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'getaddrinfo', refuse)


@pytest.fixture
def files_read(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, int, str]]:
    """
    Serve :data:`PLACES` and :data:`COUNTIES` in place of the TIGER/Line files.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Replaces the reader the module calls.

    Returns:
        list[tuple[str, str, int, str]]:
            The layer, area, vintage and notice of each file read, in order.
    """
    read: list[tuple[str, str, int, str]] = []

    def read_tiger_file(
        layer: str, area: str, vintage: int, notice: str = ''
    ) -> gpd.GeoDataFrame:
        read.append((layer, area, vintage, notice))
        return {'PLACE': PLACES, 'COUNTY': COUNTIES}[layer].copy()

    monkeypatch.setattr(census_boundary, 'read_tiger_file', read_tiger_file)
    return read


PLACE_FILE = ('PLACE', '06', 2023, '')


@pytest.mark.parametrize(
    ('text', 'geoid', 'name'),
    [
        ('Berkeley city, CA', '0606000', 'Berkeley city, California'),
        ('Berkeley, CA', '0606000', 'Berkeley city, California'),
        ('berkeley ,  california, U.S.A.', '0606000', 'Berkeley city, California'),
        ('Alameda, CA', '0600562', 'Alameda city, California'),
        (
            'La Canada  Flintridge, ca',
            '0639003',
            'La Cañada Flintridge city, California',
        ),
        ('0604478', '0604478', 'Bayview CDP, California'),
        ('0604478, CA', '0604478', 'Bayview CDP, California'),
        ('Riverbend city, CA', '0699001', 'Riverbend city, California'),
        ('St Helena, CA', '0664140', 'St. Helena city, California'),
    ],
)
def test_place_found(files_read: list, text: str, geoid: str, name: str) -> None:
    """A place's NAMELSAD, NAME or GEOID finds it without the county file."""
    boundary, found, found_geoid = CensusBoundary(text).get_boundary()
    assert (found, found_geoid) == (name, geoid)
    assert boundary.equals(PLACES.set_index('GEOID').geometry[geoid])
    assert files_read == [PLACE_FILE]


def test_county_by_its_name_with_descriptor(
    files_read: list, capsys: pytest.CaptureFixture[str]
) -> None:
    """A name ending in a county descriptor reads the county file after places."""
    region = CensusBoundary('Alameda County, CA', vintage=2024)
    boundary, name, geoid = region.get_boundary()
    assert (name, geoid) == ('Alameda County, California', '06001')
    assert boundary.equals(COUNTIES.geometry[0])
    assert files_read == [
        ('PLACE', '06', 2024, ''),
        ('COUNTY', 'us', 2024, COUNTY_NOTICE.format(2024)),
    ]
    assert region.record == {
        'provider': 'census',
        'kind': 'county',
        'geoid': '06001',
        'name': 'Alameda County',
        'state': 'California',
        'vintage': 2024,
        'url': 'https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/'
        'tl_2024_us_county.zip',
    }
    assert capsys.readouterr().out == (
        'Found Alameda County, California, census county 06001, TIGER/Line 2024.\n'
    )


@pytest.mark.usefixtures('files_read')
def test_record_of_a_place() -> None:
    """The record is empty before the lookup and names the file read after it."""
    region = CensusBoundary('Berkeley, CA')
    assert region.record == {}
    region.get_boundary(print_progress=False)
    assert region.record == {
        'provider': 'census',
        'kind': 'place',
        'geoid': '0606000',
        'name': 'Berkeley city',
        'state': 'California',
        'vintage': 2023,
        'url': 'https://www2.census.gov/geo/tiger/TIGER2023/PLACE/'
        'tl_2023_06_place.zip',
    }


def test_county_only_by_its_descriptor(files_read: list) -> None:
    """A county name without its descriptor is no match, and no county is read."""
    with pytest.raises(
        ValueError, match='the word that ends its census name'
    ) as error:
        CensusBoundary('Marin, CA').get_boundary()
    assert 'Marin City CDP (0645820), Marina city (0645778)' in str(error.value)
    assert files_read == [PLACE_FILE]


def test_county_of_another_state_not_matched(files_read: list) -> None:
    """A county is looked up in the text's state only."""
    with pytest.raises(ValueError, match='No census place or county in California'):
        CensusBoundary('Davidson County, CA').get_boundary()
    assert [read[0] for read in files_read] == ['PLACE', 'COUNTY']


@pytest.mark.usefixtures('files_read')
def test_misspelled_county_suggests_counties() -> None:
    """A misspelled county name lists the county first among its near matches."""
    with pytest.raises(
        ValueError, match=r'Near matches: Alameda County \(06001\), '
    ):
        CensusBoundary('Alamda County, CA').get_boundary()


@pytest.mark.usefixtures('files_read')
def test_misspelled_place_with_a_descriptor_suggests_it() -> None:
    """A place name ending in a descriptor word is suggested beside counties."""
    with pytest.raises(
        ValueError, match=r'Near matches: Bethel Island CDP \(0606210\)'
    ):
        CensusBoundary('Bethal Island, CA').get_boundary()


@pytest.mark.usefixtures('files_read')
def test_name_of_no_letters_refused() -> None:
    """A name that normalizes to nothing is no match."""
    with pytest.raises(ValueError, match='No census place or county in California'):
        CensusBoundary('\u00b4, CA').get_boundary()


@pytest.mark.usefixtures('files_read')
def test_incorporated_over_cdp(capsys: pytest.CaptureFixture[str]) -> None:
    """Of a city and a CDP of one name the city is taken, with a line naming both."""
    region = CensusBoundary('El Cerrito, CA')
    _, name, geoid = region.get_boundary(print_progress=False)
    assert (name, geoid) == ('El Cerrito city, California', '0621796')
    assert capsys.readouterr().out == (
        "'El Cerrito, CA' names El Cerrito city (0621796) and El Cerrito CDP "
        '(0621810) in California. El Cerrito city, the incorporated place, is '
        'taken. For another, pass its GEOID.\n'
    )


@pytest.mark.usefixtures('files_read')
def test_nonfunctioning_place_not_matched_by_name() -> None:
    """A nonfunctioning city is no match by NAME, and its balance is suggested."""
    message = (
        "No census place or county in California is named 'Riverbend' (TIGER/Line "
        '2023). Near matches: Riverbend metro government (balance) (0699002). Pass '
        "one of these names or its GEOID, or pass source='osm' to filter_to_region "
        'to look the name up in OpenStreetMap. A county is written with the word '
        "that ends its census name, such as County or Parish, as in 'Marin County, "
        "CA'."
    )
    with pytest.raises(ValueError, match=f'^{re.escape(message)}$'):
        CensusBoundary('Riverbend, CA').get_boundary()


@pytest.mark.usefixtures('files_read')
def test_two_cdps_of_one_name_refused() -> None:
    """Two places that still tie are listed with their GEOIDs and land areas."""
    message = (
        "'Bayview, CA' matches 2 census places in California: Bayview CDP "
        '(0604478, 1.9 km² of land), Bayview CDP (0604470, 0.8 km² of land). Pass '
        'the GEOID of the one you mean.'
    )
    with pytest.raises(ValueError, match=f'^{re.escape(message)}$'):
        CensusBoundary('Bayview, CA').get_boundary()


@pytest.mark.usefixtures('files_read')
def test_misspelled_name_suggests_by_similarity() -> None:
    """A name that matches nothing lists near matches and what to pass instead."""
    message = (
        "No census place or county in California is named 'Berkely' (TIGER/Line "
        '2023). Near matches: Berkeley city (0606000). Pass one of these names or '
        "its GEOID, or pass source='osm' to filter_to_region to look the name up in "
        'OpenStreetMap. A county is written with the word that ends its census '
        "name, such as County or Parish, as in 'Marin County, CA'."
    )
    with pytest.raises(ValueError, match=f'^{re.escape(message)}$'):
        CensusBoundary('Berkely, CA').get_boundary()


@pytest.mark.usefixtures('files_read')
def test_no_near_match() -> None:
    """A name with no word and no likeness gets the forms to pass instead."""
    with pytest.raises(
        ValueError,
        match=r"named '--' .*\. Pass the place's name as the census writes it, "
        r"as in 'Berkeley city, CA', or its 7-digit GEOID alone",
    ):
        CensusBoundary('--, CA').get_boundary()


@pytest.mark.usefixtures('files_read')
def test_near_matches_by_whole_words_first() -> None:
    """A place whose NAMELSAD holds every word of the name is suggested first."""
    with pytest.raises(
        ValueError, match=r'Near matches: San Buenaventura \(Ventura\)'
    ):
        CensusBoundary('Ventura, CA').get_boundary()


@pytest.mark.usefixtures('files_read')
def test_unknown_geoid_refused() -> None:
    """A GEOID of the state that no place has is refused with what to pass."""
    with pytest.raises(
        ValueError, match="has the GEOID 0699999 .* as in 'Berkeley, CA'"
    ):
        CensusBoundary('0699999').get_boundary()


@pytest.mark.parametrize(
    ('text', 'message'),
    [
        ('Berkeley', "'Berkeley' names no state. Add the place's state"),
        ('Lima, Peru', "'Peru' is not a US state, DC or territory. Write the state"),
        ('06001', "'06001' is not the GEOID of a census place. A place's GEOID"),
        ('9912345', "'9912345' is not the GEOID of a census place."),
        ('0606000, TN', "'0606000' is not the GEOID of a census place in Tennessee"),
    ],
)
def test_text_refused_before_any_file(
    files_read: list, text: str, message: str
) -> None:
    """A text without a US state, or a number that is no place GEOID, is refused."""
    with pytest.raises(ValueError, match=message):
        CensusBoundary(text)
    assert files_read == []


@pytest.mark.parametrize(
    ('state', 'fips'),
    [
        ('D.C.', '11'),
        ('Virgin Islands', '78'),
        ('U.S. Virgin Islands', '78'),
        ('Northern Mariana Islands', '69'),
        ('Commonwealth of the Northern Mariana Islands', '69'),
        ('puerto rico', '72'),
    ],
)
def test_state_spellings(files_read: list, state: str, fips: str) -> None:
    """A state with periods, or a territory without its first words, is found."""
    with pytest.raises(ValueError, match='No census place or county'):
        CensusBoundary(f'Testville, {state}, USA').get_boundary()
    assert files_read[0][:2] == ('PLACE', fips)


@pytest.mark.usefixtures('files_read')
def test_holes_kept() -> None:
    """Oakland keeps Piedmont as a hole."""
    boundary, _, _ = CensusBoundary('Oakland, CA').get_boundary()
    assert len(boundary.interiors) == 1
    assert not boundary.contains(Point(-122.22, 37.825))
    assert boundary.contains(Point(-122.25, 37.75))


def test_boundary_in_wgs84(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file in another CRS gives the boundary in longitude and latitude."""
    projected = PLACES.to_crs('EPSG:3310')
    monkeypatch.setattr(
        census_boundary, 'read_tiger_file', lambda *_args: projected.copy()
    )
    boundary, _, _ = CensusBoundary('Berkeley, CA').get_boundary()
    assert boundary.bounds == pytest.approx((-122.3, 37.8, -122.2, 37.9), abs=1e-6)


def test_quiet_without_progress(
    files_read: list, capsys: pytest.CaptureFixture[str]
) -> None:
    """With print_progress False no line is printed and no notice is passed on."""
    CensusBoundary('Berkeley, CA').get_boundary(print_progress=False)
    CensusBoundary('Alameda County, CA').get_boundary(print_progress=False)
    assert capsys.readouterr().out == ''
    assert files_read[-1] == ('COUNTY', 'us', 2023, '')


def test_a_region_boundary_the_importer_finds() -> None:
    """A CensusBoundary is a RegionBoundary, and the Importer finds it."""
    assert isinstance(CensusBoundary('Berkeley, CA'), RegionBoundary)
    assert Importer().get_class('CensusBoundary') is CensusBoundary
