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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.

"""
Regenerate the stored NSI API sample used by the parser tests.

The fixture is a verbatim NSI API response for a small, pinned San
Francisco block chosen for its variety: RES1 codes with and without
basement and multi-story suffixes, several RES3 subtypes, commercial
occupancies, and more than one general building type. Rerun this script
manually when the NSI schema changes and the live test starts failing:

    python tests/scrapers/data/nsi_fixture_generator.py

The test suite never runs this module; it only reads the JSON it writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from brails.scrapers.nsi_parser.nsi_parser import API_ENDPOINT

# (minlon, minlat, maxlon, maxlat); about one Inner Sunset block.
FIXTURE_BBOX = (-122.4688, 37.7627, -122.4668, 37.7639)
FIXTURE_PATH = Path(__file__).parent / 'nsi_2026_sample.json'


def _bbox_polygon_string(
    minlon: float, minlat: float, maxlon: float, maxlat: float
) -> str:
    """
    Build the closed-ring coordinate string the NSI API expects.

    Args:
        minlon (float):
            Western edge of the bounding box.
        minlat (float):
            Southern edge of the bounding box.
        maxlon (float):
            Eastern edge of the bounding box.
        maxlat (float):
            Northern edge of the bounding box.

    Returns:
        str:
            The five-vertex ring string for the ``bbox`` parameter.
    """
    return (
        f'{minlon:.5f},{minlat:.5f},{minlon:.5f},{maxlat:.5f},'
        f'{maxlon:.5f},{maxlat:.5f},{maxlon:.5f},{minlat:.5f},'
        f'{minlon:.5f},{minlat:.5f}'
    )


def fetch_sample() -> dict:
    """
    Download the pinned sample area from the live NSI API.

    Returns:
        dict:
            The API response, verbatim.
    """
    url = API_ENDPOINT + _bbox_polygon_string(*FIXTURE_BBOX)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Fetch the sample and rewrite the stored fixture."""
    sample = fetch_sample()
    count = len(sample.get('features', []))
    FIXTURE_PATH.write_text(json.dumps(sample, indent=1) + '\n')
    print(f'Wrote {count} records to {FIXTURE_PATH}.')


if __name__ == '__main__':
    main()
