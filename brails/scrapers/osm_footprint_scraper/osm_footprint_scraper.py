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
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 08-14-2025

"""
This module defines the class scraping building data from OSM.

.. autosummary::

    OSM_FootprintScraper
"""

import requests

from brails.scrapers.footprint_scraper import FootprintScraper
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import AssetInventory
from brails.utils import InputValidator


class OSM_FootprintScraper(FootprintScraper):
    """
    A class for retrieving and processing building footprint data from OSM.

    This class provides methods for querying and extracting building footprint
    data, including attributes such as building height, era of construction,
    and number of stories, using the OpenStreetMap API. It handles both
    rectangular bounding boxes and specific region boundaries defined by OSM
    data.

    Attributes:
        length_unit (str):
            The unit of length for building height measurements. Default is
            'ft'.

    Methods:
        get_footprints(region: RegionBoundary) -> AssetInventory:
            Retrieves the building footprints and associated attributes for a
            given region using the OpenStreetMap API.
    """

    def __init__(self, input_dict: dict):
        """
        Initialize the class object with length units.

        Args:
            input_dict (dict):
                A dictionary specifying length units. If not provided, 'ft' is
                assumed by default..
        """
        self.length_unit = input_dict.get('length', 'ft')

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        Get the OSM footprints and atrributes for buildings in an area.

        Args:
            region (RegionBoundary):
                The region of interest.

        Returns:
            AssetInventory:
                A building inventory for buildings in the region.

        """
        bpoly, queryarea_printname, osmid = region.get_boundary()

        # If the bounding polygon was obtained by calling a region name:
        if osmid is not None:

            queryarea_turboid = osmid + 3600000000
            query = f"""
            [out:json][timeout:5000][maxsize:2000000000];
            area({queryarea_turboid})->.searchArea;
            way["building"](area.searchArea);
            out body;
            >;
            out skel qt;
            """

        else:
            # If the bounding polygon is rectangular:
            if InputValidator.is_box(bpoly):
                # Convert the bounding polygon coordinates to latitude and
                # longitude fashion:
                bbox_coords = bpoly.bounds
                bbox = f'{bbox_coords[1]},{bbox_coords[0]},{bbox_coords[3]},'\
                    f'{bbox_coords[2]}'
            else:
                bbox_coords = list(bpoly.exterior.coords)
                bbox = 'poly:"'
                for (lon, lat) in bbox_coords[:-1]:
                    bbox += f'{lat} {lon} '
                bbox = bbox[:-1] + '"'

            query = f"""
            [out:json][timeout:5000][maxsize:2000000000];
            way["building"]({bbox});
            out body;
            >;
            out skel qt;
            """

        url = "http://overpass-api.de/api/interpreter"
        r = requests.get(url, params={"data": query})

        datalist = r.json()["elements"]
        nodedict = {}
        for data in datalist:
            if data["type"] == "node":
                nodedict[data["id"]] = [data["lon"], data["lat"]]

        attrmap = {
            "start_date": "erabuilt",
            "building:start_date": "erabuilt",
            "construction_date": "erabuilt",
            "roof:shape": "roofshape",
            "height": "buildingheight",
        }

        levelkeys = {"building:levels", "roof:levels",
                     "building:levels:underground"}
        otherattrkeys = set(attrmap.keys())
        datakeys = levelkeys.union(otherattrkeys)

        attrkeys = ["buildingheight", "erabuilt", "numstories", "roofshape"]
        attributes = {key: [] for key in attrkeys}
        fpcount = 0
        footprints = []
        for data in datalist:
            if data["type"] == "way":
                nodes = data["nodes"]
                footprint = []
                for node in nodes:
                    footprint.append(nodedict[node])
                footprints.append(footprint)

                fpcount += 1
                availableTags = set(data["tags"].keys()).intersection(datakeys)
                for tag in availableTags:
                    nstory = 0
                    if tag in otherattrkeys:
                        attributes[attrmap[tag]].append(data["tags"][tag])
                    elif tag in levelkeys:
                        try:
                            nstory += int(data["tags"][tag])
                        except ValueError:
                            pass

                    if nstory > 0:
                        attributes["numstories"].append(nstory)
                for attr in attrkeys:
                    if len(attributes[attr]) != fpcount:
                        attributes[attr].append("NA")

        attributes["buildingheight"] = [
            self._height2float(height, self.length_unit)
            for height in attributes["buildingheight"]
        ]

        attributes["erabuilt"] = [
            self._yearstr2int(year) for year in attributes["erabuilt"]
        ]

        attributes["numstories"] = [
            nstories if nstories != "NA" else None
            for nstories in attributes["numstories"]
        ]

        print(f'\nFound a total of {fpcount} building footprints in '
              f'{queryarea_printname}')

        return self._create_asset_inventory(footprints,
                                            attributes,
                                            self.length_unit)

    def _cleanstr(self, input_str: str) -> str:
        """
        Return a string containing only alphanumeric characters, '.', spaces.

        Args:
            input_str (str):
                Input string
        Returns:
            str:
                A string containing only alphanumeric characters, '.' and
                spaces
        """
        return "".join(
            char
            for char in input_str
            if not char.isalpha()
            and not char.isspace()
            and (char == "." or char.isalnum())
        )

    def _yearstr2int(self, input_str: str) -> int:
        """
        Convert a year string to an integer.

        The method first cleans the input string by removing unwanted
        characters and extracting the first four digits. If the cleaned string
        has exactly four digits, it attempts to convert the string to an
        integer and returns it. If the conversion fails or the string does not
        contain exactly four digits, it returns None.

        Args:
            input_str (str):
                The input string representing a year.

        Returns:
            int or None:
                The year as an integer if valid, or None if the input is 'NA'
                or cannot be converted to a valid year.
        """
        # If year is NA, return None:
        if input_str == 'NA':
            return None

        # Get first 4 digits of the cleaned year string:
        cleaned_str = self._cleanstr(input_str)[:4]

        # Convert the year string to an integer. If unsuccessful return None:
        try:
            return int(cleaned_str) if len(cleaned_str) == 4 else None
        except ValueError:
            return None

    def _height2float(self, input_str: str, length_unit: str) -> float:
        """
        Convert a height string to a float.

        This function first cleans the input string to remove unwanted
        characters, then attempts to convert it to a float. If the conversion
        is successful, it will optionally convert the height from meters
        (default unit) to feet if the specified `length_unit` is "ft". The
        result is rounded to one decimal place. If the height is 'NA' or the
        conversion fails, None is returned.

        Args:
            input_str (str):
                The input string representing the height, which may include
                non-numeric characters that will be cleaned.
            length_unit (str):
                The unit to which the height should be converted. If "ft", the
                height will be converted from meters to feet. Otherwise, the
                height is returned in meters.

        Returns:
            float or None:
                The height as a float, possibly converted to the specified
                length unit, or None if the input string is "NA" or the
                conversion fails.
        """
        # If height is NA, return None:
        if input_str == "NA":
            return None

        # Convert the height string to a float. If unsuccessful return None:
        try:
            height = float(self._cleanstr(input_str))

            # If the length_unit is "ft", convert to feet (default OSM units
            # are in meters). The result is rounded to one decimal place:
            if length_unit == 'ft':
                return round(height * 3.28084, 1)
            return round(height, 1)
        except ValueError:
            return None
