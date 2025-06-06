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
# Frank McKenna
#
# Last updated:
# 06-05-2025

"""
This module defines RegionBoundary class to store region boundary polygons.

.. autosummary::

    RegionBoundary
"""

import unicodedata
from dataclasses import dataclass
from itertools import groupby

import requests
from shapely.geometry import box, LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, polygonize, unary_union
from typing import Any, Callable, Dict, Optional, Tuple, Union

from brails.utils import GeoTools


@dataclass
class RegionInput:
    """
    Input configuration for a specific region type.

    Attributes:
        dataType (type):
            The expected data type for the input (e.g., str, tuple).
        validationConditions (function):
            A function that validates the input data based on the expected data
            type.
        errorMessage (str):
            The error message to be raised when the validation fails for the
            input data.

    Example:
        location_name_input = RegionInput(
            dataType=str,
            validationConditions=lambda data, dataType: isinstance(data,
                                                                   dataType),
            errorMessage="'data' must be a string"
        )
    """

    dataType: type                  # Expected data type for the input
    validationConditions: Callable  # Validation function that checks the data
    errorMessage: str               # Error message when validation fails


# Define the supported input types with validation logic
SUPPORTED_INPUTS: Dict[str, RegionInput] = {
    'locationName': RegionInput(
        dataType=str,
        validationConditions=lambda data, dataType: isinstance(data, dataType),
        errorMessage="'data' must be a string"
    ),
    'locationPolygon': RegionInput(
        dataType=tuple,
        validationConditions=lambda data, dataType: isinstance(
            data, dataType) and len(data) >= 4 and len(data) % 2 == 0,
        errorMessage="'data' must be a tuple containing an even number of "
        'entries, with at least two pairs of longitude and latitude values.'
    )
}


class RegionBoundary:
    """
    A class for retrieving the bounding polygon of a specified region.

    This class processes input data, validates it, and provides functionality
    to obtain the region boundary based on the given data. The input must
    specify the type of region (e.g., location name or polygon) and the
    corresponding data.

    Attributes:
        type (str):
            The type of the region (e.g., 'locationName', 'locationPolygon').
        data (str or tuple):
            The data associated with the region, validated according to the
            specified type. Either a location name, e.g., 'Berkeley, CA', or a
            tuple, such as (-93.27968, 30.22733, -93.14492, 30.15774)

    Methods:
        get_boundary():
            Returns the boundary polygon of the specified region.
    """

    def __init__(self, input_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize and validate a RegionBoundary instance from input data.

        Args:
            input_dict (Optional[Dict[str, Any]]):
                Dictionary containing:
                - 'type': the region input type (must match a key in
                   SUPPORTED_INPUTS)
                - 'data': the actual input data (validated using the
                   corresponding rule)

        Raises:
            TypeError:
                If input_dict is not a dictionary.
            ValueError:
                If required keys are missing or if validation fails.
        """
        if not isinstance(input_dict, dict):
            raise TypeError('Input must be a dictionary that includes '
                            "information on the 'type' and 'data' required to "
                            'specify the region. For example to retrieve the '
                            'region boundary for Berkeley, CA, the input '
                            'should be:'
                            " {'type': 'locationName', 'data': Berkeley, CA}")

        if 'type' not in input_dict:
            raise ValueError("Input dictionary must contain a 'type' key.")

        if 'data' not in input_dict:
            raise ValueError("Input dictionary must contain a 'data' key.")

        input_type = input_dict['type']

        if input_type not in SUPPORTED_INPUTS:
            raise ValueError(f"{input_type} is not supported.")

        self.type = input_type

        # Get the validation conditions for the given type:
        data_type = SUPPORTED_INPUTS[input_type].dataType
        validation_conditions = \
            SUPPORTED_INPUTS[input_type].validationConditions

        # Check if the data matches the expected type and passes validation
        # conditions:
        if validation_conditions(input_dict['data'], data_type):
            self.data = input_dict['data']
        else:
            raise ValueError(
                f"Invalid 'data' specified for {input_dict['type']}: "
                f"{SUPPORTED_INPUTS[input_type].errorMessage}")

    def get_boundary(
        self
    ) -> Tuple[BaseGeometry, str, Optional[Union[int, str]]]:
        """
        Return the boundary of the region based on the provided input type.

        This method processes the region's data based on its type and returns
        the corresponding boundary. If the type is 'locationName', it fetches
        the region boundary from an external data source. If the type is
        'locationPolygon', it converts the provided coordinates (bounding box)
        into a Shapely polygon.

        Returns:
            Tuple[BaseGeometry, str, Optional[Union[int, str]]]:
                - A Shapely geometry object representing the region boundary.
                - A human-readable description of the query area.
                - An optional OSM ID (or similar identifier) for the boundary,
                  if available.

        Raises:
            ValueError: If the input data type is not 'locationName' or
                        'locationPolygon', or if the data is invalid for the
                        given type.
        """
        queryarea = self.data

        if self.type == 'locationName':
            result = self._fetch_roi(queryarea)
        elif self.type == 'locationPolygon':
            result = self._bbox2poly(queryarea)
        else:
            raise ValueError('Invalid data type for query area.')

        return result

    def _fetch_roi(
        self,
        queryarea: str,
        outfile: Union[bool, str] = False
    ) -> Tuple[BaseGeometry, str, str]:
        """
        Get the boundary polygon for a region based on its specified name.

        Fetches the region of interest (ROI) based on the provided query area.
        It performs a search using the Nominatim API to find the region's
        OpenStreetMap (OSM) ID and fetch the boundary polygon using the
        Overpass API.

        Args:
            queryarea (str):
                The area to search for, which can is a string representing a
                location name (e.g., 'Berkeley, CA')
            outfile (bool or str, optional):
                If a file name is given, the resulting polygon will be saved to
                the specified file in GeoJSON format. The default value is
                False.

        Returns:
            tuple:
                A tuple containing the following:
                - bpoly: The bounding polygon as a Shapely geometry object
                         (e.g., Polygon or MultiPolygon).
                - queryarea_printname: A human-readable name of the query area
                                       (e.g., 'Berkeley').
                - queryarea_osmid: The OpenStreetMap ID of the found area.

        Raises:
            ValueError:
                If the query area cannot be found or the boundary cannot be
                retrieved.

        Example:
            result = _fetch_roi('Berkeley, CA')
            # result[0] is the boundary polygon (bpoly),
            # result[1] is the query area name ('Berkeley'),
            # result[2] is the OSM ID for Berkeley.
        """
        # Search for the query area using Nominatim API:
        print(f"\nSearching for {queryarea}...")
        queryarea = queryarea.replace(" ", "+").replace(",", "+")

        queryarea_formatted = ""
        for i, j in groupby(queryarea):
            if i == "+":
                queryarea_formatted += i
            else:
                queryarea_formatted += "".join(list(j))

        nominatim_endpoint = 'https://nominatim.openstreetmap.org/search'

        params = {'q': queryarea_formatted,
                  'format': 'jsonv2'
                  }

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/91.0.4472.124 Safari/537.36',
                   'Accept-Language': 'en-US,en;q=0.5',
                   'Referer': 'https://www.openstreetmap.org/',
                   'Connection': 'keep-alive'
                   }

        r = requests.get(nominatim_endpoint, params=params, headers=headers)
        datalist = r.json()

        areafound = False
        for data in datalist:
            queryarea_osmid = data["osm_id"]
            queryarea_name = data["display_name"]
            if data["osm_type"] == "relation":
                areafound = True
                break

        if areafound:
            try:
                print(f"Found {queryarea_name}")
            except UnicodeEncodeError:
                queryareaNameUTF = unicodedata.normalize(
                    "NFKD", queryarea_name).encode("ascii", "ignore")
                queryareaNameUTF = queryareaNameUTF.decode("utf-8")
                print(f"Found {queryareaNameUTF}")
        else:
            raise ValueError(
                f"Could not locate an area named {queryarea}. "
                + "Please check your location query to make sure "
                + "it was entered correctly."
            )

        queryarea_printname = queryarea_name.split(",")[0]

        url = "http://overpass-api.de/api/interpreter"

        # Get the polygon boundary for the query area:
        query = f"""
        [out:json][timeout:5000];
        rel({queryarea_osmid});
        out geom;
        """

        r = requests.get(url, params={"data": query})

        datastruct = r.json()["elements"][0]
        if datastruct["tags"]["type"] in ["boundary", "multipolygon"]:
            lss = []
            for coorddict in datastruct["members"]:
                if coorddict["role"] == "outer":
                    ls = []
                    for coord in coorddict["geometry"]:
                        ls.append([coord["lon"], coord["lat"]])
                    lss.append(LineString(ls))

            merged = linemerge([*lss])
            borders = unary_union(merged)  # linestrings to a MultiLineString
            polygons = list(polygonize(borders))

            if len(polygons) == 1:
                bpoly = polygons[0]
            else:
                bpoly = MultiPolygon(polygons)

        else:
            raise ValueError(
                f"Could not retrieve the boundary for {queryarea}. "
                + "Please check your location query to make sure "
                + "it was entered correctly."
            )
        if outfile:
            GeoTools.write_polygon_to_geojson(bpoly, outfile)

        return bpoly, queryarea_printname, queryarea_osmid

    def _bbox2poly(
        self,
        queryarea: Tuple[float, ...],
        outfile: Union[bool, str] = False
    ) -> Tuple[Polygon, str, Optional[None]]:
        """
        Get the boundary polygon for a region based on its coordinates.

        This method parses the provided bounding polygon coordinates into a
        polygon object. The polygon can be defined by at least two pairs of
        longitude/latitude values, with an even number of elements. If a file
        name is provided in the `outfile` argument, the resulting polygon is
        saved to a GeoJSON file.

        Args:
            queryarea (tuple):
                A tuple containing longitude/latitude pairs that define a
                bounding box. The tuple should contain at least two pairs of
                coordinates (i.e., 4 values), and the number of elements must
                be an even number.
            outfile (bool or str, optional):
                If a file name is provided, the resulting polygon will be
                written to the specified file in GeoJSON format. Defaults to
                False.

        Raises:
            ValueError:
                If the `queryarea` contains an odd number of elements or fewer
                than two pairs of coordinates.

        Returns:
            Tuple[Polygon, str, Optional[None]]:
                - The bounding polygon (`bpoly`).
                - A human-readable string representation of the bounding box
                  (`queryarea_printname`).
                - `None` as a placeholder (for future extensions, if needed).
        """
        # Parse the entered bounding box into a polygon:
        if len(queryarea) % 2 == 0 and len(queryarea) != 0:
            if len(queryarea) == 4:
                bpoly = box(*queryarea)
                queryarea_printname = f"the bounding box: {list(queryarea)}"
            elif len(queryarea) > 4:
                queryarea_printname = "the bounding box: ["
                bpolycoords = []
                for i in range(int(len(queryarea) / 2)):
                    bpolycoords.append([queryarea[2 * i],
                                        queryarea[2 * i + 1]])
                    queryarea_printname += (f'{queryarea[2*i]}, '
                                            f'{queryarea[2*i+1]}, ')
                bpoly = Polygon(bpolycoords)
                queryarea_printname = queryarea_printname[:-2] + "]"
            else:
                raise ValueError(
                    'Less than two longitude/latitude pairs were entered to '
                    'define the bounding box entry. A bounding box can be '
                    'defined by using at least two longitude/latitude pairs.'
                )
        else:
            raise ValueError('Incorrect number of elements detected in the '
                             'tuple for the bounding box. Please check to see '
                             'if you are missing a longitude or latitude '
                             'value.'
                             )
        if outfile:
            GeoTools.write_polygon_to_geojson(bpoly, outfile)

        return bpoly, queryarea_printname, None
