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
# 10-02-2025

"""
This module defines RegionBoundary class to store region boundary polygons.

.. autosummary::

    RegionBoundary
"""

import unicodedata
from itertools import groupby
from typing import Any, Callable, Dict, Optional, Tuple, Union

import requests
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, polygonize, unary_union

from brails.utils import GeoTools
from brails.utils.safe_get_json import safe_get_json

class RegionInput:
    """
    Input configuration for a specific region type.

    Parameters:
        dataType (type):
            The expected data type for the input (e.g., str, tuple).
        validationConditions (function):
            A function that validates the input data based on the expected data
            type.
        errorMessage (str):
            The error message to be raised when the validation fails for the
            input data.

    Example:
        >>> location_name_input = RegionInput(
        ...     dataType=str,
        ...     validationConditions=lambda data, dataType: isinstance(
        ...         data,
        ...         dataType
        ...     ),
        ...     errorMessage="'data' must be a string"
        ... )
    """

    def __init__(
        self,
        dataType: type,
        validationConditions: Callable[[Any, type], bool],
        errorMessage: str
    ):
        """Initialize a RegionInput instance."""
        self.dataType = dataType
        self.validationConditions = validationConditions
        self.errorMessage = errorMessage


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

    To import the :class:`RegionBoundary` class, you can use one of the
    following approaches:

    **Using the importer utility:**

    .. code-block:: python

        from brails import Importer

        importer = Importer()
        region_boundary_class = importer.get_class("RegionBoundary")

    **Or by direct import:**

    .. code-block:: python

        from brails.types.region_boundary import RegionBoundary

    Parameters:
        input_type (str):
            The type of the region (e.g., 'locationName', 'locationPolygon').
        query_area (str or tuple):
            Description of the region, validated according to the
            specified type. Either a location name, e.g., 'Berkeley, CA', or a
            tuple, such as (-93.27968, 30.22733, -93.14492, 30.15774)
    """

    def __init__(self, input_dict: Dict[str, Any]) -> None:
        """
        Initialize a RegionBoundary instance from input data.

        Args:
            input_dict (Dict[str, Any]):
                A dictionary specifying the region definition. Must include:

                - ``'type'`` (str):
                    How the region is defined. Supported values are:
                      - ``'locationName'`` – a place name
                          (e.g., 'Berkeley, CA')
                      - ``'locationPolygon'`` – a polygon defined by
                          coordinates
                - ``'data'`` (str | tuple[float, ...]):
                    Region data corresponding to the chosen type:
                      - For ``'locationName'``: a string
                      - For ``'locationPolygon'``: a tuple of coordinates in
                        the form ``(lon1, lat1, lon2, lat2, ..., lonN, latN)``

        Raises:
            TypeError: If ``input_dict`` is not a dictionary.
            ValueError: If required keys are missing, the type is unsupported,
                or the data fails validation.

        Example:
            >>> RegionBoundary(
            ...     {'type': 'locationName', 'data': 'Berkeley, CA'}
            ... )
        """
        # Check if input_dict is a dictionary:
        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input must be a dictionary that includes '
                "information on the 'type' and 'data' required to "
                'specify the region. For example to retrieve the region '
                'boundary for Berkeley, CA, the input should be: '
                "{'type': 'locationName', 'data': 'Berkeley, CA'}"
            )

        # Check if input_dict contains the required keys:
        required_keys = {'type', 'data'}
        missing = required_keys - input_dict.keys()
        if missing:
            label = 'keys' if len(missing) > 1 else 'key'
            raise ValueError(
                f'Input dictionary is missing required {label}: {missing}'
            )

        # Get input type information:
        input_type = input_dict['type']

        if input_type not in SUPPORTED_INPUTS:
            raise ValueError(f"{input_type} is not supported.")

        self.input_type = input_type

        # Get the validation conditions for the given type:
        data_type = SUPPORTED_INPUTS[input_type].dataType
        validation_conditions = \
            SUPPORTED_INPUTS[input_type].validationConditions

        # Check if the data matches the expected type and passes validation
        # conditions:
        if validation_conditions(input_dict['data'], data_type):
            self.query_area = input_dict['data']
        else:
            raise ValueError(
                f"Invalid 'data' specified for {input_dict['type']}. "
                f"{SUPPORTED_INPUTS[input_type].errorMessage}")

    def get_boundary(
        self,
        print_progress: bool = True
    ) -> Tuple[BaseGeometry, str, Optional[Union[int, str]]]:
        """
        Return the boundary of the region based on the provided input type.

        This method processes the region's data based on its type and returns
        the corresponding boundary. If the type is 'locationName', it fetches
        the region boundary from an external data source. If the type is
        'locationPolygon', it converts the provided coordinates (bounding box)
        into a Shapely polygon.

        Args:
            print_progress (bool, optional):
                If set to ``True``, progress messages are printed to the 
                console while retrieving the boundary for the region. If set 
                to ``False``, no messages are displayed. The default is 
                ``True``.

        Returns:
            Tuple[BaseGeometry, str, Optional[Union[int, str]]]:
                A tuple containing, in sequence:
                  1. A Shapely geometry representing the region boundary.
                  2. A human-readable description of the query area.
                  3. An optional identifier (e.g., OSM ID) if available.

        Raises:
            RuntimeError: If input type is unexpectedly invalid.

        Examples:
            Fetch boundary information based on the provided location name.

            >>> from brails import Importer
            >>> importer = Importer()
            >>> region_boundary_class = importer.get_class("RegionBoundary")
            >>> rb = region_boundary_class(
            ...     {'type': 'locationName', 'data': 'Berkeley, CA'}
            ... )
            >>> geom, desc, osm_id = rb.get_boundary()
            Searching for Berkeley, CA...
            Found Berkeley, Alameda County, California, United States
            >>> print(geom.bounds)
            (-122.3686918, 37.8356877, -122.2341962, 37.9066896)
            >>> print(desc)
            Berkeley
            >>> print(osm_id)
            2833528

            Get region boundary data for a bounding box.

            >>> coords = (-122.3, 37.85, -122.25, 37.9)
            >>> rb = region_boundary_class(
            ...     {'type': 'locationPolygon', 'data': coords}
            ... )
            >>> geom, desc, osm_id = rb.get_boundary()
            >>> geom.geom_type
            'Polygon'
            >>> print(desc)
            the bounding box: (-122.3, 37.85, -122.25, 37.9)
        """
        if self.input_type == "locationName":
            return self._fetch_roi(
                self.query_area, 
                print_progress=print_progress
            )

        if self.input_type == "locationPolygon":
            bpoly_tuple = GeoTools.bbox2poly(self.query_area)
            return bpoly_tuple + (None,)

        # Defensive check – should never happen due to __init__ validation:
        raise RuntimeError(f"Unexpected input_type: {self.input_type!r}")

    @staticmethod
    def _fetch_roi(
        queryarea: str,
        outfile: Union[bool, str] = False,
        print_progress: bool = True
    ) -> Tuple[BaseGeometry, str, str]:
        """
        Get the boundary polygon for a region based on its specified name.

        Fetches the region of interest (ROI) based on the provided query area.
        It performs a search using the Nominatim API to find the region's
        OpenStreetMap (OSM) ID and fetch the boundary polygon using the
        Overpass API.

        Args:
            queryarea (str):
                The area to search for, which is a string representing a
                location name (e.g., 'Berkeley, CA')
            outfile (bool or str, optional):
                If a file name is given, the resulting polygon will be saved to
                the specified file in GeoJSON format. The default value is
                ``False``.
            print_progress (bool, optional):
                If set to ``True``, progress messages are printed to the 
                console while retrieving the region of interest. If set to 
                ``False``, no messages are displayed. The default is ``True``.

        Returns:
            tuple:
                A tuple containing the following:

                - bpoly: The bounding polygon as a Shapely geometry object
                         (e.g., ``Polygon`` or ``MultiPolygon``).
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
        if print_progress:
            print(f"\nSearching for {queryarea}...")

        # Format query string for Nominatim:
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

        '''
        r = requests.get(nominatim_endpoint, params=params, headers=headers)
        # datalist = r.json()

        # fmk - adding checks we have json returned - instead of just
        # datalist = r.json()
        
        try:
            datalist = r.json()
        except ValueError as e:  # or requests.JSONDecodeError
            # JSON parse failed — maybe server responded with plain text
            text = r.text.strip()
            raise RuntimeError(f"Failed to parse JSON response. Body was: {text!r}") from e
        if not datalist:
            raise RuntimeError(f"Empty JSON response from {url!r}")        
        '''        

        datalist = safe_get_json(nominatim_endpoint, params=params, headers=headers)
        
        areafound = False
        for data in datalist:
            queryarea_osmid = data["osm_id"]
            queryarea_name = data["display_name"]
            if data["osm_type"] == "relation":
                areafound = True
                break

        if print_progress:
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
                    f"Could not locate an area named '{queryarea}'. "
                    'Please check your location query to make sure '
                    'it was entered correctly.'
                )

        queryarea_printname = queryarea_name.split(",")[0]

        url = "http://overpass-api.de/api/interpreter"

        # Get the polygon boundary for the query area:
        query = f"""
        [out:json][timeout:5000];
        rel({queryarea_osmid});
        out geom;
        """

        '''
        r = requests.get(url, params={"data": query})
        datastruct = r.json()["elements"][0]
        
        '''

        data = safe_get_json(url,
                             params={"data":query},
                             headers=None,
                             timeout=10,
                             retries=3,
                             backoff_factor=2.,
                             valid_key="elements")
        
        datastruct=data["elements"][0]        
        
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
