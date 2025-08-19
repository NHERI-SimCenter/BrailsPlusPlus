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
This module provides a collection of utility classes for BRAILS++.

Available classes:

    - :class:`Importer`: Class for importing class files.
    - :class:`GeoTools`: Class containing static methods for geospatial
      analysis and operations.
    - :class:`ArcgisAPIServiceHelper`: Class for interacting with an ArcGIS API
      service.
    - :class:`PlotTools`: Class for creating visually appealing figures.
    - :class:`UnitConverter`: Class for converting between different units of
      length, area, and weight.
    - :class:`InputValidator`: Class for validating BRAILS input data.
    - :class:`InventoryValidator`: Class for validating BRAILS
      ``AssetInventory`` objects.
    - :class:`SpatialJoinMethods`: Methods to spatially merge
      ``AssetInventory`` objects.
    - :class:`ModelUtils`: Utility class for computer vision models in BRAILS.
    - :class:`RAPIDUtils`: Utility class for datasets created by the RAPID
      facility at UW.
    - :class:`SegmentationUtils`:  Utility class for creating visualizations of 
      segmentation masks on images.
"""
from .importer import Importer
from .geo_tools import GeoTools
from .arcgisapi_service_helper import ArcgisAPIServiceHelper
from .plot_tools import PlotTools
from .unit_converter import UnitConverter
from .input_validator import InputValidator
from .inventory_validator import InventoryValidator
from .spatial_join_methods.base import SpatialJoinMethods
from .model_utils import ModelUtils
from .rapid_utils import RAPIDUtils
from .segmentation_utils import SegmentationUtils
