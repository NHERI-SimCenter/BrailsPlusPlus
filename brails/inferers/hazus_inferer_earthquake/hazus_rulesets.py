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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.

"""
Hazus rulesets transcribed from the FEMA Hazus Inventory Technical Manual.

Structure type distributions, region groupings, and replacement cost
data, each returned by a getter whose docstring names the manual tables
it transcribes. The README beside this module states which editions of
the manual the tables come from.

.. autosummary::

      get_hazus_occ_type_mapping
      get_hazus_state_region_mapping
      get_hazus_height_classes
      get_hazus_year_classes
      get_hazus_sampling_height_bands
      get_hazus_sampling_height_band
      get_hazus_story_count
      get_hazus_height_classes_by_type
      get_hazus_unreinforced_design_levels
      get_hazus_general_type_membership
      get_hazus_base_replacement_cost
      get_hazus_region_to_garage
      get_hazus_income_to_const_class
      get_hazus_height_classes_RES1
"""

from __future__ import annotations

import math

# The region group whose Appendix A tables are stratified by era.
WEST_COAST = 'West Coast'

# The types the Appendix A footnote of the FEMA Hazus Inventory Technical
# Manual 7.0 calls unreinforced: "Unreinforced SBTs URM, C3 and S5 are not
# allowed in West Coast Region where Design Levels are higher than Pre-Code"
# (West Coast tables pp. A-2 to A-12), with Low-Code in place of Pre-Code
# elsewhere. Which levels admit them by region comes from
# get_hazus_unreinforced_design_levels, whose docstring carries the rest.
UNREINFORCED_MASONRY_TYPES = ('S5', 'C3', 'URM')


def get_hazus_occ_type_mapping() -> tuple[dict, dict]:
    """
    Return the Appendix A structure type lists and sampling weights.

    The weights give the percentage of each structure type by occupancy
    class and height band. West Coast tables are further stratified by
    building era, and RES1 tables are organized by state instead of
    height band. Each weight vector aligns position by position with the
    type list of its band.

    Sources, all in Appendix A of the 6.0 edition (the 6.1 appendix no
    longer prints these weights):

    - West Coast non-RES1: Tables A-2 (low-rise) to A-10 (high-rise).
    - West Coast RES1: Tables A-17 to A-19, one per era.
    - Mid-West non-RES1: Tables A-11 to A-13; RES1: Table A-20.
    - East Coast non-RES1: Tables A-14 to A-16; RES1: Table A-21.

    An empty weight list means the manual table has no row for that
    occupancy and band, so no such stock was observed in the region.

    Returns:
        tuple[dict, dict]:
            (type_lists, type_weights), each keyed by region group
            ('West Coast', 'Mid-West', 'East Coast'). type_lists maps
            height bands to structure type rows; type_weights maps
            occupancy classes to percentage vectors over those rows.
    """
    #
    # The global hierarchy of json components are REGION - OCC_CLASS - (HEIGHT_CLASS) - (YEAR_CLASS) - (STATE)
    #
    # HEIGHT_CLASS exists only for nonRES1
    # YEAR_CLASS exists only for West-coast
    # STATE exists only for RES1

    #
    # WEST COAST
    #

    # json hierarchy for nonRES1 buildings is OCC_CLASS - HEIGHT_CLASS - YEAR_CLASS
    # json hierarchy for RES1 buildings is    OCC_CLASS - YEAR_CLASS - STATE

    type_lists_west = {
        "low_rise": [
            "W1",
            "W2",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "C1",
            "C2",
            "C3",
            "PC1",
            "PC2",
            "RM1",
            "RM2",
            "URM",
            "MH",
        ],
        "mid_rise": [
            "S1",
            "S2",
            "S4",
            "S5",
            "C1",
            "C2",
            "C3",
            "PC2",
            "RM1",
            "RM2",
            "URM",
        ],
        "high_rise": ["S1", "S2", "S4", "S5", "C1", "C2", "C3", "PC2", "RM2"],
        "RES1": ["W1", "S3", "S5", "C2", "RM1", "URM"],
    }

    type_weights_west = {
        "RES1": {
            "pre_1950": {
                "Alaska": [99, 0, 0, 1, 0, 0],
                "Arizona": [60, 0, 0, 0, 25, 16],
                "California": [99, 0, 0, 0, 1, 0],
                "Colorado": [76, 0, 0, 0, 15, 9],
                "Hawaii": [92, 0, 0, 1, 4, 3],
                "Idaho": [95, 0, 0, 0, 3, 2],
                "Montana": [98, 0, 0, 0, 1, 1],
                "New Mexico": [74, 0, 0, 0, 16, 10],
                "Nevada": [97, 0, 0, 0, 2, 1],
                "Oregon": [99, 0, 0, 0, 1, 0],
                "Utah": [82, 0, 0, 0, 11, 7],
                "Washington": [98, 0, 0, 0, 1, 1],
                "Wyoming": [92, 0, 0, 0, 5, 3],
            },
            "1950_to_1970": {
                "Alaska": [99, 0, 0, 1, 0, 0],
                "Arizona": [60, 0, 0, 0, 36, 4],
                "California": [99, 0, 0, 0, 1, 0],
                "Colorado": [76, 0, 0, 0, 21, 3],
                "Hawaii": [92, 0, 0, 1, 6, 1],
                "Idaho": [95, 0, 0, 0, 4, 1],
                "Montana": [98, 0, 0, 0, 2, 0],
                "New Mexico": [74, 0, 0, 0, 23, 3],
                "Nevada": [97, 0, 0, 0, 3, 0],
                "Oregon": [99, 0, 0, 0, 1, 0],
                "Utah": [82, 0, 0, 0, 16, 2],
                "Washington": [98, 0, 0, 0, 2, 0],
                "Wyoming": [92, 0, 0, 0, 7, 1],
            },
            "post_1970": {
                "Alaska": [99, 0, 0, 1, 0, 0],
                "Arizona": [60, 0, 0, 0, 40, 0],
                "American Samoa": [92, 0, 0, 1, 7, 0],
                "California": [99, 0, 0, 0, 1, 0],
                "Colorado": [76, 0, 0, 0, 24, 0],
                "Guam": [92, 0, 0, 1, 7, 0],
                "Hawaii": [92, 0, 0, 1, 7, 0],
                "Idaho": [95, 0, 0, 0, 5, 0],
                "Northern Mariana Islands": [92, 0, 0, 1, 7, 0],
                "Montana": [98, 0, 0, 0, 2, 0],
                "New Mexico": [74, 0, 0, 0, 26, 0],
                "Nevada": [97, 0, 0, 0, 3, 0],
                "Oregon": [99, 0, 0, 0, 1, 0],
                "Utah": [82, 0, 0, 0, 18, 0],
                "Washington": [98, 0, 0, 0, 2, 0],
                "Wyoming": [92, 0, 0, 0, 8, 0],
            },
        },
        "RES2": {
            "low_rise": {
                "pre_1950": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],
                "1950_to_1970": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],
                "post_1970": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],
            },
            "mid_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "RES3": {
            "low_rise": {
                "pre_1950": [73, 0, 1, 1, 1, 0, 6, 0, 3, 3, 0, 0, 1, 0, 9, 2],
                "1950_to_1970": [72, 0, 1, 2, 2, 0, 1, 0, 6, 2, 0, 0, 8, 0, 3, 3],
                "post_1970": [73, 0, 0, 0, 2, 3, 0, 0, 6, 1, 0, 1, 9, 0, 0, 5],
            },
            "mid_rise": {
                "pre_1950": [15, 4, 5, 0, 1, 19, 25, 0, 8, 0, 23],
                "1950_to_1970": [10, 15, 6, 0, 4, 37, 0, 1, 21, 6, 0],
                "post_1970": [9, 23, 8, 0, 10, 28, 0, 7, 12, 3, 0],
            },
            "high_rise": {
                "pre_1950": [39, 1, 2, 0, 8, 24, 23, 3, 0],
                "1950_to_1970": [30, 21, 6, 0, 13, 24, 0, 3, 3],
                "post_1970": [44, 6, 5, 0, 18, 20, 0, 5, 2],
            },
        },
        "RES4": {
            "low_rise": {
                "pre_1950": [34, 0, 2, 1, 2, 1, 19, 0, 16, 3, 0, 0, 4, 0, 18, 0],
                "1950_to_1970": [55, 0, 1, 2, 2, 2, 3, 0, 11, 2, 0, 0, 18, 1, 3, 0],
                "post_1970": [53, 0, 3, 0, 2, 3, 0, 4, 13, 0, 0, 0, 20, 2, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [18, 4, 12, 0, 1, 20, 20, 0, 8, 0, 17],
                "1950_to_1970": [9, 24, 9, 0, 5, 34, 1, 0, 14, 4, 0],
                "post_1970": [16, 28, 8, 0, 11, 18, 0, 3, 13, 3, 0],
            },
            "high_rise": {
                "pre_1950": [45, 3, 3, 0, 8, 20, 18, 3, 0],
                "1950_to_1970": [48, 10, 9, 0, 12, 19, 0, 1, 1],
                "post_1970": [56, 10, 6, 0, 16, 9, 0, 2, 1],
            },
        },
        "RES5": {
            "low_rise": {
                "pre_1950": [20, 0, 5, 1, 0, 1, 0, 0, 28, 18, 0, 0, 6, 0, 21, 0],
                "1950_to_1970": [39, 0, 3, 3, 0, 1, 8, 0, 16, 6, 0, 0, 18, 1, 5, 0],
                "post_1970": [33, 0, 3, 3, 0, 6, 0, 5, 24, 0, 0, 0, 23, 3, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [16, 1, 5, 0, 0, 40, 20, 0, 0, 0, 18],
                "1950_to_1970": [6, 1, 11, 0, 9, 45, 0, 0, 18, 10, 0],
                "post_1970": [9, 10, 11, 0, 16, 34, 0, 4, 11, 5, 0],
            },
            "high_rise": {
                "pre_1950": [15, 5, 10, 0, 0, 30, 40, 0, 0],
                "1950_to_1970": [20, 15, 25, 0, 30, 5, 0, 0, 5],
                "post_1970": [25, 18, 20, 0, 37, 0, 0, 0, 0],
            },
        },
        "RES6": {
            "low_rise": {
                "pre_1950": [45, 0, 0, 0, 10, 0, 5, 0, 10, 0, 0, 0, 20, 0, 10, 0],
                "1950_to_1970": [70, 0, 0, 0, 3, 1, 1, 0, 5, 0, 0, 0, 20, 0, 0, 0],
                "post_1970": [70, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 20, 0, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [20, 0, 5, 0, 0, 35, 20, 0, 10, 0, 10],
                "1950_to_1970": [15, 10, 15, 0, 5, 25, 0, 0, 25, 5, 0],
                "post_1970": [25, 10, 15, 0, 10, 35, 0, 0, 5, 0, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "AGR1": {
            "low_rise": {
                "pre_1950": [56, 0, 3, 2, 14, 0, 2, 0, 9, 0, 0, 0, 0, 1, 13, 0],
                "1950_to_1970": [51, 0, 4, 8, 12, 0, 0, 0, 2, 0, 10, 0, 11, 2, 0, 0],
                "post_1970": [40, 0, 8, 11, 8, 0, 0, 0, 3, 0, 11, 1, 15, 1, 0, 2],
            },
            "mid_rise": {
                "pre_1950": [0, 0, 0, 0, 0, 25, 75, 0, 0, 0, 0],
                # "1950_to_1970": [0,0,0,0,0,0,0,0,0,0,0],
                # "post_1970": [0,0,0,0,0,0,0,0,0,0,0]
                "1950_to_1970": [],
                "post_1970": [],
            },
            "high_rise": {
                # "pre_1950": [0,0,0,0,0,0,0,0,0],
                # "1950_to_1970": [0,0,0,0,0,0,0,0,0],
                # "post_1970": [0,0,0,0,0,0,0,0,0]
                "pre_1950": [],
                "1950_to_1970": [],
                "post_1970": [],
            },
        },
        "COM1": {
            "low_rise": {
                "pre_1950": [0, 22, 2, 0, 6, 3, 20, 0, 17, 1, 0, 0, 6, 0, 23, 0],
                "1950_to_1970": [0, 34, 3, 1, 3, 2, 4, 0, 13, 5, 10, 1, 18, 2, 4, 0],
                "post_1970": [0, 26, 9, 1, 2, 1, 0, 6, 10, 1, 15, 5, 21, 3, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [8, 6, 3, 0, 0, 21, 34, 0, 11, 1, 16],
                "1950_to_1970": [7, 25, 5, 0, 3, 31, 0, 0, 22, 7, 0],
                "post_1970": [34, 9, 3, 0, 12, 17, 0, 5, 15, 5, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "COM2": {
            "low_rise": {
                "pre_1950": [0, 8, 3, 0, 4, 2, 41, 0, 18, 1, 3, 0, 5, 2, 13, 0],
                "1950_to_1970": [0, 12, 4, 5, 5, 3, 3, 0, 18, 0, 22, 1, 19, 4, 4, 0],
                "post_1970": [0, 8, 4, 1, 3, 4, 0, 2, 12, 0, 41, 3, 19, 3, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [8, 0, 0, 0, 0, 27, 53, 0, 5, 0, 7],
                "1950_to_1970": [21, 3, 0, 0, 2, 34, 0, 1, 34, 5, 0],
                "post_1970": [20, 17, 0, 0, 15, 10, 0, 8, 15, 15, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "COM3": {
            "low_rise": {
                "pre_1950": [0, 28, 1, 1, 3, 0, 18, 0, 7, 0, 1, 0, 8, 0, 33, 0],
                "1950_to_1970": [0, 12, 3, 5, 5, 2, 3, 0, 23, 4, 12, 1, 22, 4, 4, 0],
                "post_1970": [0, 13, 3, 2, 2, 3, 0, 3, 13, 0, 20, 5, 34, 2, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [18, 0, 0, 0, 0, 22, 42, 0, 5, 0, 13],
                "1950_to_1970": [10, 3, 0, 0, 0, 28, 0, 0, 54, 5, 0],
                "post_1970": [11, 17, 3, 0, 10, 17, 0, 12, 17, 13, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "COM4": {
            "low_rise": {
                "pre_1950": [0, 27, 2, 1, 3, 0, 19, 0, 15, 0, 0, 0, 7, 0, 26, 0],
                "1950_to_1970": [0, 34, 3, 3, 1, 2, 3, 0, 17, 5, 3, 0, 23, 4, 2, 0],
                "post_1970": [0, 35, 3, 2, 1, 3, 0, 4, 15, 0, 8, 3, 24, 2, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [25, 7, 10, 0, 2, 22, 16, 0, 9, 0, 9],
                "1950_to_1970": [17, 18, 9, 0, 9, 18, 0, 2, 23, 4, 0],
                "post_1970": [37, 10, 12, 0, 9, 15, 0, 3, 9, 5, 0],
            },
            "high_rise": {
                "pre_1950": [47, 10, 4, 0, 1, 21, 16, 1, 0],
                "1950_to_1970": [40, 26, 18, 0, 6, 7, 0, 1, 2],
                "post_1970": [56, 10, 14, 0, 14, 5, 0, 1, 0],
            },
        },
        "COM5": {
            "low_rise": {
                "pre_1950": [0, 27, 2, 1, 3, 0, 19, 0, 15, 0, 0, 0, 7, 0, 26, 0],
                "1950_to_1970": [0, 34, 3, 3, 1, 2, 3, 0, 17, 5, 3, 0, 23, 4, 2, 0],
                "post_1970": [0, 35, 3, 2, 1, 3, 0, 4, 15, 0, 8, 3, 24, 2, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [25, 7, 10, 0, 2, 22, 16, 0, 9, 0, 9],
                "1950_to_1970": [17, 18, 9, 0, 9, 18, 0, 2, 23, 4, 0],
                "post_1970": [37, 10, 12, 0, 9, 15, 0, 3, 9, 5, 0],
            },
            "high_rise": {
                "pre_1950": [47, 10, 4, 0, 1, 21, 16, 1, 0],
                "1950_to_1970": [40, 26, 18, 0, 6, 7, 0, 1, 2],
                "post_1970": [54, 10, 15, 0, 15, 5, 0, 1, 0],
            },
        },
        "COM6": {
            "low_rise": {
                "pre_1950": [0, 8, 5, 2, 11, 0, 11, 0, 27, 2, 1, 0, 27, 0, 6, 0],
                "1950_to_1970": [0, 32, 5, 2, 4, 3, 0, 0, 16, 6, 0, 0, 28, 4, 0, 0],
                "post_1970": [0, 31, 6, 1, 1, 7, 0, 4, 13, 0, 7, 0, 28, 2, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [18, 4, 6, 0, 1, 35, 19, 0, 8, 0, 9],
                "1950_to_1970": [14, 10, 14, 0, 5, 23, 0, 3, 23, 8, 0],
                "post_1970": [25, 9, 15, 0, 10, 33, 0, 1, 6, 1, 0],
            },
            "high_rise": {
                "pre_1950": [56, 9, 1, 0, 1, 24, 8, 1, 0],
                "1950_to_1970": [35, 27, 17, 0, 4, 15, 0, 1, 1],
                "post_1970": [45, 6, 19, 0, 13, 17, 0, 0, 0],
            },
        },
        "COM7": {
            "low_rise": {
                "pre_1950": [0, 25, 5, 2, 10, 0, 10, 0, 15, 2, 1, 0, 20, 0, 10, 0],
                "1950_to_1970": [0, 46, 13, 1, 3, 3, 0, 0, 9, 0, 0, 0, 20, 0, 5, 0],
                "post_1970": [0, 47, 16, 0, 0, 5, 0, 4, 6, 0, 2, 0, 20, 0, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [20, 5, 5, 0, 0, 30, 20, 0, 10, 0, 10],
                "1950_to_1970": [15, 10, 15, 0, 5, 25, 0, 0, 25, 5, 0],
                "post_1970": [25, 10, 15, 0, 10, 35, 0, 0, 5, 0, 0],
            },
            "high_rise": {
                # "pre_1950": [0,0,0,0,0,0,0,0,0],
                # "1950_to_1970": [0,0,0,0,0,0,0,0,0],
                # "post_1970": [0,0,0,0,0,0,0,0,0]
                "pre_1950": [],
                "1950_to_1970": [],
                "post_1970": [],
            },
        },
        "COM8": {
            "low_rise": {
                "pre_1950": [0, 8, 12, 1, 2, 3, 16, 0, 27, 4, 0, 0, 5, 1, 21, 0],
                "1950_to_1970": [0, 13, 17, 12, 3, 3, 0, 0, 13, 6, 0, 0, 30, 3, 0, 0],
                "post_1970": [0, 4, 23, 8, 1, 3, 0, 2, 15, 0, 4, 1, 32, 7, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [25, 0, 20, 0, 0, 40, 5, 0, 0, 0, 10],
                "1950_to_1970": [5, 0, 28, 0, 0, 52, 0, 0, 10, 5, 0],
                "post_1970": [0, 10, 0, 0, 90, 0, 0, 0, 0, 0, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "COM9": {
            "low_rise": {
                "pre_1950": [0, 5, 20, 7, 0, 0, 15, 0, 20, 3, 0, 0, 10, 0, 20, 0],
                "1950_to_1970": [0, 10, 10, 30, 0, 0, 5, 0, 10, 0, 5, 0, 30, 0, 0, 0],
                "post_1970": [0, 5, 27, 20, 0, 0, 0, 0, 12, 0, 4, 0, 27, 5, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [30, 0, 10, 0, 0, 40, 10, 0, 0, 0, 10],
                "1950_to_1970": [5, 0, 30, 0, 0, 50, 0, 0, 10, 5, 0],
                "post_1970": [0, 10, 0, 0, 90, 0, 0, 0, 0, 0, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "COM10": {
            "low_rise": {
                "pre_1950": [0, 0, 0, 8, 0, 8, 18, 0, 43, 7, 0, 1, 6, 3, 6, 0],
                "1950_to_1970": [0, 0, 5, 8, 0, 20, 0, 0, 34, 0, 0, 5, 20, 6, 2, 0],
                "post_1970": [0, 0, 8, 8, 0, 6, 0, 3, 49, 0, 3, 13, 7, 3, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [0, 10, 5, 0, 2, 55, 18, 0, 3, 2, 5],
                "1950_to_1970": [5, 8, 8, 0, 7, 39, 0, 8, 18, 7, 0],
                "post_1970": [4, 8, 3, 0, 4, 66, 0, 8, 6, 1, 0],
            },
            "high_rise": {
                # "pre_1950": [0,0,0,0,0,0,0,0,0],
                # "1950_to_1970": [0,0,0,0,0,0,0,0,0],
                # "post_1970": [0,0,0,0,0,0,0,0,0]
                "pre_1950": [],
                "1950_to_1970": [],
                "post_1970": [],
            },
        },
        "EDU1": {
            "low_rise": {
                "pre_1950": [11, 0, 6, 0, 3, 3, 21, 0, 21, 4, 0, 0, 9, 0, 22, 0],
                "1950_to_1970": [25, 0, 3, 4, 5, 4, 0, 0, 20, 0, 4, 2, 29, 4, 0, 0],
                "post_1970": [24, 0, 9, 6, 1, 5, 0, 3, 16, 3, 4, 3, 21, 5, 0, 0],
            },
            "mid_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "EDU2": {
            "low_rise": {
                "pre_1950": [2, 0, 5, 10, 0, 5, 15, 0, 20, 0, 0, 0, 20, 5, 18, 0],
                "1950_to_1970": [5, 0, 2, 12, 0, 5, 0, 0, 20, 0, 0, 0, 50, 6, 0, 0],
                "post_1970": [5, 0, 10, 10, 0, 5, 0, 0, 20, 0, 5, 0, 40, 5, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [10, 0, 20, 0, 0, 60, 3, 0, 5, 0, 2],
                "1950_to_1970": [20, 0, 15, 0, 5, 35, 0, 0, 15, 10, 0],
                "post_1970": [20, 5, 10, 0, 25, 25, 0, 0, 10, 5, 0],
            },
            "high_rise": {
                "pre_1950": [5, 5, 35, 0, 0, 40, 15, 0, 0],
                "1950_to_1970": [35, 20, 20, 0, 25, 0, 0, 0, 0],
                "post_1970": [30, 10, 10, 0, 50, 0, 0, 0, 0],
            },
        },
        "GOV1": {
            "low_rise": {
                "pre_1950": [0, 9, 8, 1, 3, 4, 12, 0, 42, 4, 0, 0, 6, 0, 11, 0],
                "1950_to_1970": [0, 21, 6, 3, 2, 2, 0, 0, 26, 5, 4, 2, 27, 2, 0, 0],
                "post_1970": [0, 8, 15, 4, 3, 7, 0, 2, 32, 0, 0, 4, 16, 9, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [30, 15, 5, 0, 3, 23, 10, 0, 4, 0, 10],
                "1950_to_1970": [15, 6, 15, 0, 11, 28, 0, 2, 18, 5, 0],
                "post_1970": [25, 11, 15, 0, 22, 12, 0, 4, 9, 2, 0],
            },
            "high_rise": {
                "pre_1950": [53, 5, 5, 0, 3, 30, 3, 1, 0],
                "1950_to_1970": [46, 13, 22, 0, 10, 8, 0, 0, 1],
                "post_1970": [52, 14, 14, 0, 14, 6, 0, 0, 0],
            },
        },
        "GOV2": {
            "low_rise": {
                "pre_1950": [45, 0, 0, 0, 0, 2, 0, 0, 37, 0, 0, 0, 3, 0, 13, 0],
                "1950_to_1970": [50, 0, 0, 0, 0, 0, 0, 0, 13, 0, 7, 0, 20, 10, 0, 0],
                "post_1970": [40, 0, 3, 7, 0, 23, 0, 0, 10, 0, 0, 7, 3, 7, 0, 0],
            },
            "mid_rise": {
                # "pre_1950": [0,0,0,0,0,0,0,0,0,0,0],
                "pre_1950": [],
                "1950_to_1970": [5, 10, 10, 0, 5, 60, 0, 0, 0, 10, 0],
                "post_1970": [25, 20, 35, 0, 0, 20, 0, 0, 0, 0, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "IND1": {
            "low_rise": {
                "pre_1950": [0, 3, 29, 13, 2, 2, 15, 0, 14, 7, 1, 0, 4, 2, 8, 0],
                "1950_to_1970": [0, 10, 25, 30, 3, 0, 0, 7, 14, 0, 0, 0, 9, 2, 0, 0],
                "post_1970": [0, 11, 19, 28, 3, 2, 0, 1, 9, 0, 11, 3, 11, 1, 0, 1],
            },
            "mid_rise": {
                # "pre_1950": [0,0,0,0,0,0,0,0,0,0,0],
                "pre_1950": [],
                "1950_to_1970": [0, 10, 20, 0, 0, 40, 0, 0, 20, 10, 0],
                # "post_1970": [0,0,0,0,0,0,0,0,0,0,0]
                "post_1970": [],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "IND2": {
            "low_rise": {
                "pre_1950": [0, 4, 14, 8, 22, 1, 18, 0, 16, 1, 1, 0, 2, 0, 13, 0],
                "1950_to_1970": [0, 8, 5, 14, 17, 4, 0, 0, 10, 5, 22, 3, 12, 0, 0, 0],
                "post_1970": [0, 3, 13, 9, 6, 3, 0, 0, 10, 0, 41, 3, 12, 0, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [0, 0, 10, 0, 0, 5, 75, 0, 0, 0, 10],
                "1950_to_1970": [0, 15, 10, 0, 0, 50, 0, 0, 20, 5, 0],
                # "post_1970": [0,0,0,0,0,0,0,0,0,0,0]
                "post_1970": [],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "IND3": {
            "low_rise": {
                "pre_1950": [0, 1, 18, 8, 3, 3, 20, 0, 22, 0, 2, 0, 3, 0, 20, 0],
                "1950_to_1970": [0, 0, 14, 16, 6, 1, 0, 5, 17, 0, 28, 1, 10, 2, 0, 0],
                "post_1970": [0, 2, 15, 10, 5, 3, 0, 0, 12, 0, 28, 7, 18, 0, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [32, 3, 1, 0, 1, 14, 41, 0, 3, 0, 5],
                "1950_to_1970": [11, 4, 10, 0, 30, 20, 0, 1, 15, 9, 0],
                "post_1970": [62, 5, 1, 0, 23, 4, 0, 1, 3, 1, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "IND4": {
            "low_rise": {
                "pre_1950": [0, 2, 24, 12, 7, 2, 13, 0, 16, 0, 2, 0, 2, 6, 14, 0],
                "1950_to_1970": [0, 0, 18, 25, 9, 0, 0, 11, 10, 0, 7, 0, 15, 3, 0, 2],
                "post_1970": [0, 1, 26, 18, 5, 4, 0, 1, 11, 1, 12, 5, 15, 1, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [25, 3, 1, 0, 0, 9, 52, 0, 0, 0, 10],
                "1950_to_1970": [0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0],
                "post_1970": [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "IND5": {
            "low_rise": {
                "pre_1950": [0, 0, 21, 5, 5, 0, 3, 0, 35, 2, 10, 2, 15, 0, 2, 0],
                "1950_to_1970": [0, 0, 4, 9, 3, 2, 0, 4, 20, 0, 35, 3, 15, 4, 0, 1],
                "post_1970": [0, 1, 12, 8, 2, 3, 0, 0, 10, 0, 38, 7, 17, 1, 0, 1],
            },
            "mid_rise": {
                "pre_1950": [35, 10, 0, 0, 0, 30, 5, 0, 20, 0, 0],
                "1950_to_1970": [10, 5, 13, 0, 0, 32, 0, 0, 30, 10, 0],
                "post_1970": [18, 14, 3, 0, 34, 13, 0, 5, 10, 3, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "IND6": {
            "low_rise": {
                "pre_1950": [0, 32, 3, 2, 10, 0, 18, 0, 8, 7, 0, 0, 0, 0, 13, 7],
                "1950_to_1970": [0, 30, 0, 1, 15, 0, 0, 0, 7, 0, 4, 0, 20, 3, 0, 20],
                "post_1970": [0, 30, 4, 6, 11, 0, 0, 0, 8, 0, 16, 6, 14, 0, 0, 5],
            },
            "mid_rise": {
                "pre_1950": [0, 0, 0, 0, 0, 20, 80, 0, 0, 0, 0],
                # "1950_to_1970": [0,0,0,0,0,0,0,0,0,0,0],
                # "post_1970": [0,0,0,0,0,0,0,0,0,0,0]
                "1950_to_1970": [],
                "post_1970": [],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
        "REL1": {
            "low_rise": {
                "pre_1950": [22, 0, 8, 0, 2, 0, 21, 0, 15, 5, 0, 0, 8, 0, 19, 0],
                "1950_to_1970": [20, 0, 4, 1, 3, 3, 0, 0, 24, 0, 4, 0, 37, 4, 0, 0],
                "post_1970": [23, 0, 12, 3, 1, 6, 0, 0, 26, 0, 1, 3, 22, 3, 0, 0],
            },
            "mid_rise": {
                "pre_1950": [0, 0, 0, 0, 0, 10, 90, 0, 0, 0, 0],
                "1950_to_1970": [0, 0, 0, 0, 0, 80, 0, 0, 10, 10, 0],
                "post_1970": [0, 5, 0, 0, 90, 0, 0, 0, 0, 5, 0],
            },
            "high_rise": {"pre_1950": [], "1950_to_1970": [], "post_1970": []},
        },
    }

    #
    # Mid-west
    #

    # hierarchy for nonRES1 buildings is OCC_CLASS - HEIGHT_CLASS
    # hierarchy for RES1 buildings is    OCC_CLASS - STATE

    type_lists_mid = {
        "low_rise": [
            "W1",
            "W2",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "C1",
            "C2",
            "C3",
            "PC1",
            "PC2",
            "RM1",
            "RM2",
            "URM",
            "MH",
        ],
        "mid_rise": [
            "S1",
            "S2",
            "S4",
            "S5",
            "C1",
            "C2",
            "C3",
            "PC2",
            "RM1",
            "RM2",
            "URM",
        ],
        "high_rise": ["S1", "S2", "S4", "S5", "C1", "C2", "C3", "PC2", "RM2"],
        "RES1": ["W1", "C2", "URM"],
    }

    type_weights_mid = {
        "RES1": {
            "Arkansas": [87, 0, 13],
            "Iowa": [92, 0, 8],
            "Illinois": [77, 1, 22],
            "Indiana": [80, 0, 20],
            "Kansas": [91, 0, 9],
            "Kentucky": [88, 0, 12],
            "Louisiana": [89, 0, 11],
            "Michigan": [86, 0, 14],
            "Minnesota": [95, 1, 4],
            "Missouri": [76, 0, 24],
            "Mississippi": [94, 0, 6],
            "North Dakota": [98, 0, 2],
            "Nebraska": [89, 1, 10],
            "Ohio": [76, 0, 24],
            "Oklahoma": [71, 0, 29],
            "South Dakota": [97, 0, 3],
            "Tennessee": [90, 0, 10],
            "Texas": [100, 0, 0],
            "Wisconsin": [90, 0, 10],
        },
        "RES2": {
            "low_rise": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],
            "mid_rise": [],
            "high_rise": [],
        },
        "RES3": {
            "low_rise": [75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 23, 0],
            "mid_rise": [0, 10, 7, 3, 14, 39, 0, 7, 0, 2, 18],
            "high_rise": [3, 13, 4, 0, 16, 44, 7, 7, 6],
        },
        "RES4": {
            "low_rise": [50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 45, 0],
            "mid_rise": [0, 10, 7, 3, 14, 37, 2, 7, 0, 2, 18],
            "high_rise": [3, 13, 4, 0, 16, 44, 7, 7, 6],
        },
        "RES5": {
            "low_rise": [20, 0, 0, 0, 0, 0, 0, 4, 13, 2, 22, 4, 2, 0, 33, 0],
            "mid_rise": [0, 0, 0, 0, 25, 62, 2, 11, 0, 0, 0],
            "high_rise": [0, 0, 0, 0, 26, 74, 0, 0, 0],
        },
        "RES6": {
            "low_rise": [90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "AGR1": {
            "low_rise": [0, 10, 2, 4, 11, 6, 7, 2, 10, 2, 14, 2, 2, 0, 28, 0],
            "mid_rise": [0, 7, 3, 0, 14, 37, 2, 7, 0, 3, 27],
            "high_rise": [2, 6, 2, 0, 16, 44, 11, 11, 8],
        },
        "COM1": {
            "low_rise": [0, 30, 2, 4, 11, 6, 7, 0, 5, 0, 5, 0, 2, 0, 28, 0],
            "mid_rise": [3, 20, 16, 6, 11, 27, 2, 5, 0, 2, 8],
            "high_rise": [],
        },
        "COM2": {
            "low_rise": [0, 10, 2, 4, 11, 6, 7, 2, 10, 2, 14, 2, 2, 0, 28, 0],
            "mid_rise": [0, 7, 3, 0, 14, 37, 2, 7, 0, 3, 27],
            "high_rise": [],
        },
        "COM3": {
            "low_rise": [0, 30, 2, 4, 11, 6, 7, 0, 5, 0, 5, 0, 2, 0, 28, 0],
            "mid_rise": [3, 20, 16, 6, 11, 27, 2, 5, 0, 2, 8],
            "high_rise": [],
        },
        "COM4": {
            "low_rise": [0, 30, 2, 4, 11, 6, 7, 0, 5, 0, 5, 0, 2, 0, 28, 0],
            "mid_rise": [3, 20, 16, 6, 11, 27, 2, 5, 0, 2, 8],
            "high_rise": [7, 29, 9, 0, 12, 32, 4, 4, 3],
        },
        "COM5": {
            "low_rise": [0, 30, 2, 4, 11, 6, 7, 0, 5, 0, 5, 0, 2, 0, 28, 0],
            "mid_rise": [3, 20, 16, 6, 11, 27, 2, 5, 0, 2, 8],
            "high_rise": [7, 29, 9, 0, 12, 32, 4, 4, 3],
        },
        "COM6": {
            "low_rise": [0, 0, 0, 2, 4, 2, 2, 6, 21, 4, 33, 6, 2, 0, 18, 0],
            "mid_rise": [3, 20, 16, 6, 12, 30, 2, 6, 0, 0, 5],
            "high_rise": [7, 29, 9, 0, 13, 36, 2, 2, 2],
        },
        "COM7": {
            "low_rise": [0, 30, 2, 4, 11, 6, 7, 0, 5, 0, 5, 0, 2, 0, 28, 0],
            "mid_rise": [3, 20, 16, 6, 11, 27, 2, 5, 0, 2, 8],
            "high_rise": [7, 29, 9, 0, 12, 32, 4, 4, 3],
        },
        "COM8": {
            "low_rise": [0, 30, 2, 4, 11, 6, 7, 0, 5, 0, 5, 0, 2, 0, 28, 0],
            "mid_rise": [3, 20, 16, 6, 11, 27, 2, 5, 0, 2, 8],
            "high_rise": [],
        },
        "COM9": {
            "low_rise": [0, 0, 2, 6, 14, 8, 10, 4, 13, 2, 22, 4, 0, 0, 15, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "COM10": {
            "low_rise": [0, 0, 2, 4, 11, 6, 7, 6, 21, 4, 33, 6, 0, 0, 0, 0],
            "mid_rise": [2, 14, 10, 4, 17, 43, 2, 8, 0, 0, 0],
            "high_rise": [5, 19, 6, 0, 18, 52, 0, 0, 0],
        },
        "EDU1": {
            "low_rise": [0, 10, 5, 12, 0, 0, 0, 5, 7, 0, 0, 0, 11, 0, 50, 0],
            "mid_rise": [],
            "high_rise": [],
        },
        "EDU2": {
            "low_rise": [0, 14, 6, 12, 0, 0, 2, 8, 11, 0, 0, 0, 0, 10, 37, 0],
            "mid_rise": [7, 14, 0, 0, 9, 13, 0, 0, 0, 13, 44],
            # "high_rise": [0,0,0,0,0,0,0,0,0]
            "high_rise": [],
        },
        "GOV1": {
            "low_rise": [0, 15, 14, 21, 0, 0, 0, 7, 6, 0, 4, 0, 3, 0, 30, 0],
            "mid_rise": [20, 24, 0, 0, 11, 9, 0, 0, 0, 5, 31],
            # "high_rise": [0,0,0,0,0,0,0,0,0]
            "high_rise": [],
        },
        "GOV2": {
            "low_rise": [0, 14, 7, 17, 0, 0, 0, 4, 12, 0, 0, 0, 0, 3, 43, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "IND1": {
            "low_rise": [0, 0, 5, 10, 25, 13, 17, 2, 7, 2, 12, 2, 0, 0, 5, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "IND2": {
            "low_rise": [0, 10, 2, 4, 11, 6, 7, 2, 10, 2, 14, 2, 3, 0, 27, 0],
            "mid_rise": [0, 7, 3, 0, 14, 37, 2, 7, 0, 3, 27],
            "high_rise": [],
        },
        "IND3": {
            "low_rise": [0, 10, 2, 4, 11, 6, 7, 2, 10, 2, 14, 2, 3, 0, 27, 0],
            "mid_rise": [0, 7, 3, 0, 14, 37, 2, 7, 0, 3, 27],
            "high_rise": [],
        },
        "IND4": {
            "low_rise": [0, 0, 5, 10, 25, 13, 17, 2, 7, 2, 12, 2, 0, 0, 5, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "IND5": {
            "low_rise": [0, 10, 2, 4, 11, 6, 7, 2, 10, 2, 14, 2, 2, 0, 28, 0],
            "mid_rise": [0, 7, 3, 0, 14, 37, 2, 7, 0, 3, 27],
            "high_rise": [],
        },
        "IND6": {
            "low_rise": [0, 30, 2, 4, 11, 6, 7, 0, 5, 0, 5, 0, 2, 0, 28, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "REL1": {
            "low_rise": [30, 0, 0, 3, 5, 3, 4, 0, 5, 0, 5, 0, 2, 2, 41, 0],
            "mid_rise": [3, 20, 16, 6, 11, 27, 2, 5, 0, 2, 8],
            "high_rise": [],
        },
    }

    #
    # East
    #

    # hierarchy for nonRES1 buildings is OCC_CLASS - HEIGHT_CLASS
    # hierarchy for RES1 buildings is    OCC_CLASS - STATE

    type_lists_east = {
        "low_rise": [
            "W1",
            "W2",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "C1",
            "C2",
            "C3",
            "PC1",
            "PC2",
            "RM1",
            "RM2",
            "URM",
            "MH",
        ],
        "mid_rise": [
            "S1",
            "S2",
            "S4",
            "S5",
            "C1",
            "C2",
            "C3",
            "PC2",
            "RM1",
            "RM2",
            "URM",
        ],
        "high_rise": ["S1", "S2", "S4", "S5", "C1", "C2", "C3", "PC2", "RM2"],
        "RES1": ["W1", "C2", "URM"],
    }

    type_weights_east = {
        "RES1": {
            "Alabama": [95, 0, 5],
            "Connecticut": [96, 0, 4],
            "District of Columbia": [21, 3, 76],
            "Delaware": [71, 1, 28],
            "Florida": [25, 5, 70],
            "Georgia": [93, 0, 7],
            "Massachusetts": [96, 0, 4],
            "Maryland": [71, 1, 28],
            "Maine": [99, 0, 1],
            "North Carolina": [90, 0, 10],
            "New Hampshire": [97, 1, 2],
            "New Jersey": [91, 0, 9],
            "New York": [85, 1, 14],
            "Pennsylvania": [66, 0, 34],
            "Puerto Rico": [24, 0, 76],
            "Rhode Island": [98, 0, 2],
            "South Carolina": [92, 0, 8],
            "Virginia": [75, 0, 25],
            "Vermont": [96, 2, 2],
            "U.S. Virgin Islands": [23, 0, 77],
            "West Virginia": [72, 0, 28],
        },
        "RES2": {
            "low_rise": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],
            "mid_rise": [],
            "high_rise": [],
        },
        "RES3": {
            "low_rise": [62, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 0, 5, 4, 22, 0],
            "mid_rise": [3, 4, 0, 0, 6, 3, 0, 14, 0, 13, 57],
            "high_rise": [8, 21, 8, 0, 34, 17, 2, 5, 5],
        },
        "RES4": {
            "low_rise": [48, 0, 5, 4, 0, 0, 4, 8, 4, 0, 3, 3, 3, 3, 15, 0],
            "mid_rise": [9, 12, 0, 3, 18, 9, 2, 11, 0, 7, 29],
            "high_rise": [8, 21, 8, 0, 34, 17, 2, 5, 5],
        },
        "RES5": {
            "low_rise": [7, 0, 7, 6, 0, 0, 6, 17, 6, 3, 8, 6, 5, 5, 24, 0],
            "mid_rise": [7, 10, 0, 3, 23, 11, 3, 12, 0, 5, 26],
            "high_rise": [6, 16, 6, 0, 40, 20, 3, 5, 4],
        },
        "RES6": {
            "low_rise": [22, 0, 11, 8, 0, 0, 8, 8, 3, 2, 4, 3, 5, 4, 22, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "AGR1": {
            "low_rise": [0, 48, 8, 6, 12, 0, 7, 2, 0, 0, 0, 0, 3, 2, 12, 0],
            "mid_rise": [19, 25, 2, 7, 4, 2, 0, 7, 0, 6, 28],
            "high_rise": [7, 4, 18, 0, 20, 42, 0, 0, 9],
        },
        "COM1": {
            "low_rise": [0, 14, 20, 15, 5, 0, 16, 3, 2, 0, 2, 0, 4, 2, 17, 0],
            "mid_rise": [23, 29, 2, 8, 5, 3, 0, 5, 0, 5, 20],
            "high_rise": [],
        },
        "COM2": {
            "low_rise": [0, 10, 21, 15, 7, 0, 16, 3, 2, 0, 2, 0, 3, 4, 17, 0],
            "mid_rise": [23, 30, 3, 8, 4, 3, 0, 5, 0, 5, 19],
            "high_rise": [],
        },
        "COM3": {
            "low_rise": [0, 25, 7, 5, 11, 0, 5, 3, 2, 0, 2, 0, 6, 4, 30, 0],
            "mid_rise": [10, 13, 0, 3, 5, 4, 0, 11, 0, 10, 44],
            "high_rise": [],
        },
        "COM4": {
            "low_rise": [0, 26, 11, 8, 4, 0, 9, 4, 2, 0, 3, 0, 5, 4, 24, 0],
            "mid_rise": [14, 19, 2, 5, 7, 4, 0, 9, 0, 7, 33],
            "high_rise": [15, 36, 15, 0, 15, 8, 0, 2, 9],
        },
        "COM5": {
            "low_rise": [0, 13, 13, 9, 13, 0, 10, 5, 3, 0, 2, 2, 5, 3, 22, 0],
            "mid_rise": [15, 21, 2, 6, 8, 5, 0, 8, 0, 6, 29],
            "high_rise": [15, 36, 15, 0, 15, 8, 0, 2, 9],
        },
        "COM6": {
            "low_rise": [0, 2, 22, 15, 0, 0, 18, 10, 4, 2, 5, 4, 3, 2, 13, 0],
            "mid_rise": [21, 27, 2, 8, 12, 6, 2, 7, 0, 2, 13],
            "high_rise": [14, 35, 14, 0, 17, 8, 2, 2, 8],
        },
        "COM7": {
            "low_rise": [0, 24, 10, 7, 15, 0, 8, 3, 2, 0, 3, 0, 4, 4, 20, 0],
            "mid_rise": [15, 20, 2, 5, 7, 4, 0, 9, 0, 6, 32],
            "high_rise": [15, 38, 15, 0, 14, 8, 0, 2, 8],
        },
        "COM8": {
            "low_rise": [0, 19, 19, 13, 6, 0, 15, 3, 2, 0, 2, 0, 3, 3, 15, 0],
            "mid_rise": [22, 30, 3, 8, 5, 3, 0, 5, 0, 5, 19],
            "high_rise": [],
        },
        "COM9": {
            "low_rise": [0, 5, 20, 13, 12, 2, 16, 7, 2, 0, 3, 3, 3, 2, 12, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "high_rise": [],
        },
        "COM10": {
            "low_rise": [0, 0, 10, 7, 0, 0, 8, 30, 11, 6, 14, 12, 0, 0, 2, 0],
            "mid_rise": [10, 13, 0, 3, 38, 17, 6, 11, 0, 0, 2],
            "high_rise": [5, 12, 5, 0, 43, 21, 4, 6, 4],
        },
        "EDU1": {
            "low_rise": [0, 13, 17, 13, 0, 0, 13, 5, 3, 0, 2, 2, 5, 5, 22, 0],
            "mid_rise": [],
            "high_rise": [],
        },
        "EDU2": {
            "low_rise": [0, 4, 18, 13, 0, 0, 14, 8, 3, 2, 4, 3, 5, 4, 22, 0],
            "mid_rise": [17, 23, 2, 6, 10, 5, 2, 8, 0, 4, 23],
            # "high_rise": [0,0,0,0,0,0,0,0,0]
            "high_rise": [],
        },
        "GOV1": {
            "low_rise": [0, 7, 24, 16, 3, 0, 19, 5, 3, 0, 2, 1, 3, 3, 13, 0],
            "mid_rise": [24, 30, 3, 9, 7, 5, 0, 5, 0, 3, 14],
            # "high_rise": [0,0,0,0,0,0,0,0,0]
            "high_rise": [],
        },
        "GOV2": {
            "low_rise": [0, 8, 16, 11, 4, 0, 13, 8, 3, 2, 4, 3, 4, 5, 19, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "IND1": {
            "low_rise": [0, 5, 22, 15, 4, 2, 17, 7, 3, 0, 3, 3, 3, 3, 13, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "IND2": {
            "low_rise": [0, 10, 15, 9, 15, 0, 11, 5, 3, 0, 2, 2, 4, 5, 19, 0],
            "mid_rise": [22, 28, 2, 8, 10, 5, 2, 6, 0, 3, 14],
            "high_rise": [],
        },
        "IND3": {
            "low_rise": [0, 7, 25, 18, 3, 0, 19, 4, 2, 0, 2, 2, 3, 2, 13, 0],
            "mid_rise": [25, 32, 3, 9, 6, 4, 0, 4, 0, 3, 14],
            "high_rise": [],
        },
        "IND4": {
            "low_rise": [0, 7, 26, 19, 3, 0, 20, 3, 2, 0, 2, 0, 2, 3, 13, 0],
            "mid_rise": [],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "high_rise": [],
        },
        "IND5": {
            "low_rise": [0, 5, 25, 17, 3, 2, 20, 7, 3, 0, 3, 3, 0, 2, 10, 0],
            "mid_rise": [24, 32, 3, 9, 9, 6, 0, 5, 0, 2, 10],
            "high_rise": [],
        },
        "IND6": {
            "low_rise": [0, 10, 21, 14, 7, 2, 16, 5, 2, 0, 2, 2, 2, 3, 14, 0],
            # "mid_rise": [0,0,0,0,0,0,0,0,0,0,0],
            "mid_rise": [],
            "high_rise": [],
        },
        "REL1": {
            "low_rise": [36, 0, 4, 4, 0, 0, 3, 2, 2, 0, 2, 0, 7, 6, 34, 0],
            "mid_rise": [5, 9, 0, 2, 4, 3, 0, 12, 0, 12, 53],
            "high_rise": [],
        },
    }

    # known issue (?) GOV1-low_rise sums up to 99
    type_weights_east["GOV1"]["low_rise"] = [
        p / 99.0 * 100.0 for p in type_weights_east["GOV1"]["low_rise"]
    ]  # sy - it sums up to 99. See table A

    # if you want to be specific in RES3
    # for res3 in ["RES3A", "RES3B", "RES3C", "RES3D", "RES3E", "RES3F"]:
    #     type_weights_east[res3] = type_weights_east["RES3"]
    #     type_weights_mid[res3] = type_lists_mid["RES3"]
    #     type_weights_east[res3] = type_lists_east["RES3"]

    type_lists = {}
    type_lists["West Coast"] = type_lists_west
    type_lists["Mid-West"] = type_lists_mid
    type_lists["East Coast"] = type_lists_east

    type_weights = {}
    type_weights["West Coast"] = type_weights_west
    type_weights["Mid-West"] = type_weights_mid
    type_weights["East Coast"] = type_weights_east

    return type_lists, type_weights


def get_hazus_state_region_mapping() -> dict[str, dict[str, str]]:
    """
    Return the census region, division, and Hazus region group by state.

    CensusRegion and CensusDivision transcribe Table 3-1 of the manual
    and feed the building value tables. RegionGroup transcribes Table
    A-22 of the 6.0 edition (A-21 in 6.1) and selects the Appendix A
    structure type tables.

    Returns:
        dict[str, dict[str, str]]:
            State name to a dict with keys 'CensusRegion',
            'CensusDivision', and 'RegionGroup'.
    """
    return {
        "Connecticut": {
            "CensusRegion": "Northeast",
            "CensusDivision": "New England",
            "RegionGroup": "East Coast",
        },
        "Massachusetts": {
            "CensusRegion": "Northeast",
            "CensusDivision": "New England",
            "RegionGroup": "East Coast",
        },
        "Maine": {
            "CensusRegion": "Northeast",
            "CensusDivision": "New England",
            "RegionGroup": "East Coast",
        },
        "New Hampshire": {
            "CensusRegion": "Northeast",
            "CensusDivision": "New England",
            "RegionGroup": "East Coast",
        },
        "Rhode Island": {
            "CensusRegion": "Northeast",
            "CensusDivision": "New England",
            "RegionGroup": "East Coast",
        },
        "Vermont": {
            "CensusRegion": "Northeast",
            "CensusDivision": "New England",
            "RegionGroup": "East Coast",
        },
        "New Jersey": {
            "CensusRegion": "Northeast",
            "CensusDivision": "Middle Atlantic",
            "RegionGroup": "East Coast",
        },
        "New York": {
            "CensusRegion": "Northeast",
            "CensusDivision": "Middle Atlantic",
            "RegionGroup": "East Coast",
        },
        "Pennsylvania": {
            "CensusRegion": "Northeast",
            "CensusDivision": "Middle Atlantic",
            "RegionGroup": "East Coast",
        },
        "Illinois": {
            "CensusRegion": "Midwest",
            "CensusDivision": "East North Central",
            "RegionGroup": "Mid-West",
        },
        "Indiana": {
            "CensusRegion": "Midwest",
            "CensusDivision": "East North Central",
            "RegionGroup": "Mid-West",
        },
        "Michigan": {
            "CensusRegion": "Midwest",
            "CensusDivision": "East North Central",
            "RegionGroup": "Mid-West",
        },
        "Ohio": {
            "CensusRegion": "Midwest",
            "CensusDivision": "East North Central",
            "RegionGroup": "Mid-West",
        },
        "Wisconsin": {
            "CensusRegion": "Midwest",
            "CensusDivision": "East North Central",
            "RegionGroup": "Mid-West",
        },
        "Iowa": {
            "CensusRegion": "Midwest",
            "CensusDivision": "West North Central",
            "RegionGroup": "Mid-West",
        },
        "Kansas": {
            "CensusRegion": "Midwest",
            "CensusDivision": "West North Central",
            "RegionGroup": "Mid-West",
        },
        "Minnesota": {
            "CensusRegion": "Midwest",
            "CensusDivision": "West North Central",
            "RegionGroup": "Mid-West",
        },
        "Missouri": {
            "CensusRegion": "Midwest",
            "CensusDivision": "West North Central",
            "RegionGroup": "Mid-West",
        },
        "North Dakota": {
            "CensusRegion": "Midwest",
            "CensusDivision": "West North Central",
            "RegionGroup": "Mid-West",
        },
        "Nebraska": {
            "CensusRegion": "Midwest",
            "CensusDivision": "West North Central",
            "RegionGroup": "Mid-West",
        },
        "South Dakota": {
            "CensusRegion": "Midwest",
            "CensusDivision": "West North Central",
            "RegionGroup": "Mid-West",
        },
        "District of Columbia": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "Delaware": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "Florida": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "Georgia": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "Maryland": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "North Carolina": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "Puerto Rico": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "South Carolina": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "Virginia": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "U.S. Virgin Islands": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "West Virginia": {
            "CensusRegion": "South",
            "CensusDivision": "South Atlantic",
            "RegionGroup": "East Coast",
        },
        "Alabama": {
            "CensusRegion": "South",
            "CensusDivision": "East South Central",
            "RegionGroup": "East Coast",
        },
        "Kentucky": {
            "CensusRegion": "South",
            "CensusDivision": "East South Central",
            "RegionGroup": "Mid-West",
        },
        "Mississippi": {
            "CensusRegion": "South",
            "CensusDivision": "East South Central",
            "RegionGroup": "Mid-West",
        },
        "Tennessee": {
            "CensusRegion": "South",
            "CensusDivision": "East South Central",
            "RegionGroup": "Mid-West",
        },
        "Arkansas": {
            "CensusRegion": "South",
            "CensusDivision": "West South Central",
            "RegionGroup": "Mid-West",
        },
        "Louisiana": {
            "CensusRegion": "South",
            "CensusDivision": "West South Central",
            "RegionGroup": "Mid-West",
        },
        "Oklahoma": {
            "CensusRegion": "South",
            "CensusDivision": "West South Central",
            "RegionGroup": "Mid-West",
        },
        "Texas": {
            "CensusRegion": "South",
            "CensusDivision": "West South Central",
            "RegionGroup": "Mid-West",
        },
        "Arizona": {
            "CensusRegion": "West",
            "CensusDivision": "Mountain",
            "RegionGroup": "West Coast",
        },
        "Colorado": {
            "CensusRegion": "West",
            "CensusDivision": "Mountain",
            "RegionGroup": "West Coast",
        },
        "Idaho": {
            "CensusRegion": "West",
            "CensusDivision": "Mountain",
            "RegionGroup": "West Coast",
        },
        "Montana": {
            "CensusRegion": "West",
            "CensusDivision": "Mountain",
            "RegionGroup": "West Coast",
        },
        "New Mexico": {
            "CensusRegion": "West",
            "CensusDivision": "Mountain",
            "RegionGroup": "West Coast",
        },
        "Nevada": {
            "CensusRegion": "West",
            "CensusDivision": "Mountain",
            "RegionGroup": "West Coast",
        },
        "Utah": {
            "CensusRegion": "West",
            "CensusDivision": "Mountain",
            "RegionGroup": "West Coast",
        },
        "Wyoming": {
            "CensusRegion": "West",
            "CensusDivision": "Mountain",
            "RegionGroup": "West Coast",
        },
        "Alaska": {
            "CensusRegion": "West",
            "CensusDivision": "Pacific",
            "RegionGroup": "West Coast",
        },
        "American Samoa": {
            "CensusRegion": "West",
            "CensusDivision": "Pacific",
            "RegionGroup": "West Coast",
        },
        "California": {
            "CensusRegion": "West",
            "CensusDivision": "Pacific",
            "RegionGroup": "West Coast",
        },
        "Oregon": {
            "CensusRegion": "West",
            "CensusDivision": "Pacific",
            "RegionGroup": "West Coast",
        },
        "Guam": {
            "CensusRegion": "West",
            "CensusDivision": "Pacific",
            "RegionGroup": "West Coast",
        },
        "Hawaii": {
            "CensusRegion": "West",
            "CensusDivision": "Pacific",
            "RegionGroup": "West Coast",
        },
        "Northern Mariana Islands": {
            "CensusRegion": "West",
            "CensusDivision": "Pacific",
            "RegionGroup": "West Coast",
        },
        "Washington": {
            "CensusRegion": "West",
            "CensusDivision": "Pacific",
            "RegionGroup": "West Coast",
        },
    }


def get_hazus_height_classes() -> dict[str, list[int]]:
    """
    Return the legacy story bands used to pick Appendix A weight rows.

    These bands (1-3, 4-6, 7 and up) do not match the manual, whose
    tables split at 1-3, 4-7, and 8 and up (Table 4-2 in 6.1). They are kept
    unchanged for the legacy HazusInferer; new code uses
    get_hazus_sampling_height_bands. Buildings over 199 stories fall
    outside every band.

    Returns:
        dict[str, list[int]]:
            Band name to the list of story counts it covers.
    """
    return {
        "low_rise": [1, 2, 3],
        "mid_rise": [4, 5, 6],
        "high_rise": list(range(7, 200)),
    }


def get_hazus_year_classes() -> dict[str, list[int]]:
    """
    Return the building era bands of the West Coast Appendix A tables.

    The eras (pre-1950, 1950-1970, post 1970) stratify the West Coast
    structure type tables. Years from 2030 on fall outside every band.

    Returns:
        dict[str, list[int]]:
            Era name to the list of construction years it covers.
    """
    return {
        "pre_1950": list(range(1, 1950)),
        "1950_to_1970": list(range(1950, 1970)),
        "post_1970": list(range(1970, 2030)),
    }


def get_hazus_sampling_height_bands() -> dict[str, tuple[int, int | None]]:
    """
    Return the story bands of the Appendix A structure type tables.

    These are the bands the manual uses to split the non-RES1 tables:
    low-rise 1-3, mid-rise 4-7, high-rise 8 and up (Table 4-2 of the 6.1
    edition, matching the Appendix A table captions). They pick the
    Appendix A weight row for a story count and are distinct from
    the Table 5-1 height classes assigned to a drawn structure type
    (see get_hazus_height_classes_by_type).

    Returns:
        dict[str, tuple[int, int | None]]:
            Band name to an inclusive (lowest, highest) story range,
            with None for no upper bound.
    """
    return {
        "low_rise": (1, 3),
        "mid_rise": (4, 7),
        "high_rise": (8, None),
    }


def get_hazus_sampling_height_band(stories: int) -> str:
    """
    Return the Appendix A story band a story count falls in.

    Args:
        stories (int):
            Story count.

    Returns:
        str:
            The band of :func:`get_hazus_sampling_height_bands` that
            covers the count.

    Raises:
        ValueError:
            If no band covers the count.
    """
    for band, (lowest, highest) in get_hazus_sampling_height_bands().items():
        if stories >= lowest and (highest is None or stories <= highest):
            return band
    raise ValueError(f'No story band covers {stories} stories.')


def get_hazus_story_count(stories: object) -> int:
    """
    Read a story count as the Appendix A tables take it.

    Args:
        stories (object):
            A feature value: a number, or a string ``float`` reads.

    Returns:
        int:
            The count.

    Raises:
        TypeError:
            If ``float`` cannot read the value.
        ValueError:
            If the value is not a finite whole number of one or more.
    """
    try:
        count = float(stories)
    except (TypeError, ValueError) as error:
        msg = f'{stories!r} is not a number.'
        raise TypeError(msg) from error
    if not math.isfinite(count) or count != int(count) or count < 1:
        msg = f'{stories!r} is not a whole number of stories of one or more.'
        raise ValueError(msg)
    return int(count)


def get_hazus_height_classes_by_type() -> (
    dict[str, dict[str, tuple[int, int | None]] | None]
):
    """
    Return the Hazus height class of each structure type by story count.

    Transcribed from Table 5-1 of the FEMA Hazus Earthquake Model
    Technical Manual 6.1. Height class is a function of structure type
    and story count, and its bands differ from the sampling bands of
    the Appendix A tables. URM splits at two stories, the mid-rise
    classes of RM1 and URM absorb every taller building, and W1, W2,
    S3, PC1, and MH carry no height class at all (None; the damage
    models for these types ignore height).

    Returns:
        dict[str, dict[str, tuple[int, int | None]] | None]:
            Structure type to either None or a dict mapping height
            class ('Low-Rise', 'Mid-Rise', 'High-Rise') to an inclusive
            (lowest, highest) story range, with None for no upper
            bound.
    """
    full_grid = {
        "Low-Rise": (1, 3),
        "Mid-Rise": (4, 7),
        "High-Rise": (8, None),
    }
    return {
        "W1": None,
        "W2": None,
        "S1": dict(full_grid),
        "S2": dict(full_grid),
        "S3": None,
        "S4": dict(full_grid),
        "S5": dict(full_grid),
        "C1": dict(full_grid),
        "C2": dict(full_grid),
        "C3": dict(full_grid),
        "PC1": None,
        "PC2": dict(full_grid),
        "RM1": {"Low-Rise": (1, 3), "Mid-Rise": (4, None)},
        "RM2": dict(full_grid),
        "URM": {"Low-Rise": (1, 2), "Mid-Rise": (3, None)},
        "MH": None,
    }


def get_hazus_unreinforced_design_levels() -> dict[str, tuple[str, ...]]:
    """
    Return the design levels each region allows for URM, C3 and S5.

    Transcribed from the footnote under the Appendix A structure type
    tables of the FEMA Hazus Inventory Technical Manual 7.0:
    "Unreinforced SBTs URM, C3 and S5 are not allowed in West Coast
    Region where Design Levels are higher than Pre-Code", with Low-Code
    in place of Pre-Code for the other two regions.

    Returns:
        dict[str, tuple[str, ...]]:
            Region group to the design levels that admit these types.
    """
    return {
        "West Coast": ("Pre-Code",),
        "Mid-West": ("Pre-Code", "Low-Code"),
        "East Coast": ("Pre-Code", "Low-Code"),
    }


def get_hazus_general_type_membership() -> dict[str, list[str]]:
    """
    Return the Hazus specific types under each general building type.

    The keys follow the NSI BuildingType letters (NSI's H arrives as MH
    after the conventions rename): W wood, S steel, C concrete
    including precast, M masonry both reinforced and unreinforced, MH
    manufactured housing. An observed general type restricts the
    structure type draw to its members.

    Returns:
        dict[str, list[str]]:
            General building type letter to its Hazus structure types.
    """
    return {
        "W": ["W1", "W2"],
        "S": ["S1", "S2", "S3", "S4", "S5"],
        "C": ["C1", "C2", "C3", "PC1", "PC2"],
        "M": ["RM1", "RM2", "URM"],
        "MH": ["MH"],
    }


#
# ========================== Below for replacement cost
#


def get_hazus_base_replacement_cost() -> tuple[dict, dict]:
    """
    Return base replacement costs per square foot and garage costs.

    Transcribed from the 6.1 edition. RES1 costs by construction class,
    story count, and basement finish come from Table 6-3; the single
    costs for the other occupancies from Table 6-2; the RES2 costs by
    census region from Table 6-6, in 2020 dollars where the other
    tables are 2022; and the garage costs per residence from Table 6-4,
    whose Carport and None zeros are implied by the manual rather than
    printed.

    Returns:
        tuple[dict, dict]:
            (replacement_cost_per_ft2, garage_cost_per_residence). The
            first maps occupancy class to cost, with nested dicts for
            RES1 and RES2. The second maps construction class to cost by
            garage type.
    """
    replacement_cost_per_ft2_res1 = {
        "Economy": {
            "One-story": {"none": 114.63, "finished": 148.48, "unfinished": 125.03},
            "Two-story": {"none": 121.00, "finished": 139.95, "unfinished": 127.75},
            "Three-story": {"none": 121.00, "finished": 139.95, "unfinished": 127.75},
            "Split level": {"none": 111.49, "finished": 130.44, "unfinished": 118.24},
        },
        "Average": {
            "One-story": {"none": 128.20, "finished": 168.75, "unfinished": 140.95},
            "Two-story": {"none": 132.88, "finished": 158.08, "unfinished": 140.98},
            "Three-story": {"none": 137.43, "finished": 156.88, "unfinished": 143.73},
            "Split level": {"none": 122.86, "finished": 148.06, "unfinished": 130.96},
        },
        "Custom": {
            "One-story": {"none": 172.73, "finished": 234.63, "unfinished": 196.98},
            "Two-story": {"none": 174.96, "finished": 210.21, "unfinished": 189.21},
            "Three-story": {"none": 179.34, "finished": 204.94, "unfinished": 189.84},
            "Split level": {"none": 162.83, "finished": 198.08, "unfinished": 177.08},
        },
        "Luxury": {
            "One-story": {"none": 209.44, "finished": 274.34, "unfinished": 234.29},
            "Two-story": {"none": 212.86, "finished": 250.56, "unfinished": 227.86},
            "Three-story": {"none": 218.28, "finished": 245.93, "unfinished": 229.53},
            "Split level": {"none": 197.95, "finished": 235.65, "unfinished": 212.95},
        },
    }

    replacement_cost_per_ft2 = {
        "RES1": replacement_cost_per_ft2_res1,
        "RES2": {"Northeast": 68.65, "Midwest": 59.14, "South": 55.97, "West": 73.92},
        "RES3A": 134.57,
        "RES3B": 118.30,
        "RES3C": 250.19,
        "RES3D": 232.69,
        "RES3E": 217.05,
        "RES3F": 198.83,
        "RES4": 204.58,
        "RES5": 225.12,
        "RES6": 261.45,
        "COM1": 133.61,
        "COM2": 135.47,
        "COM3": 162.58,
        "COM4": 204.43,
        "COM5": 303.88,
        "COM6": 392.76,
        "COM7": 279.38,
        "COM8": 259.48,
        "COM9": 213.33,
        "COM10": 93.88,
        "IND1": 162.65,
        "IND2": 135.47,
        "IND3": 230.43,
        "IND4": 230.43,
        "IND5": 230.43,
        "IND6": 135.47,
        "AGR1": 135.47,
        "REL1": 222.10,
        "GOV1": 174.35,
        "GOV2": 284.46,
        "EDU1": 237.73,
        "EDU2": 197.10,
    }

    # TODO(adam): what does per residence mean here?
    # Table 6-4
    garage_cost_per_residence = {
        "Economy": {
            "One-car": 19394,
            "Two-car": 30397,
            "Three-car": 41125,
            "Carport": 0,
            "None": 0,
        },
        "Average": {
            "One-car": 20313,
            "Two-car": 31600,
            "Three-car": 42613,
            "Carport": 0,
            "None": 0,
        },
        "Custom": {
            "One-car": 22377,
            "Two-car": 35519,
            "Three-car": 48307,
            "Carport": 0,
            "None": 0,
        },
        "Luxury": {
            "One-car": 28233,
            "Two-car": 44515,
            "Three-car": 60444,
            "Carport": 0,
            "None": 0,
        },
    }

    return replacement_cost_per_ft2, garage_cost_per_residence


def get_hazus_region_to_garage() -> tuple[list[str], dict[str, list[int]]]:
    """
    Return garage type shares by census region.

    Transcribed from Table 5-5 of the 6.1 edition. The weights are percentages
    aligned with the garage type list.

    Returns:
        tuple[list[str], dict[str, list[int]]]:
            (garage_type_list, census_to_garage_weight), the latter
            keyed by census region.
    """
    garage_type_list = ["One-car", "Two-car", "Three-car", "Carport", "None"]
    census_to_garage_weight = {
        "Northeast": [29, 31, 5, 2, 33],
        "Midwest": [21, 51, 9, 2, 17],
        "South": [14, 39, 4, 8, 35],
        "West": [16, 51, 11, 6, 16],
    }

    return garage_type_list, census_to_garage_weight


def get_hazus_income_to_const_class() -> tuple[
    list[float], list[str], list[str], dict[str, list[int]], dict[str, int]
]:
    """
    Return the income group definitions and construction class weights.

    The income ratio thresholds and the weights mapping income groups to
    RES1 construction classes transcribe Table 6-5 of the 6.1 edition;
    the final threshold is a sentinel for the manual's open-ended top
    band. The state median household incomes transcribe Table 5-39.

    Returns:
        tuple[list[float], list[str], list[str], dict, dict]:
            (income_ratio_thres, income_group_list, const_class_list,
            income_to_const_class_weight, state_average_income). The
            weights are percentages aligned with const_class_list.
    """
    # Table 6-5
    income_ratio_thres = [0, 0.5, 0.85, 1.25, 2.0, 1.0e15]
    income_group_list = ["Group1", "Group2", "Group3", "Group4", "Group5"]
    const_class_list = ["Luxury", "Custom", "Average", "Economy"]
    income_to_const_class_weight = {
        "Group1": [0, 0, 0, 100],
        "Group2": [0, 0, 25, 75],
        "Group3": [0, 25, 75, 0],
        "Group4": [0, 100, 0, 0],
        "Group5": [100, 0, 0, 0],
    }

    # Table 5-39
    state_average_income = {
        "Alabama": 52035,
        "Alaska": 77790,
        "Arizona": 61529,
        "Arkansas": 49475,
        "California": 78672,
        "Colorado": 75231,
        "Connecticut": 79855,
        "Delaware": 69110,
        "District of Columbia": 90842,
        "Florida": 57703,
        "Georgia": 61224,
        "Hawaii": 83173,
        "Idaho": 58915,
        "Illinois": 68428,
        "Indiana": 58235,
        "Iowa": 61836,
        "Kansas": 61091,
        "Kentucky": 52238,
        "Louisiana": 50800,
        "Maine": 59489,
        "Maryland": 87063,
        "Massachusetts": 84385,
        "Michigan": 59234,
        "Minnesota": 73382,
        "Mississippi": 46511,
        "Missouri": 57290,
        "Montana": 56539,
        "Nebraska": 63015,
        "Nevada": 62043,
        "New Hampshire": 77923,
        "New Jersey": 85245,
        "New Mexico": 51243,
        "New York": 71117,
        "North Carolina": 56642,
        "North Dakota": 65315,
        "Ohio": 58116,
        "Oklahoma": 53840,
        "Oregon": 65667,
        "Pennsylvania": 63627,
        "Rhode Island": 70305,
        "South Carolina": 54864,
        "South Dakota": 59896,
        "Tennessee": 54833,
        "Texas": 63826,
        "Utah": 74197,
        "Vermont": 63477,
        "Virginia": 76398,
        "Washington": 77006,
        "West Virginia": 48037,
        "Wisconsin": 63293,
        "Wyoming": 65304,
        "Puerto Rico": 21058,
    }

    return (
        income_ratio_thres,
        income_group_list,
        const_class_list,
        income_to_const_class_weight,
        state_average_income,
    )


def get_hazus_height_classes_RES1() -> dict[str, list[int]]:  # noqa: N802
    """
    Return story bands for the RES1 replacement cost table rows.

    The bands map story counts onto the RES1 replacement cost rows of
    the manual. Three or more stories fall into the Three-story row, and
    the Split level row cannot be reached from a story count. The name
    keeps its legacy capitalization for the legacy HazusInferer.

    Returns:
        dict[str, list[int]]:
            RES1 cost row name to the list of story counts it covers.
    """
    return {
        "One-story": [1],
        "Two-story": [2],
        "Three-story": list(range(3, 1000)),  # TODO(adam): double check
    }
