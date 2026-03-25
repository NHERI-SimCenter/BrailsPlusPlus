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
# Adam Zsarnoczay
#
# Last updated:
# 03-24-2026 

from brails.validators.inventory_validator import InventoryValidator
from brails.types.asset_inventory import AssetInventory
from copy import deepcopy
import random
import pandas as pd

"""
This module attempts to fix a building inventory for a Hazus evaluation.
"""

class HazusEarthquakeValidator(InventoryValidator):

    def fix_inventory( self, inventory: AssetInventory) -> tuple[AssetInventory, pd.DataFrame]:

        """
        Fix normalize an AssetInventory to comply with Hazus Earthquake
        occupancy and design assumptions.

        Returns:
            (fixed_inventory, issues_df):
                - fixed_inventory: corrected deep copy of input inventory
                - issues_df: DataFrame of assets that were modified
        """

        # Work on a copy to avoid mutating the original input
        bldg_inventory = deepcopy(inventory)

        # Track assets that required correction
        issues_dict = {}

        for asset_id, asset in bldg_inventory.inventory.items():
            
            # Extract key features required for Hazus logic
            stories = asset.features.get('NumberOfStories', None)
            occupancy = asset.features.get('OccupancyClass', None)
            year = asset.features.get('YearBuilt', None)

            # Skip assets with missing critical attributes
            # (cannot safely apply Hazus rules)
            if stories is None or occupancy is None or year is None:
                continue
            
            # Normalize occupancy class to Hazus-compatible format
            # Keep original NSI value while truncating for Hazus mapping
            if occupancy[:4] == 'RES3':
                asset.add_features({
                    'OccupancyClassNSI': occupancy,
                    'OccupancyClass': occupancy[:5]  # RES3X subtype
                })
            else:
                asset.add_features({
                    'OccupancyClassNSI': occupancy,
                    'OccupancyClass': occupancy[:4]
                })

            # Ensure numeric types for rule evaluation
            stories = int(stories)
            year = int(year)

            # Flag indicates whether this asset was corrected
            flagged = False

            # -------------------------
            # HIGH-RISE RULES (>= 7 stories)
            # -------------------------
            if stories >= 7 and occupancy not in [
                    'RES3A', 'RES3B', 'RES3C', 'RES3D', 'RES3E', 'RES3F',
                    'RES4', 'RES5',
                    'COM4', 'COM5', 'COM6',
                    'GOV1',
                    'EDU2'
            ]:

                flagged = True
                
                # Reassign to valid high-rise categories based on prefix
                if occupancy.startswith('RES'):
                    # Default to high-rise residential
                    asset.add_features(
                        {
                            'OccupancyClass': 'RES3F',
                            'OccupancyClassNSI': 'RES3F'
                        },
                        overwrite=True
                    )

                elif occupancy.startswith('COM'):
                    # Randomly assign among valid commercial high-rise classes
                    occupancy_to_use = random.choice(['COM4', 'COM5', 'COM6'])
                    asset.add_features(
                        {
                            'OccupancyClass': occupancy_to_use,
                            'OccupancyClassNSI': occupancy_to_use
                        },
                        overwrite=True
                    )

                elif occupancy.startswith('GOV'):
                    asset.add_features(
                        {
                            'OccupancyClass': 'GOV1',
                            'OccupancyClassNSI': 'GOV1'
                        },
                        overwrite=True
                    )

                elif occupancy.startswith('EDU'):
                    asset.add_features(
                        {
                            'OccupancyClass': 'EDU2',
                            'OccupancyClassNSI': 'EDU2'
                        },
                        overwrite=True
                    )

                elif occupancy.startswith('IND'):
                    # Industrial → mapped to commercial high-rise classes
                    occupancy_to_use = random.choice(['COM4', 'COM5', 'COM6'])
                    asset.add_features(
                        {
                            'OccupancyClass': occupancy_to_use,
                            'OccupancyClassNSI': occupancy_to_use
                        },
                        overwrite=True
                    )

                elif occupancy.startswith('REL'):
                    # Religious → treated as government
                    asset.add_features(
                        {
                            'OccupancyClass': 'GOV1',
                            'OccupancyClassNSI': 'GOV1'
                        },
                        overwrite=True
                    )

                else:
                    # Catch unexpected categories for debugging
                    print(f'UNEXPECTED OCCUPANCY CLASS for high rise: {occupancy}')

            # -------------------------
            # MID-RISE RULES (4–6 stories)
            # -------------------------
            elif stories >= 4:

                # Hazus-based heuristics depending on construction year
                if year < 1950 and occupancy in ['RES1', 'RES2', 'IND1', 'EDU1', 'GOV2']:
                    flagged = True

                elif year < 1970 and occupancy in ['RES1', 'RES2', 'IND6', 'EDU1']:
                    flagged = True

                elif occupancy in ['RES1', 'RES2', 'IND1', 'IND2', 'IND6', 'EDU1']:
                    flagged = True

                if flagged:

                    # Reassign to appropriate mid-rise categories
                    if occupancy.startswith('RES'):
                        asset.add_features(
                            {
                                'OccupancyClass': 'RES3C',
                                'OccupancyClassNSI': 'RES3C'
                            },
                            overwrite=True
                        )

                    elif occupancy.startswith('GOV'):
                        asset.add_features(
                            {
                                'OccupancyClass': 'GOV1',
                                'OccupancyClassNSI': 'GOV1'
                            },
                            overwrite=True
                        )
                        
                    elif occupancy.startswith('EDU'):
                        asset.add_features(
                            {
                                'OccupancyClass': 'EDU2',
                                'OccupancyClassNSI': 'EDU2'
                            },
                            overwrite=True
                        )
                        
                    elif occupancy.startswith('IND'):
                        # Randomly assign among valid industrial mid-rise classes
                        occupancy_to_use = random.choice(['IND3', 'IND4', 'IND5'])
                        asset.add_features(
                            {
                                'OccupancyClass': occupancy_to_use,
                                'OccupancyClassNSI': occupancy_to_use
                            },
                            overwrite=True
                        )
                        
                    else:
                        print(f'UNEXPECTED OCCUPANCY CLASS for mid rise: {occupancy}')

            # certain structural systems cannot exceed Low-Code
            if (asset.features.get('StructureType', None) in ['S5','C3','URM'] and
                asset.features.get('DesignLevel', None) in ['Moderate-Code', 'High-Code']
                ):
                asset.add_features({
                    'DesignLevel': 'Low-Code'
                })
                flagged = True;
                        
            # Record modifications for reporting/debugging
            if flagged:
                issues_dict.update({asset_id: {
                    'NumberOfStories': stories,
                    'OccupancyClass': occupancy,
                    'YearBuilt': year,
                    'DesignLevel':asset.features.get('DesignLevel', 'None')}}
                )

            # ---------------------------
            # POST-PROCESSING ADJUSTMENTS
            # ---------------------------

            # Ensure RES3 classes are truncated properly (Hazus only has RES3)
            # - but bldg cost uses RES3A, .... so cannot just change to RES3
            # occupancy = asset.features.get('OccupancyClass', None)            
            # if occupancy[:4] == 'RES3':
            #    asset.add_features({
            #        'OccupancyClass': occupancy[:4]
            #    })

                
        # Return fixed inventory and summary of modified assets
        return bldg_inventory, pd.DataFrame(issues_dict)

