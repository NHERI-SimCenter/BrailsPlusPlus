# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# Kuanshi Zhong
#
# Based on rulesets developed by:
# Karen Angeles
# Meredith Lockhead
# Tracy Kijewski-Correa

from brails.inferers.hazus_inferer_wind.WindMetaVarRulesets import is_ready_to_infer


def building_class(BIM, hazard):
    """
    Short description

    Long description

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.

    Returns
    -------
    bldg_class: str
        One of the standard building class labels from HAZUS
    """

    # check hazard
    if hazard not in ['wind', 'inundation']:
        print(f'WARNING: The provided hazard is not recognized: {hazard}')

    # just for brevity
    def quick_check_keys(needed_features):
        return is_ready_to_infer(available_features=BIM.keys(), needed_features = needed_features,inferred_feature= "BuildingClass")


    quick_check_keys(['BuildingType'])
    building_type = BIM["BuildingType"]
    if building_type in ['Manufactured Housing','Wood','Steel','Concrete','Masonry']:
        pass
    else:
        if building_type.startswith('H') or building_type.startswith('MH'):
            building_type = 'Manufactured Housing'
        elif building_type.startswith('W'):
            building_type = 'Wood'
        elif building_type.startswith('S'):
            building_type = 'Steel'
        elif building_type.startswith('C') or building_type.startswith('PC') :
            building_type = 'Concrete'
        elif building_type.startswith('M') or building_type.startswith('RM') or building_type.startswith('URM'):
            building_type = 'Masonry'
        else:
            msg = f"ERROR: building_type {building_type} not identified"
            raise Exception(msg)


    if hazard == 'wind':

        if building_type == 'Wood':
            #Wood
            quick_check_keys(['OccupancyClass','RoofShape'])
            if ((BIM['OccupancyClass'] == 'RES1') or
                ((BIM['RoofShape'] != 'Flat') and (BIM['OccupancyClass'] == 'RES1'))):
                # BuildingType = 3001
                # OccupancyClass = RES1
                # Wood Single-Family Homes (WSF1 or WSF2)
                # OR roof type = flat (HAZUS can only map flat to WSF1)
                # OR default (by '')
                if BIM['RoofShape'] == 'Flat': # checking if there is a misclassication
                    BIM['RoofShape'] = 'Gable' # ensure the WSF has gab (by default, note gab is more vulneable than hip)
                bldg_class = 'WSF'
                structure_type = 'Single Family Housing'
            else:
                # BuildingType = 3001
                # OccupancyClass = RES3, RES5, RES6, or COM8
                # Wood Multi-Unit Hotel (WMUH1, WMUH2, or WMUH3)
                bldg_class = 'WMUH'
                structure_type = 'Multi-Unit Housing'

        elif building_type == 'Steel':
            #Steel
            quick_check_keys(['OccupancyClass','DesignLevel'])
            if ((BIM['DesignLevel'] == 'E') and
                (BIM['OccupancyClass'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                                'RES3E', 'RES3F'])):
                # BuildingType = 3002
                # Steel Engineered Residential Building (SERBL, SERBM, SERBH)
                bldg_class = 'SERB'
                structure_type = 'Engineered Residential Building'
            elif ((BIM['DesignLevel'] == 'E') and
                (BIM['OccupancyClass'] in ['COM1', 'COM2', 'COM3', 'COM4', 'COM5',
                                            'COM6', 'COM7', 'COM8', 'COM9','COM10'])):
                # BuildingType = 3002
                # Steel Engineered Commercial Building (SECBL, SECBM, SECBH)
                bldg_class = 'SECB'
                structure_type = 'Engineered Commercial Building'
            elif ((BIM['DesignLevel'] == 'PE') and
                (BIM['OccupancyClass'] not in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                            'RES3E', 'RES3F'])):
                # BuildingType = 3002
                # Steel Pre-Engineered Metal Building (SPMBS, SPMBM, SPMBL)
                bldg_class = 'SPMB'
                structure_type = 'Pre-Engineered Metal Building'

            else:
                bldg_class = 'SECB'
                structure_type = 'Engineered Commercial Building'

        elif building_type == 'Concrete':
            #Concrete
            quick_check_keys(['OccupancyClass','DesignLevel'])
            if ((BIM['DesignLevel'] == 'E') and
                (BIM['OccupancyClass'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                            'RES3E', 'RES3F', 'RES5', 'RES6'])):
                # BuildingType = 3003
                # Concrete Engineered Residential Building (CERBL, CERBM, CERBH)
                bldg_class = 'CERB'
                structure_type = 'Engineered Residential Building'

            elif ((BIM['DesignLevel'] == 'E') and
                (BIM['OccupancyClass'] in ['COM1', 'COM2', 'COM3', 'COM4', 'COM5',
                                            'COM6', 'COM7', 'COM8', 'COM9','COM10'])):
                # BuildingType = 3003
                # Concrete Engineered Commercial Building (CECBL, CECBM, CECBH)
                bldg_class = 'CECB'
                structure_type = 'Engineered Commercial Building'

            else:
                bldg_class = 'CECB'
                structure_type = 'Engineered Commercial Building'


        elif building_type == 'Manufactured Housing':
            bldg_class = 'MH'

            quick_check_keys(['YearBuilt'])
            year = BIM['YearBuilt'] # just for the sake of brevity
            if year <= 1976:
                structure_type = 'Pre-HUD'
            elif year <= 1994:
                structure_type = '1976 HUD'
            else:
                quick_check_keys(['WindZone'])
                if BIM['WindZone']=='I':
                    structure_type = '1994 HUD Zone 1'
                elif BIM['WindZone']=='II':
                    structure_type = '1994 HUD Zone 2'
                elif BIM['WindZone']=='III':
                    structure_type = '1994 HUD Zone 3'

        elif building_type == 'Masonry':
            #Masonry
            quick_check_keys(['OccupancyClass','NumberOfStories','DesignLevel'])
            if BIM['OccupancyClass'] == 'RES1':
                # BuildingType = 3004
                # OccupancyClass = RES1
                # Masonry Single-Family Homes (MSF1 or MSF2)
                bldg_class = 'MSF'
                structure_type = 'Single Family Housing'

            elif ((BIM['OccupancyClass'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                            'RES3E', 'RES3F']) and (BIM['DesignLevel'] == 'E')):
                # BuildingType = 3004
                # Masonry Engineered Residential Building (MERBL, MERBM, MERBH)
                bldg_class = 'MERB'
                structure_type = 'Engineered Residential Building'

            elif ((BIM['OccupancyClass'] in ['COM1', 'COM2', 'COM3', 'COM4',
                                            'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                                            'COM10']) and (BIM['DesignLevel'] == 'E')):
                # BuildingType = 3004
                # Masonry Engineered Commercial Building (MECBL, MECBM, MECBH)
                bldg_class = 'MECB'
                structure_type = 'Engineered Commercial Building'

            elif BIM['OccupancyClass'] in ['IND1', 'IND2', 'IND3', 'IND4', 'IND5', 'IND6']:
                # BuildingType = 3004
                # Masonry Low-Rise Masonry Warehouse/Factory (MLRI)
                bldg_class = 'MLRI'
                structure_type = 'Low-Rise Industrial Building'

            elif BIM['OccupancyClass'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                            'RES3E', 'RES3F', 'RES5', 'RES6', 'COM8']:
                # BuildingType = 3004
                # OccupancyClass = RES3X or COM8
                # Masonry Multi-Unit Hotel/Motel (MMUH1, MMUH2, or MMUH3)
                bldg_class = 'MMUH'
                structure_type = 'Multi-Unit Housing'


            elif ((BIM['NumberOfStories'] == 1) and
                    (BIM['OccupancyClass'] in ['COM1', 'COM2'])):
                # BuildingType = 3004
                # Low-Rise Masonry Strip Mall (MLRM1 or MLRM2)
                bldg_class = 'MLRM'
                structure_type = 'Low-Rise Strip Mall'

            else:
                bldg_class = 'MECB' # for others not covered by the above
                structure_type = 'Engineered Commercial Building'

            #elif ((BIM['OccupancyClass'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
            #                                'RES3E', 'RES3F', 'RES5', 'RES6',
            #                                'COM8']) and (BIM['DesignLevel'] in ['NE', 'ME'])):
            #    # BuildingType = 3004
            #    # Masonry Multi-Unit Hotel/Motel Non-Engineered
            #    # (MMUH1NE, MMUH2NE, or MMUH3NE)
            #    return 'MMUHNE'

        else:
            bldg_class = 'WMUH'
            structure_type = 'Multi-Unit Housing'

            # if nan building type is provided, return the dominant class

        BIM['StructureType'] = structure_type
        BIM['BuildingType'] = building_type

    return bldg_class, BIM

