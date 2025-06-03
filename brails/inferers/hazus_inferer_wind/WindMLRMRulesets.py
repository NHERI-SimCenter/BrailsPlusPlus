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

import random
import datetime
from brails.inferers.hazus_inferer_wind.WindMetaVarRulesets import is_ready_to_infer

def MLRM_config(BIM):
    """
    Rules to identify a HAZUS MLRM configuration based on BIM data

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.

    Returns
    -------
    config: str
        A string that identifies a specific configration within this buidling
        class.
    """
    available_features = BIM.keys()


    # Note the only roof option for commercial masonry in NJ appraisers manual
    # is OSWJ, so this suggests they do not even see alternate roof system
    # ref: Custom Inventory google spreadsheet H-37 10/01/20
    # This could be commented for other regions if detailed data are available
    
    if "RoofSystem" not in BIM:
        BIM['RoofSystem'] = 'Open-Web Steel Joists'



    if "RoofCover" in BIM:
        roof_cover = BIM["RoofCover"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofCover"):
        # Roof cover
        # Roof cover does not apply to gable and hip roofs
        if BIM['YearBuilt'] >= 1975:
            roof_cover = 'Single-Ply Membrane'
        else:
            # year < 1975
            roof_cover = 'Built-Up Roof'

    if "Shutters" in BIM:
        roof_cover = BIM["Shutters"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["WindBorneDebris"], inferred_feature= "Shutters"):
        # Shutters
        # IRC 2000-2015:
        # R301.2.1.2 in NJ IRC 2015 says protection of openings required for
        # buildings located in WindBorneDebris regions, mentions impact-rated protection for
        # glazing, impact-resistance for garage door glazed openings, and finally
        # states that wood structural panels with a thickness > 7/16" and a
        # span <8' can be used, as long as they are precut, attached to the framing
        # surrounding the opening, and the attachments are resistant to corrosion
        # and are able to resist component and cladding loads;
        # Earlier IRC editions provide similar rules.
        shutters = BIM['WindBorneDebris']



    if "WindDebrisClass" in BIM:
        WIDD = BIM["WindDebrisClass"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["OccupancyClass"], inferred_feature= "WindDebrisClass"):

        # Wind Debris (widd in HAZSU)
        # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
        WIDD = 'C' # residential (default)
        if BIM['OccupancyClass'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                     'RES3D']:
            WIDD = 'C' # residential
        elif BIM['OccupancyClass'] == 'AGR1':
            WIDD = 'D' # None
        else:
            WIDD = 'A' # Res/Comm


    if "RoofDeckAttachment" in BIM:
        RDA = BIM["RoofDeckAttachment"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofSystem"], inferred_feature= "RoofDeckAttachment"):
        if BIM['RoofSystem'] == 'Open-Web Steel Joists':
            # RDA
            is_ready_to_infer(available_features=available_features, needed_features = ["DesignWindSpeed"], inferred_feature= "RoofDeckAttachment")
            # Metal RDA
            # 1507.2.8.1 High Wind Attachment.
            # Underlayment applied in areas subject to high winds (Vasd greater
            # than 110 mph as determined in accordance with Section 1609.3.1) shall
            #  be applied with corrosion-resistant fasteners in accordance with
            # the manufacturer’s instructions. Fasteners are to be applied along
            # the overlap not more than 36 inches on center.
            if BIM['DesignWindSpeed'] > 142:
                RDA = 'Standard'  # standard
            else:
                RDA = 'Superior'  # superior


        elif BIM['RoofSystem'] == 'Truss':
            is_ready_to_infer(available_features=available_features, needed_features = ["LandCover","DesignWindSpeed"], inferred_feature= "RoofDeckAttachment")
            # This clause should not be activated for NJ
            # RDA
            #if BIM['LandCover'] >= 35: # suburban or light trees
            if BIM['LandCover'] in ['Suburban','Light Trees','Trees']: # suburban or light trees
                if BIM['DesignWindSpeed'] > 130.0:
                    RDA = '8s'  # 8d @ 6"/6" 'D'
                else:
                    RDA = '8d'  # 8d @ 6"/12" 'B'
            else:  # light suburban or open
                if BIM['DesignWindSpeed'] > 110.0:
                    RDA = '8s'  # 8d @ 6"/6" 'D'
                else:
                    RDA = '8d'  # 8d @ 6"/12" 'B'

    # if "RoofDeckAge" in BIM:
    #     DQ = BIM["RoofDeckAge"]

    # elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofSystem"], inferred_feature= "RoofDeckAttachment"):
    #     if BIM['RoofSystem'] == 'Open-Web Steel Joists':

    #         # Roof deck age (DQ)
    #         # Average lifespan of a steel joist roof is roughly 50 years according
    #         # to the source below. Therefore, if constructed 50 years before the
    #         # current year, the roof deck should be considered old.
    #         # https://www.metalroofing.systems/metal-roofing-pros-cons/
    #         is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofDeckAttachment")
    #         if BIM['YearBuilt'] >= (datetime.datetime.now().year - 50):
    #             DQ = 'Good' # new or average
    #         else:
    #             DQ = 'Poor' # old

    #     elif BIM['RoofSystem'] == 'Truss':
    #         # Roof deck agea (DQ)
    #         DQ = '' # null # Doesn't apply to Wood Truss


    if "RoofToWallConnection" in BIM:
        RWC = BIM["RoofToWallConnection"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofSystem"], inferred_feature= "RoofToWallConnection"):
 
        if BIM['RoofSystem'] == 'Open-Web Steel Joists':
            # RWC
            RWC = '' # null  # Doesn't apply to OWSJ

        elif BIM['RoofSystem'] == 'Truss':
            # RWC
            is_ready_to_infer(available_features=available_features, needed_features = ["DesignWindSpeed"], inferred_feature= "RoofToWallConnection")

            if BIM['DesignWindSpeed'] > 110:
                RWC = 'Strap'  # Strap
            else:
                RWC = 'Toe-nail'  # Toe-nail



    # shutters
    if "Shutters" in BIM:
        shutters = BIM["Shutters"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["WindBorneDebris","YearBuilt"], inferred_feature= "Shutters"):
        if BIM['YearBuilt'] >= 2000:
            shutters = BIM['WindBorneDebris']
        else:
            if BIM['WindBorneDebris']:
                shutters = random.random() < 0.46
            else:
                shutters = False


    is_ready_to_infer(available_features=available_features, needed_features = ["Height"], inferred_feature= "M.LRM class")

    if BIM['Height'] < 15.0:
        # extend the BIM dictionary

        is_ready_to_infer(available_features=available_features, needed_features = ['BuildingType', 'StructureType', 'LandCover','MasonryReinforcing','RoofSystem','Height'], inferred_feature= "M.LRM1 class")
        essential_features = dict(
            BuildingType=BIM['BuildingType'],
            StructureType=BIM['StructureType'],
            LandCover=BIM['LandCover'],
            RoofCover = roof_cover,
            RoofDeckAttachment = RDA,
            RoofToWallConnection = RWC,
            Shutters = int(shutters),
            MasonryReinforcing = int(BIM['MasonryReinforcing']),
            WindDebrisClass = WIDD,
            RoofSystem = BIM['RoofSystem'],
            Height = BIM['Height']
            )

        BIM.update(dict(essential_features))

        # if it's MLRM1, configure outputs
        # bldg_config = f"M.LRM.1." \
        #               f"{roof_cover}." \
        #               f"{int(shutters)}." \
        #               f"{int(BIM['MasonryReinforcing'])}." \
        #               f"{WIDD}." \
        #               f"{BIM['RoofSystem']}." \
        #               f"{RDA}." \
        #               f"{RWC}." \
        #               f"{DQ}." \
        #               f"{MRDA}." \
        #               f"{BIM['LandCover']}"

    else:
        
        # if "UnitType" in BIM:
        #     unit_tag = BIM["UnitType"]

        # elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofSystem"], inferred_feature= "UnitType"):
        #     if BIM['RoofSystem'] == 'Truss':
        #         unit_tag = '' # null
        #     elif BIM['RoofSystem'] == 'Open-Web Steel Joists':
        #         is_ready_to_infer(available_features=available_features, needed_features = ["NumberOfUnits"], inferred_feature= "UnitType")
        #         if BIM['NumberOfUnits'] == 1:
        #             unit_tag = 'sgl'
        #         else:
        #             unit_tag = 'mlt'

        if "JoistSpacing" in BIM:
            joist_spacing = BIM["JoistSpacing"]
            
        elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofSystem"], inferred_feature= "JoistSpacing"):
            # MLRM2 needs more rulesets
            if BIM['RoofSystem'] == 'Truss':
                joist_spacing = '' # null
            elif BIM['RoofSystem'] == 'Open-Web Steel Joists':
                is_ready_to_infer(available_features=available_features, needed_features = ["NumberOfUnits"], inferred_feature= "JoistSpacing")
                if BIM['NumberOfUnits'] == 1:
                    joist_spacing = '' # null
                else:
                    joist_spacing = 4

        is_ready_to_infer(available_features=available_features, needed_features = ['BuildingType', 'StructureType', 'LandCover','MasonryReinforcing','RoofSystem', 'Height', 'NumberOfUnits'], inferred_feature= "M.LRI2 class")

        essential_features = dict(
            BuildingType=BIM['BuildingType'],
            StructureType=BIM['StructureType'],
            LandCover=BIM['LandCover'],
            RoofCover = roof_cover,
            RoofDeckAttachment = RDA,
            RoofToWallConnection = RWC,
            Shutters = int(shutters),
            MasonryReinforcing = int(BIM['MasonryReinforcing']),
            RoofSystem = BIM['RoofSystem'],
            WindDebrisClass = WIDD,
            JoistSpacing=joist_spacing,
            Height = BIM['Height'],
            NumberOfUnits = int(BIM['NumberOfUnits'])
            )

        # extend the BIM dictionary
        BIM.update(dict(essential_features))

        # bldg_config = f"M.LRM.2." \
        #               f"{roof_cover}." \
        #               f"{int(shutters)}." \
        #               f"{int(BIM['MasonryReinforcing'])}." \
        #               f"{WIDD}." \
        #               f"{BIM['RoofSystem']}." \
        #               f"{RDA}." \
        #               f"{RWC}." \
        #               f"{DQ}." \
        #               f"{MRDA}." \
        #               f"{unit_tag}." \
        #               f"{joist_spacing}." \
        #               f"{BIM['LandCover']}"
        
    return essential_features