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
    
    if "RoofFrameType" not in BIM:
        BIM['RoofFrameType'] = 'ows'



    if "RoofCover" in BIM:
        roof_cover = BIM["RoofCover"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofCover"):
        # Roof cover
        # Roof cover does not apply to gable and hip roofs
        if BIM['YearBuilt'] >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

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


    if "RoofDeckAttachmentW" in BIM:
        RDA = BIM["RoofDeckAttachmentW"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofFrameType"], inferred_feature= "RoofDeckAttachmentW"):
        if BIM['RoofFrameType'] == 'ows':
            # RDA
            RDA = 'null' # Doesn't apply to OWSJ

        elif BIM['RoofFrameType'] == 'trs':
            is_ready_to_infer(available_features=available_features, needed_features = ["TerrainRoughness","DesignWindSpeed"], inferred_feature= "RoofDeckAttachmentW")
            # This clause should not be activated for NJ
            # RDA
            if BIM['TerrainRoughness'] >= 35: # suburban or light trees
                if BIM['DesignWindSpeed'] > 130.0:
                    RDA = '8s'  # 8d @ 6"/6" 'D'
                else:
                    RDA = '8d'  # 8d @ 6"/12" 'B'
            else:  # light suburban or open
                if BIM['DesignWindSpeed'] > 110.0:
                    RDA = '8s'  # 8d @ 6"/6" 'D'
                else:
                    RDA = '8d'  # 8d @ 6"/12" 'B'


    if "RoofDeckAttachmentW" in BIM:
        RDA = BIM["RoofDeckAttachmentW"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofFrameType"], inferred_feature= "RoofDeckAttachmentW"):
        if BIM['RoofFrameType'] == 'ows':

            # Roof deck age (DQ)
            # Average lifespan of a steel joist roof is roughly 50 years according
            # to the source below. Therefore, if constructed 50 years before the
            # current year, the roof deck should be considered old.
            # https://www.metalroofing.systems/metal-roofing-pros-cons/
            is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofDeckAttachmentW")
            if BIM['YearBuilt'] >= (datetime.datetime.now().year - 50):
                DQ = 'god' # new or average
            else:
                DQ = 'por' # old

        elif BIM['RoofFrameType'] == 'trs':
            # Roof deck agea (DQ)
            DQ = 'null' # Doesn't apply to Wood Truss


    if "RoofToWallConnection" in BIM:
        RDA = BIM["RoofToWallConnection"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofFrameType"], inferred_feature= "RoofToWallConnection"):
 
        if BIM['RoofFrameType'] == 'ows':
            # RWC
            RWC = 'null'  # Doesn't apply to OWSJ

        elif BIM['RoofFrameType'] == 'trs':
            # RWC
            is_ready_to_infer(available_features=available_features, needed_features = ["DesignWindSpeed"], inferred_feature= "RoofToWallConnection")

            if BIM['DesignWindSpeed'] > 110:
                RWC = 'strap'  # Strap
            else:
                RWC = 'tnail'  # Toe-nail


    if "RoofDeckAttachmentM" in BIM:
        RDA = BIM["RoofDeckAttachmentM"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofFrameType"], inferred_feature= "RoofDeckAttachmentM"):
        if BIM['RoofFrameType'] == 'ows':
            is_ready_to_infer(available_features=available_features, needed_features = ["DesignWindSpeed"], inferred_feature= "RoofDeckAttachmentM")
            # Metal RDA
            # 1507.2.8.1 High Wind Attachment.
            # Underlayment applied in areas subject to high winds (Vasd greater
            # than 110 mph as determined in accordance with Section 1609.3.1) shall
            #  be applied with corrosion-resistant fasteners in accordance with
            # the manufacturer’s instructions. Fasteners are to be applied along
            # the overlap not more than 36 inches on center.
            if BIM['DesignWindSpeed'] > 142:
                MRDA = 'std'  # standard
            else:
                MRDA = 'sup'  # superior

        elif BIM['RoofFrameType'] == 'trs':
            #  Metal RDA
            MRDA = 'null' # Doesn't apply to Wood Truss

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


    is_ready_to_infer(available_features=available_features, needed_features = ["MeanRoofHt"], inferred_feature= "M.LRM class")

    if BIM['MeanRoofHt'] < 15.0:
        # extend the BIM dictionary

        is_ready_to_infer(available_features=available_features, needed_features = ["DesignWindSpeed"], inferred_feature= "RoofDeckAttachmentM")
        essential_features = dict(
            BuildingTag = "M.LRM.1.", 
            TerrainRoughness=int(BIM['TerrainRoughness']),
            RoofCover = roof_cover,
            RoofDeckAttachmentW = RDA,
            RoofDeckAttachmentM = MRDA,
            RoofDeckAge = DQ,
            RoofToWallConnection = RWC,
            Shutters = int(shutters),
            MasonryReinforcing = int(BIM['MasonryReinforcing']),
            WindDebrisClass = WIDD,
            RoofSystem = BIM['RoofFrameType']
            )

        BIM.update(dict(essential_features))

        # if it's MLRM1, configure outputs
        # bldg_config = f"M.LRM.1." \
        #               f"{roof_cover}." \
        #               f"{int(shutters)}." \
        #               f"{int(BIM['MasonryReinforcing'])}." \
        #               f"{WIDD}." \
        #               f"{BIM['RoofFrameType']}." \
        #               f"{RDA}." \
        #               f"{RWC}." \
        #               f"{DQ}." \
        #               f"{MRDA}." \
        #               f"{int(BIM['TerrainRoughness'])}"

    else:
        
        if "UnitType" in BIM:
            unit_tag = BIM["UnitType"]

        elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofFrameType"], inferred_feature= "UnitType"):
            if BIM['RoofFrameType'] == 'trs':
                unit_tag = 'null'
            elif BIM['RoofFrameType'] == 'ows':
                is_ready_to_infer(available_features=available_features, needed_features = ["NumberOfUnits"], inferred_feature= "UnitType")
                if BIM['NumberOfUnits'] == 1:
                    unit_tag = 'sgl'
                else:
                    unit_tag = 'mlt'

        if "JoistSpacing" in BIM:
            joist_spacing = BIM["JoistSpacing"]
            
        elif is_ready_to_infer(available_features=available_features, needed_features = ["RoofFrameType"], inferred_feature= "JoistSpacing"):
            # MLRM2 needs more rulesets
            if BIM['RoofFrameType'] == 'trs':
                joist_spacing = 'null'
            elif BIM['RoofFrameType'] == 'ows':
                is_ready_to_infer(available_features=available_features, needed_features = ["NumberOfUnits"], inferred_feature= "JoistSpacing")
                if BIM['NumberOfUnits'] == 1:
                    joist_spacing = 'null'
                else:
                    joist_spacing = 4

        is_ready_to_infer(available_features=available_features, needed_features = ['TerrainRoughness','MasonryReinforcing','RoofFrameType'], inferred_feature= "M.LRI class")

        essential_features = dict(
            BuildingTag = "M.LRM.2.", 
            TerrainRoughness=int(BIM['TerrainRoughness']),
            RoofCover = roof_cover,
            RoofDeckAttachmentW = RDA,
            RoofToWallConnection = RWC,
            Shutters = int(shutters),
            MasonryReinforcing = int(BIM['MasonryReinforcing']),
            RoofFrameType = BIM['RoofFrameType'],
            WindDebrisClass = WIDD,
            RoofDeckAttachmentM = MRDA,
            RoofDeckAge = DQ,
            UnitType=unit_tag,
            JoistSpacing=joist_spacing
            )

        # extend the BIM dictionary
        BIM.update(dict(essential_features))

        # bldg_config = f"M.LRM.2." \
        #               f"{roof_cover}." \
        #               f"{int(shutters)}." \
        #               f"{int(BIM['MasonryReinforcing'])}." \
        #               f"{WIDD}." \
        #               f"{BIM['RoofFrameType']}." \
        #               f"{RDA}." \
        #               f"{RWC}." \
        #               f"{DQ}." \
        #               f"{MRDA}." \
        #               f"{unit_tag}." \
        #               f"{joist_spacing}." \
        #               f"{int(BIM['TerrainRoughness'])}"
        
    return essential_features