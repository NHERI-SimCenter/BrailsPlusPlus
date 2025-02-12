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
from brails.inferers.hazus_hurricane_inferer.WindMetaVarRulesets import is_ready_to_infer

def HUEFFS_config(BIM):
    """
    Rules to identify a HAZUS HUEFFS/HUEFSS configuration based on BIM data

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

    if "RoofCover" in BIM:
        roof_cover = BIM["RoofCover"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofCover"):
        # Roof cover
        if BIM['YearBuilt'] >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    if "WindDebrisClass" in BIM:
        WIDD = BIM["WindDebrisClass"]

    else:
        # Wind debris
        WIDD = 'A'

    if "RoofDeckAge" in BIM:
        DQ = BIM["RoofDeckAge"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofDeckAge"):
        # Roof deck age
        if BIM['YearBuilt'] >= (datetime.datetime.now().year - 50):
            DQ = 'god' # new or average
        else:
            DQ = 'por' # old

    if "RoofDeckAttachmentM" in BIM:
        MRDA = BIM["RoofDeckAttachmentM"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt","DesignWindSpeed"], inferred_feature= "RoofDeckAttachmentM"):
        # Metal-RDA
        if BIM['YearBuilt'] > 2000:
            if BIM['DesignWindSpeed'] <= 142:
                MRDA = 'std'  # standard
            else:
                MRDA = 'sup'  # superior
        else:
            MRDA = 'std'  # standard

    if "Shutters" in BIM:
        shutters = BIM["Shutters"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["WBD"], inferred_feature= "Shutters"):
        # Shutters
        shutters = int(BIM['WBD'])

    essential_features = dict(
        BuildingTag = 'HUEF.FS', 
        TerrainRoughness=int(BIM['TerrainRoughness']),
        RoofCover = roof_cover,
        RoofDeckAttachmentM = MRDA,
        RoofDeckAge=DQ,
        WindDebrisClass = WIDD,
        Shutters=int(shutters)
        )

    BIM.update(essential_features)

    # bldg_tag = 'HUEF.FS'
    # bldg_config = f"{bldg_tag}." \
    #               f"{roof_cover}." \
    #               f"{shutters}." \
    #               f"{WIDD}." \
    #               f"{DQ}." \
    #               f"{MRDA}." \
    #               f"{int(BIM['TerrainRoughness'])}"

    return essential_features

def HUEFSS_config(BIM):
    """
    Rules to identify a HAZUS HUEFFS/HUEFSS configuration based on BIM data

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

    if "RoofCover" in BIM:
        roof_cover = BIM["RoofCover"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofCover"):
        # Roof cover
        if BIM['YearBuilt'] >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    if "WindDebrisClass" in BIM:
        WIDD = BIM["WindDebrisClass"]
    else:
        WIDD = 'A'


    if "RoofDeckAge" in BIM:
        DQ = BIM["RoofDeckAge"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofDeckAge"):
        # Roof deck age
        if BIM['YearBuilt'] >= (datetime.datetime.now().year - 50):
            DQ = 'god' # new or average
        else:
            DQ = 'por' # old



    if "RoofDeckAttachmentM" in BIM:
        MRDA = BIM["RoofDeckAttachmentM"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt","DesignWindSpeed"], inferred_feature= "RoofDeckAttachmentM"):
        # Metal-RDA
        if BIM['YearBuilt'] > 2000:
            if BIM['DesignWindSpeed'] <= 142:
                MRDA = 'std'  # standard
            else:
                MRDA = 'sup'  # superior
        else:
            MRDA = 'std'  # standard

    if "Shutters" in BIM:
        shutters = BIM["Shutters"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["WindBorneDebris"], inferred_feature= "Shutters"):
        # Shutters
        shutters = BIM['WindBorneDebris']


    essential_features = dict(
        BuildingTag = 'HUEF.S.S', 
        TerrainRoughness=int(BIM['TerrainRoughness']),
        RoofCover = roof_cover,
        RoofDeckAttachmentM = MRDA,
        RoofDeckAge=DQ,
        WindDebrisClass = WIDD,
        Shutters=int(shutters)
        )

    BIM.update(essential_features)
    # extend the BIM dictionary

    # bldg_tag = 'HUEF.S.S'
    # bldg_config = f"{bldg_tag}." \
    #               f"{roof_cover}." \
    #               f"{int(shutters)}." \
    #               f"{WIDD}." \
    #               f"{DQ}." \
    #               f"{MRDA}." \
    #               f"{int(BIM['TerrainRoughness'])}"

    return essential_features


def HUEFH_config(BIM):
    """
    Rules to identify a HAZUS HUEFH configuration based on BIM data

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
    if "RoofCover" in BIM:
        roof_cover = BIM["RoofCover"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofCover"):
        
        # Roof cover
        if BIM['YearBuilt'] >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    if "WindDebrisClass" in BIM:
        WIDD = BIM["WindDebrisClass"]
    else:
        # Wind debris
        WIDD = 'A'

    if "Shutters" in BIM:
        shutters = BIM["Shutters"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["WindBorneDebris"], inferred_feature= "Shutters"):
        # Shutters
        shutters = BIM['WindBorneDebris']

    if "RoofDeckAttachmentM" in BIM:
        MRDA = BIM["RoofDeckAttachmentM"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt","DesignWindSpeed"], inferred_feature= "RoofDeckAttachmentM"):
        # Metal-RDA
        if BIM['YearBuilt'] > 2000:
            if BIM['DesignWindSpeed'] <= 142:
                MRDA = 'std'  # standard
            else:
                MRDA = 'sup'  # superior
        else:
            MRDA = 'std'  # standard

    is_ready_to_infer(available_features=available_features, needed_features = ["NumberOfStories"], inferred_feature= "Building Tag HUEF.H Class")
    if BIM['NumberOfStories'] <=2:
        bldg_tag = 'HUEF.H.S'
    elif BIM['NumberOfStories'] <= 5:
        bldg_tag = 'HUEF.H.M'
    else:
        bldg_tag = 'HUEF.H.L'

    essential_features = dict(
        BuildingTag = bldg_tag, 
        TerrainRoughness=int(BIM['TerrainRoughness']),
        RoofCover = roof_cover,
        RoofDeckAttachmentM = MRDA,
        WindDebrisClass = WIDD,
        Shutters=int(shutters)
        )

    BIM.update(essential_features)

    # bldg_config = f"{bldg_tag}." \
    #               f"{roof_cover}." \
    #               f"{WIDD}." \
    #               f"{MRDA}." \
    #               f"{int(shutters)}." \
    #               f"{int(BIM['TerrainRoughness'])}"

    return essential_features

def HUEFS_config(BIM):
    """
    Rules to identify a HAZUS HUEFS configuration based on BIM data

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
    if "RoofCover" in BIM:
        roof_cover = BIM["RoofCover"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofCover"):
        # Roof cover
        if BIM['YearBuilt'] >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'


    if "WindDebrisClass" in BIM:
        WIDD = BIM["WindDebrisClass"]

    else:
        # Wind debris
        WIDD = 'C'


    if "Shutters" in BIM:
        shutters = BIM["Shutters"]
        
    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt","WindBorneDebris"], inferred_feature= "Shutters"):
        # Shutters
        if BIM['YearBuilt'] > 2000:
            shutters = BIM['WindBorneDebris']
        else:
            # year <= 2000
            if BIM['WindBorneDebris']:
                shutters = random.random() < 0.46
            else:
                shutters = False


    if "RoofDeckAttachmentM" in BIM:
        MRDA = BIM["RoofDeckAttachmentM"]
        
    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt","DesignWindSpeed"], inferred_feature= "RoofDeckAttachmentM"):
        # Metal-RDA
        if BIM['YearBuilt'] > 2000:
            if BIM['DesignWindSpeed'] <= 142:
                MRDA = 'std'  # standard
            else:
                MRDA = 'sup'  # superior
        else:
            MRDA = 'std'  # standard


    is_ready_to_infer(available_features=available_features, needed_features = ["NumberOfStories"], inferred_feature= "Building Tag HUEF.S")
    if BIM['NumberOfStories'] <=2:
        bldg_tag = 'HUEF.S.M'
    else:
        bldg_tag = 'HUEF.S.L'


    # extend the BIM dictionary
    
    essential_features = dict(
        BuildingTag = bldg_tag, 
        TerrainRoughness=int(BIM['TerrainRoughness']),
        RoofCover = roof_cover,
        RoofDeckAttachmentM = MRDA,
        WindDebrisClass = WIDD,
        Shutters=int(shutters)
        )

    BIM.update(essential_features)

    # bldg_config = f"{bldg_tag}." \
    #               f"{roof_cover}." \
    #               f"{int(shutters)}." \
    #               f"{WIDD}." \
    #               f"null." \
    #               f"{MRDA}." \
    #               f"{int(BIM['TerrainRoughness'])}"

    return essential_features # sy - modifting this