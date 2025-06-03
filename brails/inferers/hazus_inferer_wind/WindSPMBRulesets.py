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


def SPMB_config(BIM):
    """
    Rules to identify a HAZUS SPMB configuration based on BIM data

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


    if "RoofQuality" in BIM:
        roof_quality = BIM["RoofQuality"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt"], inferred_feature= "RoofQuality"):

        # Roof Deck Age (~ Roof Quality)
        if BIM['YearBuilt'] >= (datetime.datetime.now().year - 50):
            roof_quality = 'Good'
        else:
            roof_quality = 'Poor'


    if "Shutters" in BIM:
        shutters = BIM["Shutters"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["YearBuilt","WindBorneDebris"], inferred_feature= "Shutters"):

        # shutters
        if BIM['YearBuilt'] >= 2000:
            shutters = BIM['WindBorneDebris']
        # BOCA 1996 and earlier:
        # Shutters were not required by code until the 2000 IBC. Before 2000, the
        # percentage of commercial buildings that have shutters is assumed to be
        # 46%. This value is based on a study on preparedness of small businesses
        # for hurricane disasters, which says that in Sarasota County, 46% of
        # business owners had taken action to wind-proof or flood-proof their
        # facilities. In addition to that, 46% of business owners reported boarding
        # up their businesses before Hurricane Katrina. In addition, compliance
        # rates based on the Homeowners Survey data hover between 43 and 50 percent.
        else:
            if BIM['WindBorneDebris']:
                shutters = random.random() < 0.46
            else:
                shutters = False


    if "RoofDeckAttachment" in BIM:
        MRDA = BIM["RoofDeckAttachment"]

    elif is_ready_to_infer(available_features=available_features, needed_features = ["DesignWindSpeed"], inferred_feature= "RoofDeckAttachment"):
        # Metal RDA
        # 1507.2.8.1 High Wind Attachment.
        # Underlayment applied in areas subject to high winds (Vasd greater
        # than 110 mph as determined in accordance with Section 1609.3.1) shall
        #  be applied with corrosion-resistant fasteners in accordance with
        # the manufacturer’s instructions. Fasteners are to be applied along
        # the overlap not more than 36 inches on center.
        if BIM['DesignWindSpeed'] > 142:
            MRDA = 'Standard'  # standard
        else:
            MRDA = 'Superior'  # superior


    # don't allow predefined its too complicated
    is_ready_to_infer(available_features=available_features, needed_features = ["PlanArea",'LandCover','StructureType','BuildingType'], inferred_feature= "S.PMB class")

    if BIM['PlanArea'] <= 4000:
        bldg_tag = 'S.PMB.S'
    elif BIM['PlanArea'] <= 50000:
        bldg_tag = 'S.PMB.M'
    else:
        bldg_tag = 'S.PMB.L'

    essential_features = dict(
        BuildingType=BIM['BuildingType'],
        StructureType=BIM['StructureType'],
        LandCover=BIM['LandCover'],
        RoofQuality = roof_quality,
        RoofDeckAttachment = MRDA,
        Shutters = int(shutters),
        PlanArea = float(BIM['PlanArea'])
        )

    # extend the BIM dictionary
    BIM.update(dict(essential_features))

    # bldg_config = f"{bldg_tag}." \
    #               f"{int(shutters)}." \
    #               f"{roof_quality}." \
    #               f"{MRDA}." \
    #               f"{BIM['LandCover']}"

    return essential_features

