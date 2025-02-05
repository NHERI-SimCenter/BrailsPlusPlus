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
# Frank McKenna
#
# Based on rulesets developed by:
# Karen Angeles
# Meredith Lockhead
# Tracy Kijewski-Correa


from brails.inferers.hazus_hurricane_inferer.WindMetaVarRulesets import add_default
from brails.inferers.hazus_hurricane_inferer.BuildingClassRulesets import building_class
from brails.inferers.hazus_hurricane_inferer.WindCECBRulesets import CECB_config
from brails.inferers.hazus_hurricane_inferer.WindCERBRulesets import CERB_config
from brails.inferers.hazus_hurricane_inferer.WindMECBRulesets import MECB_config
from brails.inferers.hazus_hurricane_inferer.WindMERBRulesets import MERB_config
from brails.inferers.hazus_hurricane_inferer.WindMHRulesets import MH_config
from brails.inferers.hazus_hurricane_inferer.WindMLRIRulesets import MLRI_config
from brails.inferers.hazus_hurricane_inferer.WindMLRMRulesets import MLRM_config
from brails.inferers.hazus_hurricane_inferer.WindMMUHRulesets import MMUH_config
from brails.inferers.hazus_hurricane_inferer.WindMSFRulesets import MSF_config
from brails.inferers.hazus_hurricane_inferer.WindSECBRulesets import SECB_config
from brails.inferers.hazus_hurricane_inferer.WindSERBRulesets import SERB_config
from brails.inferers.hazus_hurricane_inferer.WindSPMBRulesets import SPMB_config
from brails.inferers.hazus_hurricane_inferer.WindWMUHRulesets import WMUH_config
from brails.inferers.hazus_hurricane_inferer.WindWSFRulesets import WSF_config

def auto_populate(inventory):
    """
    Populates the DL model for hurricane assessments in Atlantic County, NJ

    Assumptions:
    - Everything relevant to auto-population is provided in the Buiding
    Information Model (AIM).
    - The information expected in the AIM file is described in the parse_AIM
    method.

    Parameters
    ----------
    aim: dictionary
        Contains the information that is available about the asset and will be
        used to auto-popualate the damage and loss model.

    Returns
    -------
    GI_ap: dictionary
        Containes the extended AIM data.
    DL_ap: dictionary
        Contains the auto-populated loss model.
    """

    # parse the GI data
    GI_ap = add_default(inventory["properties"], hazards='wind')

    #print(GI_ap[0])
    # identify the building class

    bldg_class = building_class(GI_ap, hazard='wind')

    # prepare the building configuration string
    if bldg_class == 'WSF':
        essential_features = WSF_config(GI_ap)
    elif bldg_class == 'WMUH':
        essential_features = WMUH_config(GI_ap)
    elif bldg_class == 'MSF':
        essential_features = MSF_config(GI_ap)
    elif bldg_class == 'MMUH':
        essential_features = MMUH_config(GI_ap)
    elif bldg_class == 'MLRM':
        essential_features = MLRM_config(GI_ap)
    elif bldg_class == 'MLRI':
        essential_features = MLRI_config(GI_ap)
    elif bldg_class == 'MERB':
        essential_features = MERB_config(GI_ap)
    elif bldg_class == 'MECB':
        essential_features = MECB_config(GI_ap)
    elif bldg_class == 'CECB':
        essential_features = CECB_config(GI_ap)
    elif bldg_class == 'CERB':
        essential_features = CERB_config(GI_ap)
    elif bldg_class == 'SPMB':
        essential_features = SPMB_config(GI_ap)
    elif bldg_class == 'SECB':
        essential_features = SECB_config(GI_ap)
    elif bldg_class == 'SERB':
        essential_features = SERB_config(GI_ap)
    elif bldg_class == 'MH':
        essential_features = MH_config(GI_ap)
    else:
        raise ValueError(
            f"Building class {bldg_class} not recognized by the "
            f"auto-population routine."
        )


    return essential_features
