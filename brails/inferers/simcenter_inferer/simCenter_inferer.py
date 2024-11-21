# -*- coding: utf-8 -*-
#
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
# Sang-ri Yi
#
# Last updated:
# 11-12-2024

import time

import os
import sys
import copy
import json

import numpy as np
from copy import deepcopy
import logging

from brails.types.asset_inventory import AssetInventory
from brails.inferers.inferenceEngine import InferenceEngine
from brails.inferers.hazus_inferer.hazus_inferer import HazusInferer

from itertools import product

import reverse_geocode # sy - note this may not be the most accurate package but it's fast

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimCenterInferer(HazusInferer):
    """
    Imputes dataset based on k-nearest neighbors in the feature-agmented space. Sequentially generate inventory

    Attributes:
        n_pw (int):
                The number of possible worlds (i.e. samples or realizations)
        seed (int):
                For reproducibility

    Methods:


    """
    # 

    def __init__(self):
        super().__init__()
        self.options = ['no_urm', 'allow_mh_only_for_res2', 'res3_AB_to_res1']

    def modulate_weights(self, weights, structure_types, region, occ, year_class, height):

        if len(weights)==0:
            # do nothing
            return weights, structure_types

        
        # if not RES2, turn off mobile home
        if "allow_mh_only_for_res2" in self.options:
            if (not (occ=="RES2")) and ('MH' in structure_types):
                # find MH and remove it from the list
                MHidx = np.argmax(structure_types == 'MH')
                #MHidx = structure_types.index('MH')
                if weights[MHidx]>0:
                    structure_types = np.delete(structure_types, MHidx)
                    weights = np.delete(weights, MHidx)
                    weights = weights/np.sum(weights)

        # turn of urm
        if "no_urm" in self.options:
            if "URM" in structure_types:
                URMidx = np.argmax(structure_types == 'URM')
                #URMidx = structure_types.index('URM')
                if weights[URMidx]>0:
                    structure_types = np.delete(structure_types, URMidx)
                    weights = np.delete(weights, URMidx)
                    weights = weights/np.sum(weights)
        
        return weights, structure_types

    def modulate_occ(self, s):

        # if RES3 and unit<4 (i.e. RES3A or RES3B), change it to RES1
        if "res3_AB_to_res1" in self.options:
            if not s[-1].isdigit():  # Check if the last character is not a digit
                if s in ['RES3A','RES3B']:
                    s = "RES1"
                elif s in ['RES3C','RES3D','RES3E','RES3F']:
                    s = "RES3"
        else:
            s = super().modulate_occ(s)

        return s