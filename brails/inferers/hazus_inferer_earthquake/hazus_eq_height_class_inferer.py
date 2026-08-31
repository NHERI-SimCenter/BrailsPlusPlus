# Copyright (c) 2025 The Regents of the University of California
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
Assign the Hazus height class of every asset in an inventory.

.. autosummary::

      HazusEQHeightClassInferer
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from brails.inferers.hazus_inferer_earthquake.hazus_eq_diagnosis import (
    diagnosis_frame,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_height_classes_by_type,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_OK,
    preview_failures,
)
from brails.inferers.inference_engine import InferenceEngine
from brails.types.asset_inventory import is_probabilistic_feature

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


class HazusEQHeightClassInferer(InferenceEngine):
    """
    Assign a Hazus height class to every asset in an inventory.

    Height class is a function of structure type and story count: URM
    splits at two stories, RM1 and URM have no High-Rise class at all,
    and W1, W2, S3, PC1 and MH carry no height class, since the damage
    models for those types ignore height. Assets of those five types get
    an explicit null height class, which is what the damage and loss
    library expects to find.

    Nothing is drawn here. The class carries the same check, draw and
    infer surface as the inferers that do draw, so a caller can screen
    and apply either one the same way.
    """

    def __init__(
        self,
        structure_type_key: str = 'StructureType',
        stories_key: str = 'NumberOfStories',
        height_class_key: str = 'HeightClass',
    ) -> None:
        """
        Set up the inferer.

        Args:
            structure_type_key (str):
                Feature name the structure type is read from.
            stories_key (str):
                Feature name the story count is read from.
            height_class_key (str):
                Feature name the height class is written to.
        """
        self.structure_type_key = structure_type_key
        self.stories_key = stories_key
        self.height_class_key = height_class_key
        self._classes_by_type = get_hazus_height_classes_by_type()

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Diagnose every asset without assigning anything.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.DataFrame:
                One row per asset id, with the columns of
                ``DIAGNOSIS_COLUMNS``. Height class follows from the
                structure type and the story count with nothing to
                restrict, so the restriction and conflict columns are
                always empty here.
        """
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            diagnosis, _ = self._resolve(
                features, input_inventory.vector_features
            )
            rows[asset_id] = (diagnosis, '', '')
        return diagnosis_frame(rows)

    def draw(self, input_inventory: AssetInventory) -> pd.Series:
        """
        Assign a height class to every asset, assuming all are usable.

        Args:
            input_inventory (AssetInventory):
                An inventory whose assets all passed :meth:`check`.

        Returns:
            pd.Series:
                Height classes indexed by asset id, None where the
                structure type carries none.

        Raises:
            ValueError:
                If an asset cannot be classified, which means
                :meth:`check` was skipped.
        """
        classes = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            diagnosis, height_class = self._resolve(
                features, input_inventory.vector_features
            )
            if diagnosis != DIAGNOSIS_OK:
                raise ValueError(
                    f'Cannot assign {self.height_class_key} for asset '
                    f'{asset_id}: {diagnosis}. Screen the inventory with '
                    'check() before calling draw().'
                )
            classes[asset_id] = height_class
        return pd.Series(classes, dtype=object, name=self.height_class_key)

    def infer(self, input_inventory: AssetInventory) -> AssetInventory:
        """
        Assign a height class to every asset in the inventory.

        Args:
            input_inventory (AssetInventory):
                The inventory to process. Every asset must carry a
                structure type, and a story count where its type needs
                one; screen with :meth:`check` first if that is not
                guaranteed.

        Returns:
            AssetInventory:
                A copy of the input with the height class added under
                ``height_class_key``.

        Raises:
            ValueError:
                If any asset cannot be classified, naming the assets and
                their diagnoses.
        """
        verdicts = self.check(input_inventory)
        failed = verdicts[verdicts['diagnosis'] != DIAGNOSIS_OK]
        if not failed.empty:
            reasons = preview_failures(failed['diagnosis'])
            raise ValueError(
                f'Cannot assign {self.height_class_key} for {len(failed)} of '
                f'{len(verdicts)} assets: {reasons}. Remove them or supply '
                'the missing data, then call infer() again.'
            )
        classes = self.draw(input_inventory)
        output_inventory = deepcopy(input_inventory)
        for asset_id, height_class in classes.items():
            output_inventory.add_asset_features(
                asset_id, {self.height_class_key: height_class}
            )
        return output_inventory

    def _resolve(
        self, features: dict, vector_features: set
    ) -> tuple[str, str | None]:
        """
        Work out an asset's height class, or why it has none.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors, where a
                list is one value and not one value per world.

        Returns:
            tuple[str, str | None]:
                The diagnosis and the height class, which is None both
                for a type that carries no height class and for an asset
                that cannot be classified.
        """
        structure_type = features.get(self.structure_type_key)
        if structure_type is None:
            return f'no {self.structure_type_key}', None
        if is_probabilistic_feature(
            self.structure_type_key, structure_type, vector_features
        ):
            return (
                f'{self.structure_type_key} is a per-world list; multi-world '
                'assignment is planned and not supported yet'
            ), None
        structure_type = str(structure_type)
        if structure_type not in self._classes_by_type:
            return (
                f'{self.structure_type_key} {structure_type!r} is not a '
                'Hazus structure type'
            ), None
        bands = self._classes_by_type[structure_type]
        if bands is None:
            return DIAGNOSIS_OK, None
        stories = features.get(self.stories_key)
        if stories is None:
            return (
                f'no {self.stories_key}, which {structure_type} needs for its '
                'height class'
            ), None
        if is_probabilistic_feature(
            self.stories_key, stories, vector_features
        ):
            return (
                f'{self.stories_key} is a per-world list; multi-world '
                'assignment is planned and not supported yet'
            ), None
        try:
            story_count = float(stories)
        except (TypeError, ValueError):
            return f'{self.stories_key} is not a number: {stories!r}', None
        if (
            not np.isfinite(story_count)
            or story_count != int(story_count)
            or story_count < 1
        ):
            return (
                f'{self.stories_key} of {stories!r} is not a whole number of '
                'stories of one or more'
            ), None
        return DIAGNOSIS_OK, self._class_of(bands, int(story_count))

    @staticmethod
    def _class_of(
        bands: dict[str, tuple[int, int | None]], stories: int
    ) -> str:
        """
        Select the height class a story count falls in.

        Args:
            bands (dict[str, tuple[int, int | None]]):
                Height class to its inclusive story range.
            stories (int):
                Story count, one or more.

        Returns:
            str:
                The height class.
        """
        for height_class, (lowest, highest) in bands.items():
            if stories >= lowest and (highest is None or stories <= highest):
                return height_class
        raise ValueError(f'No height class covers {stories} stories.')
