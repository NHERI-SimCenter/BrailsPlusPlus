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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
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
#
# Contributors:
# Frank McKenna
# Barbaros Cetiner
#
# Last updated:
# 11-06-2024

"""
This module defines abstract filter class.

.. autosummary::

    Filter
"""

from abc import ABC, abstractmethod
from brails.types.image_set import ImageSet


class Filter(ABC):
    """
    Abstract base class for filtering an ImageSet.

    Methods:
        filter(images, dir_path): Filters the ImageSet based on specific
            criteria.
    """

    def __init__(self, input_data: dict):
        """
        Initialize the Filter with the provided input.

        Args:
            input_data (dict): Configuration or parameters required for
                filtering.
        """
        self.input_data = input_data

    @abstractmethod
    def filter(self, images: ImageSet, dir_path: str) -> ImageSet:
        """
        Abstract class that must be implemented by subclasses.

        This method will be used by the caller to obtain a filtered ImageSet
        and save the output in a specified directory.

        Args:
            images (ImageSet): The ImageSet to filter.
            dir_path (str): Path to the output directory for filtered images.

        Returns:
            ImageSet: The filtered set of images.
        """
