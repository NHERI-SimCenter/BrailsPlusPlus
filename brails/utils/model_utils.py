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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 07-24-2025

"""
This module provides a utility class for computer vision models in BRAILS.

.. autosummary::

      ModelUtils
"""

import os
import torch


class ModelUtils:
    """
    Utility class for computer vision models in BRAILS.

    This class provides static methods to ensure necessary model files are
    available locally, downloading them if needed. Intended for use in
    applications that rely on pre-trained model weights.

    Methods:
        get_model_path(model_path, default_filename, download_url,
                       model_description):
            Ensures a model file exists locally or downloads it to a default
            location.
    """

    @staticmethod
    def get_model_path(
        model_path: str = None,
        default_filename: str = None,
        download_url: str = None,
        model_description: str = 'model'
    ) -> str:
        """
        Ensure a model file is available locally, downloading it if needed.

        Args:
            model_path (str, optional):
                Custom path to the model file. If provided, no download occurs.
            default_filename (str, optional):
                Filename to use if downloading the model.
            download_url (str, optional):
                URL to download the model if it doesn't exist locally.
            model_description (str):
                Human-readable description of the model.

        Returns:
            str:
                Path to the model file.

        Raises:
            ValueError:
                If model_path is not provided and either default_filename or
                download_url is missing.
        """
        if model_path is None:
            if not default_filename or not download_url:
                raise ValueError(
                    'If model_path is not provided, default_filename and '
                    'download_url are required.'
                )

            os.makedirs('tmp/models', exist_ok=True)
            model_path = os.path.join('tmp/models', default_filename)

            if not os.path.isfile(model_path):
                print(f'\n\nLoading default {model_description} file to '
                      'tmp/models folder...')
                torch.hub.download_url_to_file(
                    download_url,
                    model_path,
                    progress=True
                )
                print(f'Default {model_description} loaded.')
            else:
                print(f'\nDefault {model_description} in {model_path} loaded.')
        else:
            print(f'\nInferences will be performed using the custom '
                  f'{model_description} in {model_path}.')

        return model_path
