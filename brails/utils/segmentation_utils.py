# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 The Regents of the University of California
#
# This file is part of BRAILS.
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
# 08-22-2024


import os
from PIL import Image
from pathlib import Path
import numpy as np


def create_overlaid_image(base_image, mask_image, output_dir):

    def decode_segmap(image, nc=5):
        label_colors = np.array([(0, 0, 0), (255, 0, 0), (255, 165, 0),
                                 (0, 0, 255), (175, 238, 238)])
        # 0=background # 1=roof, 2=facade, 3=window, 4=door

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    mask_rgb = decode_segmap(mask_image, nc=4)

    image = Image.open(base_image)  # Replace with your image file
    base_image_np = np.array(image)

    mask_image_resized = np.array(Image.fromarray(mask_rgb))

    # Define alpha (transparency) for blending the mask with the base image
    alpha = 0.5

    # Create an overlay image by blending the mask and base image
    # Convert mask and base image to float for blending
    base_image_float = base_image_np.astype(float)
    mask_image_float = mask_image_resized.astype(float)

    # Blend images
    combined_image_float = (alpha * mask_image_float +
                            (1 - alpha) * base_image_float)

    # Clip values to [0, 255] and convert back to uint8
    combined_image_np = np.clip(combined_image_float, 0, 255).astype(np.uint8)

    # Convert back to PIL Image and save
    imname = base_image.split('/')[-1]
    ext = imname.split('.')[-1]
    imout_name = imname.replace(ext, '') + '_segmented.' + ext
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir)

    combined_image_pil = Image.fromarray(combined_image_np)
    combined_image_pil.save(path / imout_name)
