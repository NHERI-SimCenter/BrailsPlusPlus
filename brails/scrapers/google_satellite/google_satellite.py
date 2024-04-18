# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The Regents of the University of California
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
# 03-08-2024

from brails.types.image_set import ImageSet
from brails.types.asset_inventory import AssetInventory
from brails.scrapers.image_scraper import ImageScraper

import os
import requests
import sys
import math
import concurrent.futures
import numpy as np
import base64
import struct
import json
import matplotlib as mpl

from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
from math import radians, sin, cos, atan2, sqrt, log, floor
from shapely.geometry import Point, Polygon, MultiPoint
from tqdm import tqdm
from pathlib import Path

class GoogleSatellite:

    def __init__(self, input_data: dict):

        api_key = input_data["apiKey"]

        # Check if the provided Google API Key successfully obtains street view
        # imagery metadata for Doe Memorial Library of UC Berkeley:
        responseStreet = requests.get(
            "https://maps.googleapis.com/maps/api/streetview/metadata?"
            + "location=37.8725187407,-122.2596028649"
            + "&source=outdoor"
            + f"&key={api_key}"
        )

        # If the requested image cannot be downloaded, notify the user of the
        # error and stop program execution:
        if "error" in responseStreet.text.lower():
            error_message = (
                "Google API key error. The entered API key is valid "
                + "but does not have Street View Static API enabled. "
                + "Please enter a key that has the Street View"
                + "Static API enabled."
            )
            sys.exit(error_message)

        self.apikey = api_key

    def GetGoogleSatelliteImage(self, footprints, dir_location):

        self.dir_location = dir_location

        def download_satellite_image(footprint, impath):

            bbox_buffered = bufferedfp(footprint)
            (xlist, ylist) = determine_tile_coords(bbox_buffered)

            imname = impath.split("/")[-1]
            imname = imname.replace("." + imname.split(".")[-1], "")

            base_url = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

            # Define a retry stratey fgor common error codes to use when
            # downloading tiles:
            s = requests.Session()
            retries = Retry(
                total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
            )
            s.mount("https://", HTTPAdapter(max_retries=retries))

            # Get the tiles for the satellite image:
            tiles = []
            offsets = []
            imbnds = []
            ntiles = (len(xlist), len(ylist))
            for yind, y in enumerate(ylist):
                for xind, x in enumerate(xlist):
                    url = base_url.format(x=x, y=y, z=20)

                    # Download tile using the defined retry strategy:
                    response = s.get(url)

                    # Save downloaded tile as a PIL image in tiles and calculate
                    # tile offsets and bounds:
                    tiles.append(Image.open(BytesIO(response.content)))
                    offsets.append((xind * 256, yind * 256))
                    tilebnds = tile_bbox(zoom=20, x=x, y=y)

                    # If the number of tiles both in x and y directions are greater than 1:
                    if ntiles[0] > 1 and ntiles[1] > 1:
                        if xind == 0 and yind == 0:
                            imbnds.append(tilebnds[2])
                            imbnds.append(tilebnds[0])
                        elif xind == ntiles[0] - 1 and yind == 0:
                            imbnds.append(tilebnds[3])
                        elif xind == 0 and yind == ntiles[1] - 1:
                            imbnds.append(tilebnds[1])
                    # If the total number of tiles is 1:
                    elif ntiles[0] == 1 and ntiles[1] == 1:
                        imbnds = [tilebnds[2], tilebnds[0], tilebnds[3], tilebnds[1]]
                    # If the total number of tiles is 1 in x-direction and greater than 1 in y-direction:
                    elif ntiles[0] == 1:
                        if yind == 0:
                            imbnds.append(tilebnds[2])
                            imbnds.append(tilebnds[0])
                            imbnds.append(tilebnds[3])
                        elif yind == ntiles[1] - 1:
                            imbnds.append(tilebnds[1])
                    # If the total number of tiles is greater than 1 in x-direction and 1 in y-direction:
                    elif ntiles[1] == 1:
                        if xind == 0:
                            imbnds.append(tilebnds[2])
                            imbnds.append(tilebnds[0])
                        elif xind == ntiles[0] - 1:
                            imbnds.append(tilebnds[3])
                            imbnds.append(tilebnds[1])

            # Combine tiles into a single image using the calculated offsets:
            combined_im = Image.new("RGB", (256 * ntiles[0], 256 * ntiles[1]))
            for ind, im in enumerate(tiles):
                combined_im.paste(im, offsets[ind])

            # Crop combined image around the footprint of the building:
            lonrange = imbnds[2] - imbnds[0]
            latrange = imbnds[3] - imbnds[1]

            left = math.floor(
                (bbox_buffered[0][0] - imbnds[0]) / lonrange * 256 * ntiles[0]
            )
            right = math.ceil(
                (bbox_buffered[0][-1] - imbnds[0]) / lonrange * 256 * ntiles[0]
            )
            bottom = math.ceil(
                (bbox_buffered[1][0] - imbnds[1]) / latrange * 256 * ntiles[1]
            )
            top = math.floor(
                (bbox_buffered[1][1] - imbnds[1]) / latrange * 256 * ntiles[1]
            )

            cropped_im = combined_im.crop((left, top, right, bottom))

            # Pad the image in horizontal or vertical directions to make it
            # square:
            (newdim, indmax, mindim, _) = maxmin_and_ind(cropped_im.size)
            padded_im = Image.new("RGB", (newdim, newdim))
            buffer = round((newdim - mindim) / 2)

            if indmax == 1:
                padded_im.paste(cropped_im, (buffer, 0))
            else:
                padded_im.paste(cropped_im, (0, buffer))

            # Resize the image to 640x640 and save it to impath:
            resized_im = padded_im.resize((640, 640))
            resized_im.save(impath)

        def determine_tile_coords(bbox_buffered):
            # Determine the tile x,y coordinates covering the area the bounding
            # box:
            xlist = []
            ylist = []
            for ind in range(4):
                (lat, lon) = (bbox_buffered[1][ind], bbox_buffered[0][ind])
                x, y = deg2num(lat, lon, 20)
                xlist.append(x)
                ylist.append(y)

            xlist = list(range(min(xlist), max(xlist) + 1))
            ylist = list(range(min(ylist), max(ylist) + 1))
            return (xlist, ylist)

        def bufferedfp(footprint):
            # Place a buffer around the footprint to account for footprint
            # inaccuracies with tall buildings:
            lon = [coord[0] for coord in footprint]
            lat = [coord[1] for coord in footprint]

            minlon = min(lon)
            maxlon = max(lon)
            minlat = min(lat)
            maxlat = max(lat)

            londiff = maxlon - minlon
            latdiff = maxlat - minlat

            minlon_buff = minlon - londiff * 0.1
            maxlon_buff = maxlon + londiff * 0.1
            minlat_buff = minlat - latdiff * 0.1
            maxlat_buff = maxlat + latdiff * 0.1

            bbox_buffered = (
                [minlon_buff, minlon_buff, maxlon_buff, maxlon_buff],
                [minlat_buff, maxlat_buff, maxlat_buff, minlat_buff],
            )
            return bbox_buffered

        def deg2num(lat, lon, zoom):
            # Calculate the x,y coordinates corresponding to a lot/lon pair
            # in degrees for a given zoom value:
            lat_rad = math.radians(lat)
            n = 2**zoom
            xtile = int((lon + 180) / 360 * n)
            ytile = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
            return (xtile, ytile)

        def tile_bbox(zoom: int, x: int, y: int):
            # [south,north,west,east]
            return [
                tile_lat(y, zoom),
                tile_lat(y + 1, zoom),
                tile_lon(x, zoom),
                tile_lon(x + 1, zoom),
            ]

        def tile_lon(x: int, z: int) -> float:
            return x / math.pow(2.0, z) * 360.0 - 180

        def tile_lat(y: int, z: int) -> float:
            return math.degrees(
                math.atan(math.sinh(math.pi - (2.0 * math.pi * y) / math.pow(2.0, z)))
            )

        def maxmin_and_ind(sizelist):
            maxval = max(sizelist)
            indmax = sizelist.index(maxval)
            minval = min(sizelist)
            indmin = sizelist.index(minval)
            return (maxval, indmax, minval, indmin)

        # Save footprints in the class object:
        self.footprints = footprints[:]

        # Compute building footprints, parse satellite image names, and save
        # them in the class object. Also, create the list of inputs required
        # for downloading satellite images in parallel:
        self.centroids = []
        self.satellite_images = []
        inps = []
        for fp in footprints:
            fp_cent = Polygon(fp).centroid
            self.centroids.append([fp_cent.x, fp_cent.y])
            imName = str(round(fp_cent.y, 8)) + str(round(fp_cent.x, 8))
            imName.replace(".", "")
            im_name = f"{self.dir_location}/imsat_{imName}.jpg"
            self.satellite_images.append(im_name)
            inps.append((fp, im_name))

        # Create a directory to save the satellite images:
        os.makedirs(self.dir_location, exist_ok=True)

        # Download satellite images corresponding to each building footprint:
        pbar = tqdm(total=len(footprints), desc="Obtaining satellite imagery")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(download_satellite_image, fp, fout): fp
                for fp, fout in inps
            }
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                pbar.update(n=1)
                try:
                    future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (url, exc))

    def get_images(self, inventory: AssetInventory, dir_path: str) -> ImageSet:
        """
        This method obtaines images given the foorprints in the asset inventory. T

        Args:
              inventory (AssetInventory):
                   The AssetInventory.
              dir_location (string):
                   The directory in which to place the images.

        Returns:
              Image_Set:
                    An image_Set for the assets in the inventory.

        """

        # ensure consistance in dir_path, i.e remove ending / if given
        
        dir_path = Path(dir_path)

        #
        # create the footprints from the asset inventory assets
        # keep the asset kets in a list for when done
        #

        result = ImageSet()

        result.dir_path = dir_path

        asset_footprints = []
        asset_keys = []

        for key, asset in inventory.inventory.items():
            asset_footprints.append(asset.coordinates)
            asset_keys.append(key)

        #
        # get the images
        #

        self.GetGoogleSatelliteImage(asset_footprints, dir_path)

        for key, im in zip(asset_keys, self.satellite_images):
            if (im is not None):            
                # strip off dirpath
                #im_stripped = im.replace(dir_path, "")
                im_stripped = Path(im).name      
                result.add_image(key, im_stripped)
            
        return result


1
