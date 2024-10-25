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
# Barbaros Cetiner
#
# Last updated:
# 10-24-2024

"""
This module defines GoogleStreetview class downloading Google street imagery.

.. autosummary::

    GoogleStreetview
"""

import math
import struct
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from io import BytesIO
from pathlib import Path
import re
import logging
from typing import Any, Union, Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib as mpl
import PIL
from PIL import Image

from brails.types.image_set import ImageSet
from brails.types.asset_inventory import AssetInventory
from brails.scrapers.image_scraper import ImageScraper

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants:
BASE_API_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata'
PANORAMA_METADATA_URL = 'https://www.google.com/maps/photometa/v1'
TILE_URL_TEMPLATE = 'https://cbk0.google.com/cbk?output=tile' + \
                    '&panoid={pano_id}&zoom={zoom}&x={x}&y={y}'
ZOOM2FOV_MAPPER = {0: 360, 1: 180, 2: 90, 3: 45, 4: 22.5, 5: 11.25}
TILE_SIZE = 512
CIRCUM_EARTH_FT = 131482560

REQUESTS_RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)


class GoogleStreetview(ImageScraper):
    """
    A class that downloads street-level imagery and depth maps for buildings.

    This class interfaces with the Google Street View API to obtain
    high-resolution street-level images and corresponding depth maps based on
    specified building footprints. It supports saving interim images and
    collecting various camera metadata.

    Attributes:
        api_key (str): API key for authenticating requests to the Google
            Street View API.

    Methods:
        get_images(inventory: AssetInventory, save_directory: str) -> ImageSet:
            Retrieves street-level imagery, depthmap, building location and
            camera parameters for the assets in the given inventory and saves
            them to the specified save_directory.
    """

    def __init__(self, input_data: dict):
        """
        Initialize the GoogleStreetview object.

        This constructor initializes an instance of the GoogleStreetview class
        by validating the provided Google API key.

        Args:
            input_data (dict): A dictionary containing the following key:
                - 'apiKey' (str): A valid Google API key with Street View
                    Static API enabled.

        Raises:
            ValueError: If the 'apiKey' is missing or empty.
            ConnectionError: If the API key validation fails when checking
                against the Google Street View Static API.
        """
        try:
            api_key = input_data['apiKey']
            if not api_key:
                raise ValueError(
                    'API key is empty. Please provide a valid API key.')
        except KeyError as exception:
            raise ValueError('Please provide a Google API key to run the'
                             'GoogleStreetview module') from exception
        self._validate_api_key(api_key)
        self.api_key = api_key

    @staticmethod
    def _validate_api_key(api_key: str):
        """
        Validate the provided Google API Key for the Street View Static API.

        This method checks whether the specified API key is valid and has
        access to the Street View Static API by attempting to retrieve metadata
        for a specific location (Doe Memorial Library of UC Berkeley). If the
        API key is valid but does not have access to the required API, a
        ValueError is raised. If the request fails due to a connection error, a
        ConnectionError is raised.

        Args:
            api_key (str): The Google API Key to validate.

        Raises:
            ValueError: If the API key is valid but does not have the
                        Street View Static API enabled.
            ConnectionError: If there is a failure in making the request
                             to validate the API key.

        Example:
            >>> _validate_api_key('YOUR_API_KEY')

        Notes:
            This method is a static utility and does not require an instance
            of the class to be invoked.
        """
        try:
            # Check if the provided Google API Key successfully obtains street
            # view imagery metadata for Doe Memorial Library of UC Berkeley:
            params = {'location': '37.8725187407,-122.2596028649',
                      'source': 'outdoor',
                      'key': api_key}
            response = requests.get(BASE_API_URL, params=params, timeout=30)

            # If the requested image cannot be downloaded, notify the user of
            # the error and stop program execution:
            if 'error' in response.text.lower():
                raise ValueError('Google API key error. The entered API key is'
                                 'valid but does not have Street View Static '
                                 'API enabled. Please enter a key that has '
                                 'the Street View Static API enabled.')
        except requests.RequestException as exception:
            raise ConnectionError(f"Failed to validate API key: {exception}") \
                from exception

    def get_images(self,
                   inventory: AssetInventory,
                   save_directory: str) -> ImageSet:
        """
        Get street-level images of buildings from footprints in AssetInventory.

        Args:
              inventory (AssetInventory): AssetInventory for which the images
                  will be retrieved.
            save_directory (str): Path to the folder where the retrieved images
                will be saved

        Returns:
              ImageSet: An ImageSet for the assets in the inventory.

        Raises:
            ValueError: If the provided inventory is not an instance of
                AssetInventory.
        """
        # Validate inputs:
        if not isinstance(inventory, AssetInventory):
            raise ValueError('Invalid AssetInventory provided.')

        # Ensure consistency in dir_path, i.e remove ending / if given:
        dir_path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info('\nImages will be saved to: %s\n', dir_path.resolve())

        # Create the footprints from the items in AssetInventory
        # Keep the asset keys in a list for later use:
        image_set = ImageSet()
        image_set.dir_path = str(dir_path)

        asset_footprints = []
        asset_keys = []
        for key, asset in inventory.inventory.items():
            asset_footprints.append(asset.coordinates)
            asset_keys.append(key)

        # Get the images:
        street_images, metadata = self._download_images(asset_footprints,
                                                        dir_path,
                                                        False,
                                                        True)
        # Get the images:
        for index, image_path in enumerate(street_images):
            image_set.add_image(asset_keys[index],
                                image_path.name,
                                metadata[image_path])

        return image_set

    def _download_images(self,
                         footprints: list[list[tuple[float, float]]],
                         save_dir: Path,
                         save_interim_images: bool,
                         save_all_cam_metadata: bool
                         ) -> tuple[list[Path], dict[str, list[Any]]]:
        """
        Download street-level imagery and depthmap for building footprints.

        Parameters:
            footprints (list[list[tuple[float, float]]]): List of building
                footprints.
            save_dir (Path): Directory to save the images and depthmaps.
            save_interim_images (bool): Whether to save interim images.
            save_all_cam_metadata (bool): Whether to save all camera metadata.

        Returns:
            tuple[list[Path], dict[str, list[Any]]]: A tuple containing:
                - A list of paths to the saved street images.
                - A dictionary with processed metadata, where keys are
                    different metadata fields (e.g., 'camElev', 'depthMap',
                    etc.) and values are lists of corresponding metadata values
                    for each image.
        """
        # Compute building footprints, parse street-level image and depthmap
        # file names, and create the list of inputs required for obtaining
        # street-level imagery and corresponding depthmap:
        street_image_paths = []
        bldg_centroids = []
        inps = []

        for footprint_raw in footprints:
            footprint = np.fliplr(np.squeeze(np.array(footprint_raw))).tolist()
            footprint_cent = Polygon(footprint_raw).centroid
            image_name = f'{footprint_cent.y:.8f}_' + \
                f'{footprint_cent.x:.8f}'
            image_name.replace('.', '')
            image_path = save_dir / f'gstrt_{image_name}.jpg'
            depthmap_path = save_dir / f'gdmap_{image_name}.txt'
            street_image_paths.append(image_path)
            bldg_centroids.append((footprint_cent.y, footprint_cent.x))
            inps.append((footprint,
                         (footprint_cent.y, footprint_cent.x),
                         image_path,
                         depthmap_path))

        # Download building-wise street-level imagery and depthmap strings:
        pbar = tqdm(total=len(footprints),
                    desc='Obtaining street-level imagery')
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._download_streetlev_image,
                                footprint,
                                footprint_cent,
                                image_path,
                                depthmap_path,
                                save_intermediate_imagery=save_interim_images,
                                save_all_cam_meta=save_all_cam_metadata
                                ): image_path
                for (footprint,
                     footprint_cent,
                     image_path,
                     depthmap_path) in inps
            }
            for future in as_completed(futures):
                image_path = futures[future]
                pbar.update(n=1)
                try:
                    results[image_path] = future.result()
                except KeyError:
                    results[image_path] = None
                    cent = re.search(r'_(.*?)\.jpg', str(image_path))
                    logger.warning('Error downloading image for building '
                                   'located at %s', cent)

        # Get the depthmap and all other required camera metadata:
        metadata = self._process_meta_for_images(street_image_paths,
                                                 bldg_centroids,
                                                 results,
                                                 save_all_cam_metadata)

        return street_image_paths, metadata

    def _download_streetlev_image(self,
                                  footprint: list[list[float, float]],
                                  fpcent: tuple[float, float],
                                  im_path: Path,
                                  depthmap_path: Path,
                                  save_intermediate_imagery: bool = False,
                                  save_all_cam_meta: bool = False
                                  ) -> Optional[Union[
                                      tuple[float, str, tuple[float, float],
                                            tuple[float, float]],
                                      tuple[float, str, tuple[float, float],
                                            tuple[float, float],
                                            tuple[int, int],
                                            float, float, float, float]]]:
        """
        Download a street-level panoramic image and processes it.

        Parameters:
            footprint(list[tuple[float, float]]): Coordinates of the building
                footprint to crop the image.
            fpcent(tuple[float, float]): Latitude and longitude of the query
                point.
            im_name(str): Filename for the output image.
            depthmap_name(str): Filename for the depthmap.
            save_intermediate_imagery(bool): Whether to save intermediate
                images such as the panorama and composite image.
            save_all_cam_meta(bool): Whether to return all camera metadata.

        Returns:
            Optional[Union[
                tuple[float, str, tuple[float, float], tuple[float, float]],
                tuple[float, str, tuple[float, float], tuple[float, float],
                      tuple[int, int], float, float, float, float]]]: If
            successful, returns camera elevation, depthmap, and optionally
            other camera metadata.
        """
        # Initialize filenames for intermediate files
        pano_name, comp_im_name = '', ''

        # Convert image and depthmap paths to string:
        im_name = str(im_path.as_posix())
        pano_dmap_name = str(depthmap_path.as_posix())

        if save_intermediate_imagery:
            im_name_base = im_name.rsplit('.', 1)[0]
            pano_name = f'{im_name_base}_pano.jpg'
            comp_im_name = f'{im_name_base}_composite.jpg'

        # Initialize panorama dictionary
        pano = {'queryLatLon': fpcent,
                'camLatLon': (),
                'id': '',
                'panoSize': (),
                'camHeading': 0,
                'depthMap': 0,
                'depthMapString': '',
                'panoImFile': pano_name,
                'panoDepthStrFile': pano_dmap_name,
                'compositeImFile': comp_im_name
                }

        # Get pano ID. If no pano exists, skip the remaining steps of the
        # function:
        try:
            pano['id'] = self._get_pano_id(fpcent, self.api_key)
        except KeyError:
            logger.info('No street-level imagery found for the building '
                        'located at %.4f, %.4f', fpcent[0], fpcent[1])
            return None

        # Get the metdata for the pano:
        pano = self._get_pano_meta(pano, dmap_outname=pano_dmap_name)

        # Download the panorama image:
        pano = self._download_pano_image(pano, im_name=pano_name)

        # Crop the image based on the building footprint:
        pano = self._get_bldg_image(pano, footprint, im_name=im_name,
                                    save_depthmap=save_intermediate_imagery)

        # If save_interim is True, generate the composite pano with depth
        # map overlaid:
        if save_intermediate_imagery:
            self._get_composite_pano(pano, compim_name=comp_im_name)

        # Return camera elevation, depthmap, and, if requested, other camera
        # metadata:
        if save_all_cam_meta:
            return (pano['camElev'], pano['panoDepthStrFile'],
                    pano['camLatLon'], pano['panoBndAngles'], pano['panoSize'],
                    pano['camHeading'], pano['panoTilt'], pano['panoFOV'],
                    pano['panoRoll'])
        return (pano['camElev'], pano['panoDepthStrFile'], pano['camLatLon'],
                pano['panoBndAngles'])

    @ staticmethod
    def _process_meta_for_images(street_image_paths: list[Path],
                                 bldg_centroids: list[tuple[float, float]],
                                 results: dict[Path,
                                               dict[str,
                                                    Union[float, str, None]]],
                                 save_all_cam_metadata: bool
                                 ) -> dict[Path,
                                           dict[str, Union[tuple[float, float],
                                                           float,
                                                           str]]]:
        """
        Process the downloaded street image data and extract relevant metadata.

        Parameters:
            street_image_paths(List[Path]): List of file paths for the
                downloaded street-level imagery.
            bldg_centroids: list[tuple[float, float]]: List of centroids of
                buildings for which street-level imagery was downloaded.
            results(Dict[Path, Dict[str, Any[float, str, None]]]):
                A dictionary where the keys are image file paths, and the
                values are dictionaries containing metadata about each image,
                such as camera elevation, depth maps locations, and other
                optional camera metadata.
            save_all_cam_metadata(bool): A flag indicating whether to save all
            available camera metadata(e.g., camera latitude/longitude, field
                                       of view, heading, pitch, zoom level).

        Returns:
            dict[Path, dict[str, Union[tuple[float, float], float, str]]]: A
                dictionary where each key is an image file path and the value
                is another dictionary containing metadata values(e.g.,
                'camElev', 'depthMap', 'camLatLon', etc.).
        """
        # Initialize the metadata dictionary and properties dictionary with
        # essential keys:
        metadata = {}
        properties = {
            'bdlgLatLon': (),
            'camElev': None,
            'depthMap': None,
            'camLatLon': None,
            'panoBndAngles': None
        }

        # If all camera metadata should be saved, extend properties dictionary
        # with additional keys:
        if save_all_cam_metadata:
            additional_keys = ['panoSize', 'camHeading',
                               'panoTilt', 'panoFOV', 'panoRoll']
            properties.update({key: [] for key in additional_keys})

        # Populate the metadata dictionary with the properties for each image:
        for image_ind, image_path in enumerate(street_image_paths):
            properties['bdlgLatLon'] = bldg_centroids[image_ind]
            if results[image_path] is not None:
                properties['camElev'] = results[image_path][0]
                properties['depthMap'] = results[image_path][1]
                properties['camLatLon'] = results[image_path][2]
                properties['panoBndAngles'] = results[image_path][3]

                if save_all_cam_metadata:
                    properties['panoSize'] = results[image_path][4]
                    properties['camHeading'] = results[image_path][5]
                    properties['panoTilt'] = results[image_path][6]
                    properties['panoFOV'] = results[image_path][7]
                    properties['panoRoll'] = results[image_path][8]
            else:
                if save_all_cam_metadata:
                    for key in additional_keys:
                        properties[key] = None

            metadata[image_path] = dict(properties)
        return metadata

    @ staticmethod
    def _get_pano_id(latlon: tuple[float, float], api_key: str) -> str:
        """
        Obtain the pano ID for the given latitude and longitude.

        Args:
            latlon(tuple[float, float]): Latitude and longitude.

        Returns:
            str: Pano ID.
        """
        params = {
            'location': f'{latlon[0]}, {latlon[1]}',
            'key': api_key,
            'source': 'outdoor',
        }
        response = requests.get(BASE_API_URL, params=params, timeout=30)
        return response.json()['pano_id']

    @ staticmethod
    def _get_pano_meta(pano: dict[str, Any],
                       dmap_outname: str = '') -> dict[str, Any]:
        """
        Retrieve metadata for the pano.

        Args:
            pano(dict[str, any]): Pano dictionary.
            dmap_outname(str): Filename to save the depth map string.

        Returns:
            Dict[str, Any]: Updated pano dictionary.
        """
        # Get the metadata for a pano image:
        params = {
            'authuser': '0',
            'hl': 'en',
            'gl': 'us',
            'pb': ('!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!'
                   '1e2!2s' + pano['id'] + '!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!'
                   '1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!'
                   '9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!'
                   '1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!'
                   '3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3')
        }

        # Send GET request to API endpoint and retrieve response:
        response = requests.get(PANORAMA_METADATA_URL,
                                params=params,
                                proxies=None,
                                timeout=30)

        # Extract depthmap and other image metadata from response:
        response_content = response.content
        response_json = json.loads(response_content[4:])  # Skip first 4 bytes
        pano['panoZoom'] = 3
        pano['panoFOV'] = ZOOM2FOV_MAPPER[pano['panoZoom']]
        pano['depthMapString'] = response_json[1][0][5][0][5][1][2]
        pano['camLatLon'] = (response_json[1][0][5][0][1][0][2],
                             response_json[1][0][5][0][1][0][3])
        pano['panoSize'] = tuple(
            response_json[1][0][2][3][0][pano['panoZoom']][0])[::-1]
        pano['camHeading'] = response_json[1][0][5][0][1][2][0]
        pano['panoTilt'] = response_json[1][0][5][0][1][2][1]
        pano['panoRoll'] = response_json[1][0][5][0][1][2][2]
        pano['camElev'] = response_json[1][0][5][0][1][1][0]
        # pano['city'] = response_json[1][0][3][2][1][0]

        # If dmap_outname is provided, write depthmap string into a text file:
        if dmap_outname:
            with open(dmap_outname, 'w', encoding='utf-8') as dmapfile:
                dmapfile.write(pano['depthMapString'])

        return pano

    def _download_pano_image(self,
                             pano: dict[str, Any],
                             im_name: str) -> dict[str, Any]:
        """
        Download the pano image composed of tiles.

        Args:
            pano(Dict[str, Any]): Pano dictionary.
            im_name(str): Filename to save the pano image.

        Returns:
            Dict[str, Any]: Updated pano dictionary.
        """
        pano_id = pano['id']
        image_size = pano['panoSize']
        zoom = pano['panoZoom']

        # Calculate tile locations (offsets) and determine corresponding
        # tile URL links:
        baseurl = TILE_URL_TEMPLATE.format(pano_id=pano_id,
                                           zoom=zoom,
                                           x='{x}',
                                           y='{y}')

        urls = []
        offsets = []
        for x_coord in range(int(image_size[0] / TILE_SIZE)):
            for y_coord in range(int(image_size[1] / TILE_SIZE)):
                urls.append(baseurl.format(x=f'{x_coord}', y=f'{y_coord}'))
                offsets.append((x_coord * TILE_SIZE, y_coord * TILE_SIZE))

        tiles = self._download_tiles(urls)

        # Combine the downloaded tiles to get the uncropped pano:
        combined_im = PIL.Image.new('RGB', image_size)

        for (ind, image) in enumerate(tiles):
            combined_im.paste(image, offsets[ind])

        # Save the uncropped pano:
        pano['panoImage'] = combined_im.copy()
        if im_name:
            combined_im.save(im_name)
        pano['panoImFile'] = im_name
        return pano

    def _get_bldg_image(self,
                        pano: dict[str, Any],
                        footprint: list[tuple[float, float]],
                        im_name: str = 'imstreet.jpg',
                        save_depthmap: bool = False
                        ) -> dict[str, Union[np.ndarray,
                                             list[float],
                                             Image.Image,
                                             np.ndarray]]:
        """
        Generate an image and depthmap cropped around a building from a pano.

        Args:
            pano(dict[str, Any]):
                A dictionary containing panorama information, including:
                    - camLatLon: The latitude and longitude of the camera
                        location.
                    - camHeading: The camera heading angle in degrees.
                    - panoSize: The size of the panorama image as
                        [width, height].
                    - panoImage: The panorama image as a PIL Image.
                    - depthMap: (Optional) The depth map associated with the
                        panorama.
            footprint(list[tuple[float, float]]): A list of tuples
                representing the latitude and longitude of the building
                footprint vertices.
            im_name(str, optional): The name of the output image file.
                Defaults to 'imstreet.jpg'.
            save_depthmap(bool, optional): Whether to save the depth map of
                the building. Defaults to False.

        Returns:
            pano(dict[str, Any]):
                The updated panorama dictionary containing additional keys
                including 'panoBndAngles' and 'depthMapBldg'.
        """
        # Calculate the viewing angle values that encompass the building
        # buffered 10 degrees in horizontal direction:
        camera_angles = self._get_view_angles(pano, footprint)
        bnd_angles = np.rint((np.array([round(min(camera_angles), -1) - 10,
                                       round(max(camera_angles), -1) + 10])
                             + 180) / 360 * pano['panoSize'][0])

        bldg_image = pano['panoImage']
        bldg_im_cropped = bldg_image.crop(
            (bnd_angles[0], 0, bnd_angles[1], pano['panoSize'][1]))
        bldg_im_cropped.save(im_name)
        pano['panoBndAngles'] = np.copy(bnd_angles)

        if save_depthmap:
            # Get the depth map for the pano:
            pano_dmap_name = im_name.replace(
                '.' + im_name.split('.')[-1], '') + '_pano_depthmap.jpg'
            pano = self._get_depth_map(
                pano, dmap_imname=pano_dmap_name)
            mask = pano['depthMap']

            # Crop the horizontal parts of the image outside the bndAngles:
            dmap_name = im_name.replace(
                '.' + im_name.split('.')[-1], '') + '_depthmap.jpg'
            mask_cropped = mask.crop(
                (bnd_angles[0], 0, bnd_angles[1], pano['panoSize'][1]))
            pano['depthMapBldg'] = mask_cropped.copy()
            mask_cropped.convert('RGB').save(dmap_name)

        return pano

    @ staticmethod
    def _get_composite_pano(pano: dict[str, Any],
                            compim_name: str = 'panoOverlaid.jpg') -> None:
        """
        Create a composite pano image by overlaying a heat map of depth map.

        Args:
            pano(dict[str, Any]): A dictionary containing the depth map
                (as a PIL image) and the panoramic image(as a PIL image).
            imname(str): The filename for saving the overlaid image. Default
                is 'panoOverlaid.jpg'.

        Returns:
            None
        """
        # Convert the depth map to grayscale and apply a heat map (jet
        # colormap). Convert depth map to grayscale:
        image = np.array(pano['depthMap'].convert('L'))
        cm_jet = mpl.colormaps['jet']  # Access the 'jet' colormap

        # Apply the colormap to the grayscale image:
        im_colored = cm_jet(image)

        # Scale the colormap output to [0, 255]:
        im_colored = np.uint8(im_colored * 255)
        im_mask = Image.fromarray(im_colored).convert(
            'RGB')  # Convert to RGB format

        # Overlay the heat map on the original panoramic image
        # Original pano image (assumed to be in RGB format):
        im_pano = pano['panoImage']

        # Blend the heatmap and pano with equal weight:
        im_overlaid = Image.blend(im_mask, im_pano, 0.5)

        # Save the composite image:
        im_overlaid.save(compim_name)

    def _get_view_angles(self,
                         pano: dict[str, Any],
                         footprint: list[tuple[float, float]]) -> list[float]:
        """
        Calculate viewing angles of each footprint vertex from camera location.

        This function converts the geographic coordinates(latitude and
        longitude) of a building footprint into Cartesian coordinates relative
        to the camera's location, and then calculates the viewing angles of the
        building's vertices based on the camera's heading.

        Args:
            pano(dict[str, any]): A dictionary containing information about
                the panorama, including:
                - camLatLon(tuple[float, float]): Latitude and longitude of
                    the camera location.
                - camHeading(float): The camera's heading angle in degrees.
            footprint(list[tuple[float, float]]): A list of tuples
                representing the latitude and longitude coordinates of the
                building footprint.

        Returns:
            list[float]: A list of viewing angles for each vertex of the
                footprint relative to the camera's heading.
        """
        # Project the coordinates of the footprint to Cartesian with the
        # approximate camera location set as the origin:
        (lat0, lon0) = pano['camLatLon']
        xy_footprint = []
        for lat1, lon1 in footprint:
            xcoord = (lon1 - lon0) * CIRCUM_EARTH_FT * \
                math.cos((lat0 + lat1)*math.pi/360)/360
            ycoord = (lat1 - lat0) * CIRCUM_EARTH_FT/360
            xy_footprint.append((xcoord, ycoord))

        # Calculate the theoretical viewing angle for each footprint vertex
        # with respect to the camera heading angle:
        return [self._get_angle_from_heading(coord, pano['camHeading'])
                for coord in xy_footprint]

    @ staticmethod
    def _download_tiles(urls: list[str]) -> list[PIL.Image.Image]:
        """
        Download image tiles from the provided URLs with retry strategy.

        Args:
            urls(List[str]): List of tile URLs.

        Returns:
            List[PIL.Image.Image]: List of downloaded tile images.
        """
        session = requests.Session()
        session.mount("https://",
                      HTTPAdapter(max_retries=REQUESTS_RETRY_STRATEGY))
        tiles = []
        for url in urls:
            response = session.get(url)
            tiles.append(PIL.Image.open(BytesIO(response.content)))

        return tiles

    def _get_depth_map(self,
                       pano: dict[str, Any],
                       dmap_imname: str = '') -> dict[str, Any]:
        """
        Compute and process the depth map for a panoramic image.

        Args:
            pano(dict[str, Any]): A dictionary containing panoramic image
                data, including the depth map string and image size.
            dmap_imname(str): The file name to use if saving the depth map
                image. Default is '', i.e., depth map image is not saved.

        Returns:
            dict[str, Any]: The input pano dictionary with the processed
                depth map and file information.
        """
        # Decode the depth map string
        depth_map_data = self._parse_dmap_str(pano['depthMapString'])

        # Parse the first bytes to get the data headers
        header = self._parse_dmap_header(depth_map_data)

        # Parse the remaining bytes into planes of float values
        data = self._parse_dmap_planes(header, depth_map_data)

        # Compute the position and depth values of pixels
        depth_map = self._compute_dmap(header, data["indices"], data["planes"])

        # Process float 1D array into a 2D array with pixel values ranging from
        # 0 to 255:
        dmap_array = depth_map["depthMap"]
        dmap_array[np.where(dmap_array == np.max(dmap_array))] = 255
        if np.min(dmap_array) < 0:
            dmap_array[np.where(dmap_array < 0)] = 0
        dmap_array = dmap_array.reshape(
            (depth_map["height"], depth_map["width"]))

        # Flip the 2D array to align it with the panoramic image pixels
        dmap_array = np.fliplr(dmap_array)

        # Convert the 2D array into an image and resize it to match the size
        # of the pano:
        im_dmap = Image.fromarray(np.uint8(dmap_array))
        im_dmap = im_dmap.resize(pano['panoSize'])
        pano['depthMap'] = im_dmap.copy()

        if dmap_imname:
            # Convert the float values to grayscale for saving
            im_dmap_save = im_dmap.convert('L')

            # Save the depth map image
            im_dmap_save.save(dmap_imname)

        pano['depthImFile'] = dmap_imname

        return pano

    def _get_angle_from_heading(self,
                                coord: tuple[float, float],
                                heading: float) -> float:
        """
        Calculate the viewing angle of a coordinate relative to camera heading.

        Args:
            coord(Tuple[float, float]): Cartesian coordinates.
            heading(float): Camera heading angle in degrees.

        Returns:
            float: Calculated viewing angle.
        """
        # Determine the cartesian coordinates of a point along the heading that
        # is 100 ft away from the origin:
        x_0 = 100 * math.sin(math.radians(heading))
        y_0 = 100 * math.cos(math.radians(heading))

        # Calculate the clockwise viewing angle for the coord with respect to
        # the heading:
        ang = 360 - self._get_3pt_angle((x_0, y_0), (0, 0), coord)

        # Return viewing angles such that anything to the left of the vertical
        # camera axis is negative and counterclockwise angle measurement:
        return ang if ang <= 180 else ang - 360

    @ staticmethod
    def _parse_dmap_str(b64_string: str) -> np.ndarray:
        """
        Parse a base64-encoded depth map & return decoded data as numpy array.

        Args:
            b64_string(str): Base64-encoded depth map string.

        Returns:
            np.ndarray: A numpy array of decoded byte data.
        """
        # Ensure correct padding (length needs to be divisible by 4):
        b64_string += '=' * ((4 - len(b64_string) % 4) % 4)

        # Convert the URL-safe format to the regular format:
        data = b64_string.replace('-', '+').replace('_', '/')

        # Decode the base64 string into bytes:
        data = base64.b64decode(data)

        return np.array(data)

    def _parse_dmap_header(self, depth_map: np.ndarray) -> dict[str, int]:
        """
        Parse the header information from the depth map.

        Args:
            depth_map(np.ndarray): Numpy array containing the depth map data.

        Returns:
            dict[str, int]: Dictionary containing parsed header information:
                - 'headerSize': The size of the header.
                - 'numberOfPlanes': The number of planes in the depth map.
                - 'width': The width of the depth map in pixels.
                - 'height': The height of the depth map in pixels.
                - 'offset': The byte offset to the plane and index data.
        """
        return {'headerSize': depth_map[0],
                'numberOfPlanes': self._get_uint16(depth_map, 1),
                'width': self._get_uint16(depth_map, 3),
                'height': self._get_uint16(depth_map, 5),
                'offset': self._get_uint16(depth_map, 7),
                }

    def _parse_dmap_planes(self,
                           header: dict[str, int],
                           depth_map: np.ndarray) -> dict[str, list]:
        """
        Parse the plane information and indices from the depth map.

        Args:
            header(dict[str, int]): Parsed header information containing
                width, height, number of planes, and offset.
            depth_map(np.ndarray): Numpy array containing the depth map data.

        Returns:
            dict[str, List]: A dictionary containing:
                - planes: A list of planes where each plane is a dictionary
                    with:
                    - normal(List[float]): The normal vector of the plane.
                    - dist(float): The distance from the origin to the plane.
                - 'indices': A list of plane indices for each pixel in the
                    depth map.
        """
        # Parse the plane indices:
        indices = []
        for i in range(header['width'] * header['height']):
            indices.append(depth_map[header['offset'] + i])

        # Parse the plane information:
        planes = []
        normal = [0.0, 0.0, 0.0]
        for i in range(header['numberOfPlanes']):
            byte_offset = header['offset'] + \
                header['width'] * header['height'] + i * 4 * 4
            normal[0] = self._get_float32(depth_map, byte_offset)
            normal[1] = self._get_float32(depth_map, byte_offset + 4)
            normal[2] = self._get_float32(depth_map, byte_offset + 8)
            dist = self._get_float32(depth_map, byte_offset + 12)
            planes.append({'n': normal.copy(), 'd': dist})

        return {'planes': planes, 'indices': indices}

    @ staticmethod
    def _compute_dmap(header: dict[str, int],
                      indices: list[int],
                      planes: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compute depth map using provided planes and indices in the image.

        This method generates a depth map by casting rays from the camera
        through each pixel, determining where the rays intersect with the
        detected planes. The depth(distance to the plane) is calculated using
        the dot product between the ray direction and the plane's normal
        vector.

        Args:
            header(Dict[str, int]): Parsed header containing image dimensions,
                including:
                - "width" (int): Width of the image in pixels.
                - "height" (int): Height of the image in pixels.
            indices(List[int]): A flat list of plane indices for each pixel
                in the image. Each index corresponds to a detected plane in the
                `planes` list.
            planes(List[Dict[str, Any]]): A list of plane information. Each
                plane is represented by a dictionary that contains:
                - "n" (List[float]): Normal vector of the plane(3D direction).
                - "d" (Callable[[float], float]): A function that computes the
                    plane's distance based on the dot product with the
                    ray direction vector.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "width" (int): The width of the image in pixels.
                - "height" (int): The height of the image in pixels.
                - "depthMap" (np.ndarray): A flattened array representing the
                    depth map where each value is the computed depth for the
                    corresponding pixel.
        """
        ray_dir = [0, 0, 0]
        width = header["width"]
        height = header["height"]

        depth_map = np.empty(width * height)

        for y_coord in range(height):
            theta = (height - y_coord - 0.5)/height * np.pi
            for x_coord in range(width):
                plane_idx = indices[y_coord * width + x_coord]

                phi = (width - x_coord - 0.5)/width * 2 * np.pi + np.pi / 2
                ray_dir[0] = np.sin(theta) * np.cos(phi)
                ray_dir[1] = np.sin(theta) * np.sin(phi)
                ray_dir[2] = np.cos(theta)

                if plane_idx > 0:
                    plane = planes[plane_idx]
                    depth = np.abs(plane["d"](ray_dir[0] * plane["n"][0] +
                                              ray_dir[1] * plane["n"][1] +
                                              ray_dir[2] * plane["n"][2])
                                   )
                    depth_map[y_coord * width + (width - x_coord - 1)] = depth
                else:
                    depth_map[y_coord * width + (width - x_coord - 1)
                              ] = 9999999999999999999.0
        return {"width": width, "height": height, "depthMap": depth_map}

    @ staticmethod
    def _get_3pt_angle(pt1: tuple[float, float],
                       pt2: tuple[float, float],
                       pt3: tuple[float, float]) -> float:
        """
        Calculate the angle formed by three points.

        Args:
            pt1(Tuple[float, float]): First point.
            pt2(Tuple[float, float]): Vertex point.
            pt3(Tuple[float, float]): Third point.

        Returns:
            float: Angle in degrees.
        """
        ang = math.degrees(math.atan2(pt3[1] - pt2[1], pt3[0] - pt2[0]) -
                           math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0]))
        return ang + 360 if ang < 0 else ang

    def _get_uint16(self, arr: list[int], ind: int) -> int:
        """
        Combine two bytes from the array into a 16-bit unsigned integer.

        Args:
            arr(list[int]): Array of byte values.
            ind(int): Starting index of the two bytes to combine.

        Returns:
            int: 16-bit unsigned integer formed from the two bytes.
        """
        int_inp1 = arr[ind]
        int_inp2 = arr[ind + 1]
        return int(self._get_bin(int_inp2) + self._get_bin(int_inp1), 2)

    @ staticmethod
    def _get_bin(int_inp: int) -> str:
        """
        Convert an integer to an 8-bit binary string.

        Args:
            a(int): The integer to be converted.

        Returns:
            str: The 8-bit binary string representation of the integer.
        """
        binary_str_int = bin(int_inp)[2:]
        return '0' * (8 - len(binary_str_int)) + binary_str_int

    def _get_float32(self, arr: list[int], ind: int) -> float:
        """
        Convert 4 bytes from input array at given index into a 32-bit float.

        Args:
            arr(list[int]): Array of byte values.
            ind(int): Starting index of the 4 bytes to convert.

        Returns:
            float: The 32-bit floating-point number.
        """
        return self._bin_to_float(''.join(
            self._get_bin(i) for i in arr[ind: ind + 4][::-1]))

    @ staticmethod
    def _bin_to_float(binary: str) -> float:
        """
        Convert a binary string to a 32-bit float.

        Args:
            binary(str): A 32-bit binary string.

        Returns:
            float: The 32-bit floating-point number corresponding to the
                binary string.
        """
        return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]
