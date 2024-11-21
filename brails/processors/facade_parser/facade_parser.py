"""Class object to parse building facade images to get metric attributes."""

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
# 11-21-2024


import os
import math
import torch
import cv2
import numpy as np
import base64
import struct
import torchvision.transforms as T
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression

from brails.types.image_set import ImageSet
from brails.processors.image_processor import ImageProcessor
from brails.utils import InputValidator


class FacadeParser(ImageProcessor):
    """
    A class to parse facade images of buildings and predict metric attributes.

    Attributes:
        cam_elevs (list[float]):
            Camera elevation angles.
        depthmaps (List[Tuple[str, Tuple[int, int], List[float]]]):
            Depth maps
            for building images.
        footprints (list[list[float, float]]):
            Building footprints.
        model_path (str):
            Path to the trained model.
        street_images (list[str]):
            Paths to the street images.
        predictions (pd.DataFrame):
            DataFrame to store predictions.
    """

    def __init__(self, input_data: dict):
        """
        Initialize the class with input data.

        Args:
            input_data (dict):
                A dictionary containing the following required keys and values:
                    - footprints (list):
                        A two-dimensional list for the building footprints in
                        [[lon1, lat1], [lon2, lat2], ..., [lonN, latN]] format.
                    - modelPath (str, optional):
                        Path to the model file. Defaults to
                        'tmp/models/facadeParser.pth'.
                    - saveSegmentedImages (bool, optional):
                        Whether to save segmented images. Defaults to False.

        Raises:
            TypeError:
                If input_data is not a dictionary.
            KeyError:
                If input_data is missing the required keys.
            ValueError:
                If any input_data value is invalid.
        """
        # Validate input data:
        self.footprints, self.model_path, self.save_segimages = \
            self._validate_input_data(input_data)

        # Initialize additional attributes:
        self.street_images = []
        self.depthmaps = []
        self.cam_elevs = []

    @staticmethod
    def _validate_input_data(input_data: dict):
        """
        Validate the input data and ensure it is correctly formatted.

        Args:
            input_data (dict):
                A dictionary containing input keys and values.

        Returns:
            tuple:
                Validated values for 'footprints', 'modelPath', and
                'saveSegmentedImages'.

        Raises:
            TypeError:
                If input_data is not a dictionary.
            KeyError:
                If input_data is missing the required keys.
            ValueError:
                If any input_data value is invalid.
        """
        # Validate input_data variable type:
        if not isinstance(input_data, dict):
            raise TypeError("input_data must be a dictionary.")

        # Validate footprints:
        footprints = input_data.get('footprints')
        if footprints is None:
            raise KeyError("The 'footprints' key is missing from input_data.")
        coords_check, output_msg = InputValidator.validate_coordinates(
            footprints)
        if not coords_check:
            raise ValueError(output_msg)

        # Validate modelPath:
        model_path = input_data.get('modelPath', 'tmp/models/facadeParser.pth')
        if not isinstance(model_path, str):
            raise ValueError("The 'modelPath' value must be a string.")

        # Validate saveSegmentedImages:
        save_segimages = input_data.get('saveSegmentedImages', False)
        if not isinstance(save_segimages, bool):
            raise ValueError(
                "The 'saveSegmentedImages' value must be a boolean.")

        return footprints, model_path, save_segimages

    def predict(self, images: ImageSet) -> dict:
        """
        Predict building metric attributes from street-level imagery.

        Args:
            images (ImageSet):
                An object that contains camera elevations, depth maps,
                footprints, and street images. It should have the following
                attributes:
                    - images: A dictionary containing image data with keys as
                    image identifiers.
                    - dir_path: The directory path where the images are stored.
                    - properties: A dictionary of properties for each image,
                    including 'depthMap' and 'camElev'.

        Returns:
            dict:
                A dictionary containing predictions for each image. Each entry
                includes:
                    - 'roofEaveHeight': The predicted roof eave height in feet,
                      rounded to one decimal place.
                    - 'buildingHeight': The predicted building height in feet,
                      rounded to the nearest integer.
                    - 'roofPitch': The predicted roof pitch, rounded to two
                      decimal places.

        Raises:
            FileNotFoundError:
                If the model path does not exist.
            ValueError:
                If the required data for predictions is missing or invalid.

        Notes:
            - Ensure that the input `images` contains valid image paths and
              properties.
            - The prediction process may be computationally expensive and
              time-consuming.
            - The method utilizes the GPU if available, and the model is
              unloaded from the GPU after inference.
        """
        for _, image in images.images.items():
            self.street_images.append(
                os.path.join(images.dir_path, image.filename))
            self.depthmaps.append(image.properties['depthMap'])
            self.cam_elevs.append(image.properties['camElev'])
        image_keys = list(images.images.keys())

        def compute_roofrun(footprint: list[list[float, float]]) -> float:
            """
            Compute the roof run for a given footprint.

            The roof run is determined by finding minimum area rectangle that
            fits around the footprint. The function first converts latitude and
            longitude values to Cartesian coordinates for better resolution,
            then calculates the slope and length of each line segment
            comprising the footprint polygon. It clusters these segments based
            on their slope values and identifies the principal directions of
            the footprint. Finally, it computes the bounding box and returns
            the minimum side length as the roof run.

            Args:
                footprint (list[list[float, float]):
                    A list of lists representing the footprint coordinates,
                    where each inner list contains [longitude, latitude]
                    values.

            Returns:
                float: The minimum side length of the bounding box,
                       representing the roof run.
            """
            # Find the mimumum area rectangle that fits around the footprint
            # What failed: 1) PCA, 2) Rotated minimum area rectangle
            # 3) Modified minimum area rectangle
            # Current implementation: Tight-fit bounding box

            # Convert lat/lon values to Cartesian coordinates for better
            # coordinate resolution:

            # Flip the footprint coordinates so that they are in latitude and
            # longitude format:
            footprint = np.fliplr(np.squeeze(np.array(footprint)))

            xy_fp = np.zeros((len(footprint), 2))
            for k in range(1, len(footprint)):
                lon1 = footprint[0, 0]
                lon2 = footprint[k, 0]
                lat1 = footprint[0, 1]
                lat2 = footprint[k, 1]

                xy_fp[k, 0] = (lon1-lon2)*40075000*3.28084 *\
                    math.cos((lat1+lat2)*math.pi/360)/360
                xy_fp[k, 1] = (lat1-lat2)*40075000*3.28084/360

            # Calculate the slope and length of each line segment comprising
            # the footprint polygon:
            slopeSeg = np.diff(xy_fp[:, 1])/np.diff(xy_fp[:, 0])

            segments = np.arange(len(slopeSeg))
            lengthSeg = np.zeros(len(slopeSeg))
            for k in range(len(xy_fp)-1):
                p1 = np.array([xy_fp[k, 0], xy_fp[k, 1]])
                p2 = np.array([xy_fp[k+1, 0], xy_fp[k+1, 1]])
                lengthSeg[k] = np.linalg.norm(p2-p1)

            # Cluster the line segments based on their slope values:
            slopeClusters = []
            totalLengthClusters = []
            segmentsClusters = []
            while slopeSeg.size > 1:
                ind = np.argwhere(
                    abs((slopeSeg-slopeSeg[0])/slopeSeg[0]) < 0.3).squeeze()
                slopeClusters.append(np.mean(slopeSeg[ind]))
                totalLengthClusters.append(np.sum(lengthSeg[ind]))
                segmentsClusters.append(segments[ind])

                indNot = np.argwhere(
                    abs((slopeSeg-slopeSeg[0])/slopeSeg[0]) >= 0.3).squeeze()
                slopeSeg = slopeSeg[indNot]
                lengthSeg = lengthSeg[indNot]
                segments = segments[indNot]

            if slopeSeg.size == 1:
                slopeClusters.append(slopeSeg)
                totalLengthClusters.append(lengthSeg)
                segmentsClusters.append(segments)

            # Mark the two clusters with the highest total segment lengths as
            # the principal directions of the footprint
            principalDirSlopes = []
            principalDirSegments = []
            counter = 0
            for ind in np.flip(np.argsort(totalLengthClusters)):
                if type(segmentsClusters[ind]) is np.ndarray:
                    principalDirSlopes.append(slopeClusters[ind])
                    principalDirSegments.append(segmentsClusters[ind])
                    counter += 1
                if counter == 2:
                    break

            x_fp = xy_fp[:, 0]
            y_fp = xy_fp[:, 1]
            slopeSeg = np.diff(xy_fp[:, 1])/np.diff(xy_fp[:, 0])

            bndLines = np.zeros((4, 4))
            for cno, cluster in enumerate(principalDirSegments):
                xp = np.zeros((2*len(cluster)))
                yp = np.zeros((2*len(cluster)))
                for idx, segment in enumerate(cluster):
                    angle = math.pi/2 - math.atan(slopeSeg[segment])
                    x = x_fp[segment:segment+2]
                    y = y_fp[segment:segment+2]
                    xp[2*idx:2*idx+2] = x*math.cos(angle) - y*math.sin(angle)
                    yp[2*idx:2*idx+2] = x*math.sin(angle) + y*math.cos(angle)

                minLineIdx = int(np.argmin(xp)/2)
                maxLineIdx = int(np.argmax(xp)/2)

                minLineIdx = cluster[int(np.argmin(xp)/2)]
                maxLineIdx = cluster[int(np.argmax(xp)/2)]

                bndLines[2*cno:2*cno+2, :] = np.array([[x_fp[minLineIdx],
                                                       y_fp[minLineIdx],
                                                       x_fp[minLineIdx+1],
                                                       y_fp[minLineIdx+1]
                                                        ],
                                                      [x_fp[maxLineIdx],
                                                       y_fp[maxLineIdx],
                                                       x_fp[maxLineIdx+1],
                                                       y_fp[maxLineIdx+1]
                                                       ]
                                                       ])
            bbox = np.zeros((5, 2))
            counter = 0
            for k in range(2):
                line1 = bndLines[k, :]
                x1 = line1[0]
                x2 = line1[2]
                y1 = line1[1]
                y2 = line1[3]
                for m in range(2, 4):
                    line2 = bndLines[m, :]
                    x3 = line2[0]
                    x4 = line2[2]
                    y3 = line2[1]
                    y4 = line2[3]
                    d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                    bbox[counter, :] = [((x1*y2-y1*x2)*(x3-x4) -
                                         (x1-x2)*(x3*y4-y3*x4))/d,
                                        ((x1*y2-y1*x2)*(y3-y4) -
                                         (y1-y2)*(x3*y4-y3*x4))/d]
                    counter += 1
            bbox[4, :] = bbox[0, :]
            bbox[2:4, :] = np.flipud(bbox[2:4, :])

            sideLengths = np.linalg.norm(np.diff(bbox, axis=0), axis=1)
            roof_run = min(sideLengths)

            return roof_run

        def install_default_model(model_path='tmp/models/facadeParser.pth'):
            if model_path == "tmp/models/facadeParser.pth":
                os.makedirs('tmp/models', exist_ok=True)
                model_path = 'tmp/models/facadeParser.pth'

                if not os.path.isfile(model_path):
                    print('Loading default facade parser model file to '
                          'tmp/models folder...')
                    torch.hub.download_url_to_file('https://zenodo.org/record'
                                                   '/5809365/files'
                                                   '/facadeParser.pth',
                                                   model_path,
                                                   progress=False)
                    print('Default facade parser model loaded')
                else:
                    print(
                        f"Default facade parser model at {model_path} loaded")
            else:
                print('Inferences will be performed using the custom model at '
                      f'{model_path}')

        def gen_bbox(contour: np.ndarray) -> Polygon:
            """
            Generate a bounding box from a given mask contour.

            The bounding box is calculated by finding the minimum and maximum x
            and y coordinates of the contour points, then creating a
            rectangular polygon that encompasses those points.

            Args__
                contour (np.ndarray): A 2D NumPy array of shape (n, 2)
                    where n is the number of points in the contour.
                    Each row represents a point with x and y coordinates.

            Returns__
                bbox_poly (Polygon): A Shapely Polygon representing the
                    bounding box of the roof contour.
            """
            minx = min(contour[:, 0])
            maxx = max(contour[:, 0])
            miny = min(contour[:, 1])
            maxy = max(contour[:, 1])
            bbox_poly = Polygon([(minx, miny),
                                 (minx, maxy),
                                 (maxx, maxy),
                                 (maxx, miny)])
            return bbox_poly

        def gen_bboxR0(roof_bbox_poly: Polygon,
                       facade_bbox_poly: Polygon) -> Polygon:
            """
            Generate a bounding box that connects roof & facade bounding boxes.

            This function calculates the bottom edges of the roof and facade
            bounding boxes and constructs a new polygon extending from the
            bottom of the facade bounding box to the bottom of the roof
            bounding box.

            Args:
                roof_bbox_poly (Polygon): A Shapely Polygon representing the
                    bounding box of the roof.
                facade_bbox_poly (Polygon): A Shapely Polygon representing the
                    bounding box of the facade.

            Returns:
                r0_bbox_poly (Polygon): A Shapely Polygon extending from the
                    bottom of the facade bounding box to the bottom of the roof
                    bounding box.
            """
            # Extract x and y coordinates of the roof contour:
            x_roof, y_roof = roof_bbox_poly.exterior.xy
            x_roof = np.array(x_roof).astype(int)
            y_roof = np.array(y_roof).astype(int)

            # Locate the bottom y-coordinate and the horizontal extent of the
            # roof contour:
            y_roof_bottom = max(y_roof)
            x_roof_min = min(x_roof)
            x_roof_max = max(x_roof)

            # ind = np.where(y_roof == y_roof_bottom)
            # x_roof_bottom = np.unique(x_roof[ind])
            # y_roof_bottom = np.tile(y_roof_bottom, len(x_roof_bottom))

            x_facade, y_facade = facade_bbox_poly.exterior.xy
            x_facade = np.array(x_facade).astype(int)
            y_facade = np.array(y_facade).astype(int)

            y_facade_bottom = max(y_facade)

            # ind = np.where(y_facade == y_facade_bottom)
            # x_facade_bottom = x_roof_bottom
            # y_facade_bottom = np.tile(y_facade_bottom, len(x_facade_bottom))
            x_facade_min = min(x_roof)
            x_facade_max = max(x_roof)

            x_min = min(x_roof_min, x_facade_min)
            x_max = max(x_roof_max, x_facade_max)

            # r0_bbox_poly = Polygon([(x_facade_bottom[0], y_facade_bottom[0]),
            #                         (x_facade_bottom[1], y_facade_bottom[1]),
            #                         (x_roof_bottom[1], y_roof_bottom[1]),
            #                         (x_roof_bottom[0], y_roof_bottom[0])])
            r0_bbox_poly = Polygon([(x_min, y_facade_bottom),
                                    (x_max, y_facade_bottom),
                                    (x_min, y_roof_bottom),
                                    (x_max, y_roof_bottom)]
                                   )

            return r0_bbox_poly

        def decode_segmap(image: np.ndarray, nc: int = 5) -> np.ndarray:
            """
            Decode a segmentation map into an RGB image.

            This function takes a label-encoded segmentation map and converts
            it into an RGB image using predefined colors for each label.

            Args:
                image (np.ndarray): A 2D NumPy array representing the
                    label-encoded segmentation map. Each pixel value
                    corresponds to a label.
                nc (int, optional): The number of classes (labels) in the
                    segmentation. Default is 5 for buildings and 0=background,
                    1=roof, 2=facade, 3=window, and 4=door

            Returns:
                np.ndarray: A 3D NumPy array representing the RGB image, with
                    the same height and width as the input image, and 3
                    channels for R, G, and B.
            """
            label_colors = np.array([(0, 0, 0),
                                     (255, 0, 0),
                                     (255, 165, 0),
                                     (0, 0, 255),
                                     (175, 238, 238)])

            r = np.zeros_like(image).astype(np.uint8)
            g = np.zeros_like(image).astype(np.uint8)
            b = np.zeros_like(image).astype(np.uint8)

            for label in range(0, nc):
                idx = image == label
                r[idx] = label_colors[label, 0]
                g[idx] = label_colors[label, 1]
                b[idx] = label_colors[label, 2]

            rgb = np.stack([r, g, b], axis=2)
            return rgb

        def get_bin(a):
            ba = bin(a)[2:]
            return "0"*(8 - len(ba)) + ba

        def getUInt16(arr, ind):
            a = arr[ind]
            b = arr[ind + 1]
            return int(get_bin(b) + get_bin(a), 2)

        def getFloat32(arr, ind):
            return bin_to_float("".join(get_bin(i) for i in
                                        arr[ind: ind + 4][::-1]))

        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        def parse_dmap_str(b64_string):
            # Ensure correct padding (The length of string needs to be
            # divisible by 4):
            b64_string += "="*((4 - len(b64_string) % 4) % 4)

            # Convert the URL safe format to regular format:
            data = b64_string.replace("-", "+").replace("_", "/")

            # Decode the string:
            data = base64.b64decode(data)

            return np.array([d for d in data])

        def parse_dmap_header(depthMap):
            return {
                "headerSize": depthMap[0],
                "numberOfPlanes": getUInt16(depthMap, 1),
                "width": getUInt16(depthMap, 3),
                "height": getUInt16(depthMap, 5),
                "offset": getUInt16(depthMap, 7),
            }

        def parse_dmap_planes(header, depthMap):
            indices = []
            planes = []
            n = [0, 0, 0]

            for i in range(header["width"] * header["height"]):
                indices.append(depthMap[header["offset"] + i])

            for i in range(header["numberOfPlanes"]):
                byteOffset = header["offset"] + \
                    header["width"]*header["height"] + i*4*4
                n = [0, 0, 0]
                n[0] = getFloat32(depthMap, byteOffset)
                n[1] = getFloat32(depthMap, byteOffset + 4)
                n[2] = getFloat32(depthMap, byteOffset + 8)
                d = getFloat32(depthMap, byteOffset + 12)
                planes.append({"n": n, "d": d})

            return {"planes": planes, "indices": indices}

        def compute_dmap(header, indices, planes):
            v = [0, 0, 0]
            w = header["width"]
            h = header["height"]

            depthMap = np.empty(w * h)

            sin_theta = np.empty(h)
            cos_theta = np.empty(h)
            sin_phi = np.empty(w)
            cos_phi = np.empty(w)

            for y in range(h):
                theta = (h - y - 0.5)/h*np.pi
                sin_theta[y] = np.sin(theta)
                cos_theta[y] = np.cos(theta)

            for x in range(w):
                phi = (w - x - 0.5)/w*2*np.pi + np.pi/2
                sin_phi[x] = np.sin(phi)
                cos_phi[x] = np.cos(phi)

            for y in range(h):
                for x in range(w):
                    planeIdx = indices[y*w + x]

                    v[0] = sin_theta[y]*cos_phi[x]
                    v[1] = sin_theta[y]*sin_phi[x]
                    v[2] = cos_theta[y]

                    if planeIdx > 0:
                        plane = planes[planeIdx]
                        t = np.abs(
                            plane["d"]
                            / (
                                v[0]*plane["n"][0]
                                + v[1]*plane["n"][1]
                                + v[2]*plane["n"][2]
                            )
                        )
                        depthMap[y*w + (w - x - 1)] = t
                    else:
                        depthMap[y*w + (w - x - 1)] = 9999999999999999999.0
            return {"width": w, "height": h, "depthMap": depthMap}

        def get_depth_map(depthfile, imsize, bndangles):
            # Decode depth map string:
            with open(depthfile, 'r') as fout:
                depthMapStr = fout.read()
            depthMapData = parse_dmap_str(depthMapStr)

            # Parse first bytes to get the data headers:
            header = parse_dmap_header(depthMapData)

            # Parse remaining bytes into planes of float values:
            data = parse_dmap_planes(header, depthMapData)

            # Compute position and depth values of pixels:
            depthMap = compute_dmap(header, data["indices"], data["planes"])

            # Process float 1D array into integer 2D array with pixel values
            # ranging from 0 to 255:
            im = depthMap["depthMap"]
            im[np.where(im == max(im))[0]] = 255
            if min(im) < 0:
                im[np.where(im < 0)[0]] = 0
            im = im.reshape((depthMap["height"], depthMap["width"]))

            # Flip the 2D array to have it line up with pano image pixels:
            im = np.fliplr(im)

            # Read the 2D array into an image and resize this image to match
            # the size of pano:
            imPanoDmap = Image.fromarray(im)
            imPanoDmap = imPanoDmap.resize(imsize)

            # Crop the depthmap such that it includes the building of interest
            # only:
            imBldgDmap = imPanoDmap.crop(
                (bndangles[0], 0, bndangles[1], imsize[1]))
            return imBldgDmap

        # Set the computing environment
        if torch.cuda.is_available():
            dev = 'cuda'
        else:
            dev = 'cpu'

        # Load the trained model and set the model to evaluate mode
        print('\nDetermining the heights and roof pitch for each building...')
        install_default_model(self.model_path)
        model = torch.load(self.model_path, map_location=torch.device(dev))
        model.eval()

        # Create the output dictionary:
        predictions = {}
        for poly_index, footprint in tqdm(enumerate(self.footprints)):
            predictions[[image_keys[poly_index]]] = {'roofEaveHeight': None,
                                                     'buildingHeight': None,
                                                     'roofPitch': None}

            # Check if the footprint is a polygon else move onto the next item
            # in footprints:
            if (footprint is None or
                len(footprint) < 3 or
                    footprint[0] != footprint[-1]):
                continue

            # Run building image through the segmentation model:

            image_path = self.street_images[poly_index]
            if os.path.isfile(image_path):
                img = Image.open(image_path)
            else:
                continue

            imsize = img.size

            trf = T.Compose([T.Resize(round(1000/max(imsize)*min(imsize))),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

            inp = trf(img).unsqueeze(0).to(dev)
            scores = model.to(dev)(inp)['out']
            predraw = torch.argmax(
                scores.squeeze(), dim=0).detach().cpu().numpy()
            pred = np.array(Image.fromarray(np.uint8(predraw)).resize(imsize))

            # Extract component masks
            maskRoof = (pred == 1).astype(np.uint8)
            maskFacade = (pred == 2).astype(np.uint8)
            maskWin = (pred == 3).astype(np.uint8)
            # maskDoor = (pred==4).astype(np.uint8)

            # Open and close masks
            kernel = np.ones((10, 10), np.uint8)
            openedFacadeMask = cv2.morphologyEx(
                maskFacade, cv2.MORPH_OPEN, kernel)
            maskFacade = cv2.morphologyEx(
                openedFacadeMask, cv2.MORPH_CLOSE, kernel)
            openedWinMask = cv2.morphologyEx(maskWin, cv2.MORPH_OPEN, kernel)
            maskWin = cv2.morphologyEx(openedWinMask, cv2.MORPH_CLOSE, kernel)

            # Find roof contours
            contours, _ = cv2.findContours(maskRoof, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                roofContour = max(contours, key=cv2.contourArea).squeeze()

                if roofContour.ndim == 2:
                    # Find the mimumum area rectangle that fits around the
                    # primary roof contour:
                    roofMinRect = cv2.minAreaRect(roofContour)
                    roofMinRect = cv2.boxPoints(roofMinRect)
                    roofMinRect = np.int0(roofMinRect)
                    roofMinRectPoly = Polygon([(roofMinRect[0, 0],
                                                roofMinRect[0, 1]),
                                               (roofMinRect[1, 0],
                                                roofMinRect[1, 1]),
                                               (roofMinRect[2, 0],
                                                roofMinRect[2, 1]),
                                               (roofMinRect[3, 0],
                                                roofMinRect[3, 1])
                                               ])
                    x, y = roofMinRectPoly.exterior.xy

                    roofBBoxPoly = gen_bbox(roofContour)
                    x, y = roofBBoxPoly.exterior.xy
                    roofPixHeight = max(y)-min(y)
                else:
                    roofPixHeight = 0
            else:
                roofPixHeight = 0

            # Find facade contours
            contours, _ = cv2.findContours(maskFacade, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                facadeContour = max(contours, key=cv2.contourArea).squeeze()
            else:
                continue

            facadeBBoxPoly = gen_bbox(facadeContour)
            x, y = facadeBBoxPoly.exterior.xy

            R0BBoxPoly = gen_bboxR0(roofBBoxPoly, facadeBBoxPoly)
            x, y = R0BBoxPoly.exterior.xy
            R0PixHeight = max(y)-min(y)
            R1PixHeight = R0PixHeight + roofPixHeight

            # Get the raw depthmap for the building and crop it so that it
            # covers just the segmentation mask of the bbox for the building
            # facade:
            (depthfile, imsize, bndangles) = self.depthmaps[poly_index]
            depthmap = get_depth_map(depthfile, imsize, bndangles)
            depthmapbbox = depthmap.crop((min(x), min(y), max(x), max(y)))

            # Convert the depthmap to a Numpy array for further processing and
            # sample the raw depthmap at its vertical centerline:
            depthmapbbox_arr = np.asarray(depthmapbbox)
            depthmap_cl = depthmapbbox_arr[:, round(depthmapbbox.size[0]/2)]

            # Calculate the vertical camera angles corresponding to the bottom
            # and top of the facade bounding box:
            imHeight = img.size[1]
            angleTop = ((imHeight/2 - min(y))/(imHeight/2))*math.pi/2
            angleBottom = ((imHeight/2 - max(y))/(imHeight/2))*math.pi/2

            # Take the first derivative of the depthmap with respect to
            # vertical pixel location and identify the depthmap discontinuity
            # (break) locations:
            break_pts = [0]
            boolval_prev = True
            depthmap_cl_dx = np.append(abs(np.diff(depthmap_cl)) < 0.1, True)
            for (counter, boolval_curr) in enumerate(depthmap_cl_dx):
                if ((boolval_prev is True and boolval_curr is False) or
                        (boolval_prev is False and boolval_curr is True)):
                    break_pts.append(counter)
                boolval_prev = boolval_curr
            break_pts.append(counter)

            # Identify the depthmap segments to keep for extrapolation, i.e.,
            # segments that are not discontinuities in the depthmap:
            segments_keep = []
            for i in range(len(break_pts)-1):
                if (all(depthmap_cl_dx[break_pts[i]:break_pts[i+1]]) and
                        all(depthmap_cl[break_pts[i]:break_pts[i+1]] != 255)):
                    segments_keep.append((break_pts[i], break_pts[i+1]))

            # Fit line models to individual (kept) segments of the depthmap and
            # determine the model that results in the smallest residual for
            # all kept depthmap points:
            lm = LinearRegression(fit_intercept=True)
            x = np.arange(depthmapbbox.size[1])
            xKeep = np.hstack([x[segment[0]:segment[1]]
                              for segment in segments_keep])
            yKeep = np.hstack([depthmap_cl[segment[0]:segment[1]]
                              for segment in segments_keep])
            residualprev = 1e10
            model_lm = deepcopy(lm)
            for segment in segments_keep:
                xvect = x[segment[0]:segment[1]]
                yvect = depthmap_cl[segment[0]:segment[1]]

                # Fit model:
                lm.fit(xvect.reshape(-1, 1), yvect)
                preds = lm.predict(xKeep.reshape(-1, 1))
                residual = np.sum(np.square(yKeep-preds))
                if residual < residualprev:
                    model_lm = deepcopy(lm)
                residualprev = residual

            # Extrapolate depthmap using the best-fit model:
            depthmap_cl_depths = model_lm.predict(x.reshape(-1, 1))

            # Calculate heigths of interest:
            R0 = (depthmap_cl_depths[0]*math.sin(angleTop)
                  - depthmap_cl_depths[-1]*math.sin(angleBottom))*3.28084
            scale = R0/R0PixHeight
            R1 = R1PixHeight*scale

            # Calculate roof pitch:
            roof_run = compute_roofrun(footprint)
            roofPitch = (R1-R0)/roof_run
            predictions[image_keys[poly_index]] = {
                'roofEaveHeight': round(R0, 1),
                'buildingHeight': round(R1),
                'roofPitch': round(roofPitch, 2)}

            # Save segmented images
            if self.save_segimages:
                rgb = decode_segmap(pred)

                rgb = Image.fromarray(rgb)
                rgb.save(self.street_images[poly_index].split('.')[0] +
                         '_segmented.png')

        # Unload the model from GPU:
        if torch.cuda.is_available():
            del model
            torch.cuda.empty_cache()

        return predictions
