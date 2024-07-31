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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 05-06-2024   

from brails.types.image_set import Image, ImageSet
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

import PIL
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
from shapely.geometry import Polygon
from tqdm import tqdm
from pathlib import Path

class GoogleStreetview(ImageScraper):
    
    def __init__(self, input_data: dict):

        api_key = input_data["apiKey"]
        
        # Check if the provided Google API Key successfully obtains street view
        # imagery metadata for Doe Memorial Library of UC Berkeley:
        responseStreet = requests.get('https://maps.googleapis.com/maps/api/streetview/metadata?' + 
                                      'location=37.8725187407,-122.2596028649' +
                                      '&source=outdoor' + 
                                      f'&key={api_key}')

        # If the requested image cannot be downloaded, notify the user of the
        # error and stop program execution:
        if 'error' in responseStreet.text.lower():
            error_message = ('Google API key error. The entered API key is valid '
                             + 'but does not have Street View Static API enabled. ' 
                             + 'Please enter a key that has the Street View' 
                             + 'Static API enabled.')
            sys.exit(error_message)
            
        self.apikey = api_key


    def GetGoogleStreetImage(self, footprints, dir_path, save_interim_images=False,save_all_cam_metadata=False):
        
        def get_bin(a):
            ba = bin(a)[2:]
            return "0"*(8 - len(ba)) + ba

        def getUInt16(arr, ind):
            a = arr[ind]
            b = arr[ind + 1]
            return int(get_bin(b) + get_bin(a), 2)

        def getFloat32(arr, ind):
            return bin_to_float("".join(get_bin(i) for i in arr[ind : ind + 4][::-1]))

        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        def parse_dmap_str(b64_string):
            # Ensure correct padding (The length of string needs to be divisible by 4):
            b64_string += "="*((4 - len(b64_string)%4)%4)

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
                byteOffset = header["offset"] + header["width"]*header["height"] + i*4*4
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

        def get_depth_map(pano, saveim=False, imname='depthmap.jpg'):
            
            # Decode depth map string:
            depthMapData = parse_dmap_str(pano['depthMapString'])
            
            # Parse first bytes to get the data headers:
            header = parse_dmap_header(depthMapData)
            
            # Parse remaining bytes into planes of float values:
            data = parse_dmap_planes(header, depthMapData)
            
            # Compute position and depth values of pixels:
            depthMap = compute_dmap(header, data["indices"], data["planes"])
            
            # Process float 1D array into integer 2D array with pixel values ranging 
            # from 0 to 255:
            im = depthMap["depthMap"]
            im[np.where(im == max(im))[0]] = 255
            if min(im) < 0:
                im[np.where(im < 0)[0]] = 0
            im = im.reshape((depthMap["height"], depthMap["width"]))
            
            # Flip the 2D array to have it line up with pano image pixels:
            im = np.fliplr(im)

            # Read the 2D array into an image and resize this image to match the size 
            # of pano:
            #imMask = Image.fromarray(np.uint8(im))
            imMask = Image.fromarray(im)
            imMask = imMask.resize(pano['imageSize'])
            pano['depthMap'] = imMask.copy()
            
            if saveim:
                # Convert the float values to integer:
                imSave = imMask.convert('L')
                        
                # Save the depth map image:
                imSave.save(imname)
                pano['depthImFile'] = imname
            else:
                pano['depthImFile'] = ''
            return pano

        def download_tiles(urls):    
            # Define a retry strategy for downloading a tile if a common error code
            # is encountered:
            s = requests.Session()
            retries = Retry(total=5, 
                            backoff_factor=0.1,
                            status_forcelist=[500, 502, 503, 504])
            s.mount('https://', HTTPAdapter(max_retries=retries))
            # Given the URLs, download tiles and save them as a PIL images: 
            tiles = []
            for url in urls:
                response = s.get(url)
                tiles.append(PIL.Image.open(BytesIO(response.content)))
            return tiles

        def download_pano(pano, saveim=False, imname='pano.jpg'):
            panoID = pano['id']
            imSize = pano['imageSize']
            zoomVal = pano['zoom']
            
            # Calculate tile locations (offsets) and determine corresponding
            # tile URL links:
            baseurl = f'https://cbk0.google.com/cbk?output=tile&panoid={panoID}&zoom={zoomVal}'
            urls = []
            offsets = []
            for x in range(int(imSize[0]/512)):
                for y in range(int(imSize[1]/512)):
                    urls.append(baseurl+f'&x={x}&y={y}')
                    offsets.append((x*512,y*512))
            images = download_tiles(urls)
            
            # Combine the downloaded tiles to get the uncropped pano:
            combined_im = PIL.Image.new('RGB', imSize)
            for (ind,im) in enumerate(images):
               combined_im.paste(im, offsets[ind])
            
            # Save the uncropped pano:
            pano['panoImage'] = combined_im.copy()
            if saveim:              
                combined_im.save(imname)
                pano['panoImFile'] = imname
            else:
                pano['panoImFile'] = ''     
            return pano

        def get_pano_id(latlon,apikey):
            # Obtain the pano id containing the image of the building at latlon:
            endpoint = 'https://maps.googleapis.com/maps/api/streetview/metadata'
            params = {
                'location': f'{latlon[0]}, {latlon[1]}',
                'key': apikey,
                'source': 'outdoor',
            }
            
            r = requests.get(endpoint, params=params)
            return r.json()['pano_id']

        def get_pano_meta(pano, savedmap = False, dmapoutname = 'depthmap.txt'):
            # Get the metadata for a pano image:
            baseurl = 'https://www.google.com/maps/photometa/v1'
            params = {
                'authuser': '0',
                'hl': 'en',
                'gl': 'us',
                'pb': '!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!1e2!2s' + pano['id'] + '!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3'
            }
            
            # Send GET request to API endpoint and retrieve response:
            response = requests.get(baseurl, params=params, proxies=None)
            
            # Extract depthmap and other image metadata from response:
            response = response.content
            response = json.loads(response[4:])
            pano['zoom'] = 3
            pano['depthMapString'] = response[1][0][5][0][5][1][2]
            pano['camLatLon'] = (response[1][0][5][0][1][0][2],response[1][0][5][0][1][0][3])
            pano['imageSize'] = tuple(response[1][0][2][3][0][pano['zoom']][0])[::-1]
            pano['heading'] = response[1][0][5][0][1][2][0]
            pano['pitch'] = response[1][0][5][0][1][2][1]
            pano['fov'] = response[1][0][5][0][1][2][2]
            pano['cam_elev'] = response[1][0][5][0][1][1][0]
            #pano['city'] = response[1][0][3][2][1][0]
            
            # If savedmap is set to True write the depthmap string into a text file:
            with open(dmapoutname,'w') as dmapfile:
                dmapfile.write(pano['depthMapString'])
            return pano

        def get_composite_pano(pano,imname='panoOverlaid.jpg'):
            # Convert depth map into a heat map:
            im = np.array(pano['depthMap'].convert('L'))
            cm_jet = mpl.colormaps['jet']
            im = cm_jet(im)
            im = np.uint8(im*255)
            imMask = Image.fromarray(im).convert('RGB')
            
            # Overlay the heat map on the pano:
            imPano = pano['panoImage']
            imOverlaid = Image.blend(imMask, imPano, 0.5)
            imOverlaid.save(imname)

        def get_3pt_angle(a, b, c):
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang

        def get_angle_from_heading(coord,heading):
            # Determine the cartesian coordinates of a point along the heading that
            # is 100 ft away from the origin:
            x0 = 100*math.sin(math.radians(heading))
            y0 = 100*math.cos(math.radians(heading))
            
            # Calculate the clockwise viewing angle for each coord with respect to the 
            # heading:
            ang = 360 - get_3pt_angle((x0,y0), (0,0), coord)
            
            # Return viewing angles such that anything to the left of the vertical 
            # camera axis is negative and counterclockwise angle measurement:
            return ang if ang<=180 else ang-360

        def get_bldg_image(pano, fp, imname='imstreet.jpg', saveDepthMap=False):
            # Project the coordinates of the footprint to Cartesian with the 
            # approximate camera location set as the origin:
            (lat0,lon0) = pano['camLatLon']
            xy = []
            for vert in fp:
                lon1 = vert[1]
                lat1 = vert[0]
                x = (lon1 - lon0)*40075000*3.28084*math.cos((lat0 + lat1)*math.pi/360)/360
                y = (lat1 - lat0)*40075000*3.28084/360
                xy.append((x,y))
            
            # Calculate the theoretical viewing angle for each footprint vertex with 
            # respect to the camera heading angle
            camera_angles = []
            for coord in xy:
                camera_angles.append(get_angle_from_heading(coord,pano['heading']))
            
            # Calculate the viewing angle values that encompass the building buffered 
            # 10 degrees in horizontal direction:
            bndAngles = np.rint((np.array([round(min(camera_angles),-1)-10,
                                           round(max(camera_angles),-1)+10]) + 180)/360*pano['imageSize'][0])
            
            im = pano['panoImage']
            imCropped = im.crop((bndAngles[0],0,bndAngles[1],pano['imageSize'][1]))
            imCropped.save(imname)
            pano['pano_bnd_angles'] = np.copy(bndAngles)
            
            if saveDepthMap:
                # Get the depth map for the pano:
                panoDmapName = imname.replace('.' + imname.split('.')[-1],'') + '_pano_depthmap.jpg'
                pano = get_depth_map(pano, saveim=saveDepthMap, imname = panoDmapName)
                mask = pano['depthMap']
                
                # Crop the horizontal parts of the image outside the bndAngles:
                dmapName = imname.replace('.' + imname.split('.')[-1],'') + '_depthmap.jpg'
                maskCropped = mask.crop((bndAngles[0],0,bndAngles[1],pano['imageSize'][1]))
                pano['depthMapBldg'] = maskCropped.copy()
                maskCropped.convert('RGB').save(dmapName)
            
            return pano

        def download_streetlev_image(fp, fpcent, im_name, depthmap_name, apikey,
                                     saveInterIm=False, saveAllCamMeta=False):
            if saveInterIm:    
                imnameBase = im_name.replace('.' + im_name.split('.')[-1],'')
                panoName = imnameBase + '_pano.jpg'
                compImName = imnameBase + '_composite.jpg'
                panoDmapName = imnameBase + '_depthmap.jpg'
            else:
                panoName = ''
                compImName = ''
                panoDmapName = ''   
            
            # Define the internal pano dictionary:
            pano = {'queryLatLon':fpcent,
                    'camLatLon':(),
                    'id':'',
                    'imageSize':(),
                    'heading':0,
                    'depthMap':0,
                    'depthMapString':'',
                    'panoImFile': panoName,
                    'depthImFile':panoDmapName,
                    'compositeImFile':compImName
                    }
            
            # Get pano ID. If no pano exists, skip the remaining steps of the 
            # function:
            try:
                pano['id'] = get_pano_id(pano['queryLatLon'],apikey)
            except:
                return None
            
            # Get the metdata for the pano:
            pano = get_pano_meta(pano, savedmap = True, dmapoutname = depthmap_name)
            
            # Download the pano image:
            pano = download_pano(pano, saveim=saveInterIm, imname = panoName)
            
            # Crop out the building-specific portions of the pano and depthmap:
            pano = get_bldg_image(pano, fp, imname=im_name, saveDepthMap=saveInterIm)

            if saveInterIm:
                # Overlay depth map on the pano:
                get_composite_pano(pano, imname=compImName)

            # Return camera elevation, depthmap, and, if requested, other camera metadata:
            if saveAllCamMeta:
                out = (pano['cam_elev'],pano['camLatLon'],(depthmap_name, pano['imageSize'],pano['pano_bnd_angles']),pano['fov'],pano['heading'],pano['pitch'],pano['zoom'])
            else:
                out = (pano['cam_elev'],(depthmap_name, pano['imageSize'],pano['pano_bnd_angles']))
            
            return out
 
        # Create a directory to save the street-level images and corresponding
        # depthmaps:
        
        #os.makedirs(f'{dir_path}',exist_ok=True)
        #os.makedirs(f'{dir_path}',exist_ok=True)
 
        # Compute building footprints, parse satellite image and depthmap file
        # names, and create the list of inputs required for obtaining 
        # street-level imagery:
        self.footprints = footprints
        self.centroids = []
        street_images = []
        inps = [] 
        for footprint in footprints:
            fp = np.fliplr(np.squeeze(np.array(footprint))).tolist()
            fp_cent = Polygon(footprint).centroid
            self.centroids.append([fp_cent.x,fp_cent.y])
            imName = str(round(fp_cent.y,8))+str(round(fp_cent.x,8))
            imName.replace(".","")
            im_name = f"{dir_path}/imstreet_{imName}.jpg"
            depthmap_name = f"{dir_path}/dmstreet_{imName}.txt"            
            street_images.append(im_name)
            inps.append((fp,(fp_cent.y,fp_cent.x),im_name,depthmap_name))
        
        # Download building-wise street-level imagery and depthmap strings:
        pbar = tqdm(total=len(footprints), desc='Obtaining street-level imagery')     
        results = {}             
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(download_streetlev_image, fp, fpcent, fout, 
                                dmapout, self.apikey, 
                                saveInterIm=save_interim_images,
                                saveAllCamMeta=save_all_cam_metadata): fout
                for fp, fpcent, fout, dmapout in inps
            }
            for future in concurrent.futures.as_completed(future_to_url):
                fout = future_to_url[future]
                pbar.update(n=1)
                try:
                    results[fout] = future.result()
                except Exception as exc:
                    results[fout] = None
                    print("%r generated an exception: %s" % (fout, exc))

        # Save the depthmap and all other required camera metadata in
        # the class object:
        
        if save_all_cam_metadata==False:
            self.cam_elevs = []
            self.depthmaps = [] 
            for (ind,im) in enumerate(street_images):
                if results[im] is not None:
                    self.cam_elevs.append(results[im][0])
                    self.depthmaps.append(results[im][1])
                else:
                    street_images[ind] = None
                    self.cam_elevs.append(None)
                    self.depthmaps.append(None)
        else:
            self.cam_elevs = []
            self.cam_latlons = []
            self.depthmaps = []
            self.fovs = []
            self.headings = []
            self.pitch = []
            self.zoom_levels = []
            for (ind,im) in enumerate(street_images):
                if results[im] is not None:
                    self.cam_elevs.append(results[im][0])
                    self.cam_latlons.append(results[im][1])
                    self.depthmaps.append(results[im][2])
                    self.fovs.append(results[im][3])
                    self.headings.append(results[im][4])
                    self.pitch.append(results[im][5])
                    self.zoom_levels.append(results[im][6])            
                else:
                    self.cam_elevs.append(None)
                    self.cam_latlons.append(None)
                    self.depthmaps.append(None)
                    self.fovs.append(None)
                    self.headings.append(None)
                    self.pitch.append(None)
                    self.zoom_levels.append(None)
                    street_images[ind] = None
        self.street_images = street_images.copy()
        

    def get_images(self, inventory: AssetInventory, dir_path: str) -> ImageSet:
        """
        This method obtains street-level imagery of buildings given the 
        footprintsin the asset inventory

        Args:
              inventory (AssetInventory):
                   The AssetInventory

        Returns:
              Image_Set:
                    An image_Set for the assets in the inventory

        """

        # Ensure consistency in dir_path, i.e remove ending / if given:
        dir_path = Path(dir_path)
        os.makedirs(f'{dir_path}',exist_ok=True)
        
        # Create the footprints from the items in AssetInventory
        # Keep the asset keys in a list for later use:
        result = ImageSet()
        result.dir_path = dir_path        

        asset_footprints = []
        asset_keys = []
        for key, asset in inventory.inventory.items():
            asset_footprints.append(asset.coordinates)
            asset_keys.append(key)

        # Get the image filenames and properties, and create a new image:
        self.GetGoogleStreetImage(asset_footprints, dir_path, False, True)

        for i in range(len(asset_keys)):
            
            filename = self.street_images[i]            
            if filename is not None:
                key = asset_keys[i]

                # going to rename image files to use key .. barbaros can fix in code so no rename
                current_file_path = Path(filename)
                new_name = f'gstrt_{key}{current_file_path.suffix}'
                new_file_path = current_file_path.parent / new_name

                # sy - to prevent error : [WinError 183] Cannot create a file when that file already exists: 'tmp\\street\\imstreet_37.87343446-122.45684953.jpg' -> 'tmp\\street\\gstrt_596.jpg'
                # we should not download the files
                #current_file_path.rename(new_file_path)
                current_file_path.replace(new_file_path)



                # might as well do same for depthmap
                current_depthfile_path = Path(self.depthmaps[i][0])
                new_depthname = f'dmap_{key}{current_depthfile_path.suffix}'
                new_depthfile_path = current_depthfile_path.parent / new_depthname                
                #current_depthfile_path.rename(new_depthfile_path) # sy - permissing error when file already exists
                current_depthfile_path.replace(new_depthfile_path)

                name_stripped = new_file_path.name
                
                properties = {}
                properties['elev'] = self.cam_elevs[i]
                properties['latlon'] = self.cam_latlons[i]
                properties['depthmap'] = current_depthfile_path.name
                properties['fov'] = self.fovs[i]
                properties['heading'] = self.headings[i]
                properties['pitch'] = self.pitch[i]
                properties['zoom_level'] = self.zoom_levels[i]
                
                result.add_image(key, name_stripped, properties)
            
        return result