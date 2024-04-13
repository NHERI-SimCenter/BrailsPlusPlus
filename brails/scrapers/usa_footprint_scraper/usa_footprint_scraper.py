# written Barbaros Cetiner 03/24

from brails.scraper.footprint_scraper import FootprintScraper
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import AssetInventory

import math
import json
import requests
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
from shapely.ops import linemerge, unary_union, polygonize
from shapely.strtree import STRtree
from brails.utils.geo_tools import *
import concurrent.futures
from requests.adapters import HTTPAdapter, Retry
import unicodedata


class USA_FootprintScraper(FootprintScraper):
    """
    A class to generate the foorprint data utilizing USA Structures

    Attributes:

    Methods:
        __init__: Constructor that just creates an empty footprint
        get_inventory(id, coordinates): to get the inventory

    """

    def __init__(self, input: dict):
        """
        Initialize the object

        Args
            input: a dict defining length units, if no ;length' ft is assumed
        """

        self.lengthUnit = input.get("length")
        if self.lengthUnit == None:
            self.lengthUnit = "ft"

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        This method will be used by the caller to obtain the footprints for builings in an area.

        Args:
            region (RegionBoundary): The region of interest.

        Returns:
            BuildingInventory: A building inventory for buildings in the region.

        """

        bpoly, queryarea_printname, osmid = region.get_boundary()

    def get_usastruct_bldg_counts(bpoly):
        # Get the coordinates of the bounding box for input polygon bpoly:
        bbox = bpoly.bounds

        # Get the number of buildings in the computed bounding box:
        query = (
            "https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/"
            + "rest/services/USA_Structures_View/FeatureServer/0/query?"
            + f"geometry={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            + "&geometryType=esriGeometryEnvelope&inSR=4326"
            + "&spatialRel=esriSpatialRelIntersects"
            + "&returnCountOnly=true&f=json"
        )

        s = requests.Session()
        retries = Retry(
            total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))

        r = s.get(query)
        totalbldgs = r.json()["count"]

        return totalbldgs

    def get_polygon_cells(bpoly, totalbldgs=None, nfeaturesInCell=4000, plotfout=False):
        if totalbldgs is None:
            # Get the number of buildings in the input polygon bpoly:
            totalbldgs = get_usastruct_bldg_counts(bpoly)

        if totalbldgs > nfeaturesInCell:
            # Calculate the number of cells required to cover the polygon area with
            # 20 percent margin of error:
            ncellsRequired = round(1.2 * totalbldgs / nfeaturesInCell)

            # Get the coordinates of the bounding box for input polygon bpoly:
            bbox = bpoly.bounds

            # Calculate the horizontal and vertical dimensions of the bounding box:
            xdist = haversine_dist((bbox[0], bbox[1]), (bbox[2], bbox[1]))
            ydist = haversine_dist((bbox[0], bbox[1]), (bbox[0], bbox[3]))

            # Determine the bounding box aspect ratio defined (as a number greater
            # than 1) and the long direction of the bounding box:
            if xdist > ydist:
                bboxAspectRatio = math.ceil(xdist / ydist)
                longSide = 1
            else:
                bboxAspectRatio = math.ceil(ydist / xdist)
                longSide = 2

            # Calculate the cells required on the short side of the bounding box (n)
            # using the relationship ncellsRequired = bboxAspectRatio*n^2:
            n = math.ceil(math.sqrt(ncellsRequired / bboxAspectRatio))

            # Based on the calculated n value determined the number of rows and
            # columns of cells required:
            if longSide == 1:
                rows = bboxAspectRatio * n
                cols = n
            else:
                rows = n
                cols = bboxAspectRatio * n

            # Determine the coordinates of each cell covering bpoly:
            rectangles = mesh_polygon(bpoly, rows, cols)
        else:
            rectangles = [bpoly.envelope]

        # Plot the generated mesh:
        if plotfout:
            plot_polygon_cells(bpoly, rectangles, plotfout)

        return rectangles

    def refine_polygon_cells(premCells, nfeaturesInCell=4000):
        # Download the building count for each cell:
        pbar = tqdm(
            total=len(premCells), desc="Obtaining the number of buildings in each cell"
        )
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(get_usastruct_bldg_counts, rect): rect
                for rect in premCells
            }
            for future in concurrent.futures.as_completed(future_to_url):
                rect = future_to_url[future]
                pbar.update(n=1)
                try:
                    results[rect] = future.result()
                except Exception as exc:
                    results[rect] = None
                    print("%r generated an exception: %s" % (rect, exc))

        indRemove = []
        cells2split = []
        cellsKeep = premCells.copy()
        for ind, rect in enumerate(premCells):
            totalbldgs = results[rect]
            if totalbldgs is not None:
                if totalbldgs == 0:
                    indRemove.append(ind)
                elif totalbldgs > nfeaturesInCell:
                    indRemove.append(ind)
                    cells2split.append(rect)

        for i in sorted(indRemove, reverse=True):
            del cellsKeep[i]

        cellsSplit = []
        for rect in cells2split:
            rectangles = get_polygon_cells(rect, totalbldgs=results[rect])
            cellsSplit += rectangles

        return cellsKeep, cellsSplit

    def download_ustruct_bldgattr(cell):
        rect = cell.bounds
        s = requests.Session()
        retries = Retry(
            total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        query = (
            "https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/"
            + "rest/services/USA_Structures_View/FeatureServer/0/query?"
            + f"geometry={rect[0]},{rect[1]},{rect[2]},{rect[3]}"
            + "&outFields=BUILD_ID,HEIGHT"
            + "&geometryType=esriGeometryEnvelope&inSR=4326"
            + "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
        )

        r = s.get(query)
        datalist = r.json()["features"]
        ids = []
        footprints = []
        bldgheight = []
        for data in datalist:
            footprint = data["geometry"]["rings"][0]
            bldgid = data["attributes"]["BUILD_ID"]
            if bldgid not in ids:
                ids.append(bldgid)
                footprints.append(footprint)
                height = data["attributes"]["HEIGHT"]
                try:
                    height = float(height)
                except:
                    height = None
                    bldgheight.append(height)

        return (ids, footprints, bldgheight)

    def download_ustruct_bldgattr4region(cellsFinal, bpoly):
        # Download building attribute data for each cell:
        pbar = tqdm(
            total=len(cellsFinal),
            desc="Obtaining the building attributes for each cell",
        )
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(download_ustruct_bldgattr, cell): cell
                for cell in cellsFinal
            }
            for future in concurrent.futures.as_completed(future_to_url):
                cell = future_to_url[future]
                pbar.update(n=1)
                try:
                    results[cell] = future.result()
                except Exception as exc:
                    results[cell] = None
                    print("%r generated an exception: %s" % (cell, exc))
        pbar.close()

        # Parse the API results into building id, footprints and height
        # information:
        ids = []
        footprints = []
        bldgheight = []
        for cell in tqdm(cellsFinal):
            res = results[cell]
            ids += res[0]
            footprints += res[1]
            bldgheight += res[2]

        # Remove the duplicate footprint data by recording the API
        # outputs to a dictionary:
        data = {}
        for ind, bldgid in enumerate(ids):
            data[bldgid] = [footprints[ind], bldgheight[ind]]

        # Define length unit conversion factor:
        if lengthUnit == "ft":
            convFactor = 3.28084
        else:
            convFactor = 1

        # Calculate building centroids and save the API outputs into
        # their corresponding variables:
        footprints = []
        attributes = {"buildingheight": []}
        centroids = []
        for value in data.values():
            fp = value[0]
            centroids.append(Polygon(fp).centroid)
            footprints.append(fp)
            heightout = value[1]
            if heightout is not None:
                attributes["buildingheight"].append(round(heightout * convFactor, 1))
            else:
                attributes["buildingheight"].append(None)

        # Identify building centroids and that fall outside of bpoly:
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(bpoly.contains, cent): cent for cent in centroids
            }
            for future in concurrent.futures.as_completed(future_to_url):
                cent = future_to_url[future]
                try:
                    results[cent] = future.result()
                except Exception as exc:
                    results[cell] = None
                    print("%r generated an exception: %s" % (cent, exc))
        indRemove = []
        for ind, cent in enumerate(centroids):
            if not results[cent]:
                indRemove.append(ind)

        # Remove data corresponding to centroids that fall outside bpoly:
        for i in sorted(indRemove, reverse=True):
            del footprints[i]
            del attributes["buildingheight"][i]

        return footprints, attributes

    def __init__(self, input: dict):
        """
        Initialize the object

        Args
            input: a dict defining length units, if no ;length' ft is assumed
        """

        self.lengthUnit = input.get("length")
        if self.lengthUnit == None:
            self.lengthUnit = "ft"

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        This method will be used by the caller to obtain the footprints for builings in an area.

        Args:
            region (RegionBoundary): The region of interest.

        Returns:
            BuildingInventory: A building inventory for buildings in the region.

        """

        bpoly, queryarea_printname, osmid = region.get_boundary()

        plotCells = True

        if plotCells:
            meshInitialfout = (
                queryarea_printname.replace(" ", "_") + "_Mesh_Initial.png"
            )
            meshFinalfout = queryarea_printname.replace(" ", "_") + "_Mesh_Final.png"

        print("\nMeshing the defined area...")
        cellsPrem = get_polygon_cells(bpoly, plotfout=meshInitialfout)

        if len(cellsPrem) > 1:
            cellsFinal = []
            cellsSplit = cellsPrem.copy()
            while len(cellsSplit) != 0:
                cellsKeep, cellsSplit = refine_polygon_cells(cellsSplit)
                cellsFinal += cellsKeep
            print(
                f"\nMeshing complete. Split {queryarea_printname} into {len(cellsFinal)} cells"
            )
        else:
            cellsFinal = cellsPrem.copy()
            print(
                f"\nMeshing complete. Covered {queryarea_printname} with a rectangular cell"
            )

        if plotCells:
            plot_polygon_cells(bpoly, cellsFinal, meshFinalfout)

        footprints, attributes = download_ustruct_bldgattr4region(cellsFinal, bpoly)
        print(
            f"\nFound a total of {len(footprints)} building footprints in {queryarea_printname}"
        )

        return self._create_asset_inventory(footprints, attributes, self.lengthUnit)
