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
# 08-29-2024

import time
import math
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from brails.types.asset_inventory import AssetInventory
from brails.imputers.imputation import Imputation


class KnnImputer(Imputation):
    """
    Imputes dataset based on k-nearest neighbors in the feature-agmented space. Sequentially generate inventory

    Attributes:
        n_pw (int):
                The number of possible worlds (i.e. samples or realizations)
                batch_size (int):
                        The number of batches for sequential generation. If non-sequential, this variable is not used
                gen_method (str):
                        Select "sequential" or "non-sequential" (one-shot). The latter is faster but does not generate the spatial correlation
                seed (int):
                        For reproducibility

    Methods:





    """

    def __init__(self):
        pass

    def impute(
        self,
        input_inventory: AssetInventory,
        n_possible_worlds=1,
        create_correlation=True,
        exclude_features=[],
        seed=1,
        batch_size=50,
        k_nn=5,
    ) -> AssetInventory:
        self.n_pw = n_possible_worlds
        self.batch_size = batch_size
        self.seed = seed
        self.k_nn = k_nn  # knn
        if create_correlation:
            self.gen_method = "sequential"
        else:
            self.gen_method = "non-sequential"

        #
        # set seed
        #

        np.random.seed(self.seed)

        #
        # convert inventory to df
        #

        bldg_properties_df, bldg_geometries_df, nbldg = input_inventory.get_dataframe()
        column_names = bldg_properties_df.columns

        #
        # drop features to exclude, if specified by the user
        #

        if len(exclude_features) > 0:
            for feature in exclude_features:
                if feature not in column_names:
                    print(
                        "The feature {} does not exist in inventory. Ignoring it from the exclude list.".format(
                            feature
                        )
                    )
                    exclude_features.remove(feature)

            bldg_properties_df = bldg_properties_df.drop(columns=exclude_features)
            column_names = bldg_properties_df.columns

        #
        # replace empty or "NA" with nan & drop entirely missing columns
        #

        pd.set_option("future.no_silent_downcasting", True)
        bldg_properties_df = bldg_properties_df.replace("NA", np.nan, inplace=False)
        bldg_properties_df = bldg_properties_df.replace("", np.nan, inplace=False)
        mask = bldg_properties_df.isnull()  # for missing

        column_entirely_missing = column_names[mask.all(axis=0)]
        bldg_properties_df = bldg_properties_df.drop(columns=column_entirely_missing)
        mask = mask.drop(columns=column_entirely_missing)

        if len(column_entirely_missing) > 1:
            print(
                "Features with no reference data cannot be imputed. Removing them from the imputation target: "
                + ", ".join(list(column_entirely_missing))
            )

        if sum(mask.any(axis=0)) == 0:
            print("No feature to impute")
            output_inventory = input_inventory
            return output_inventory

        #
        # transform category variables into integers
        #

        bldg_properties_encoded, label_encoders, is_category = (
            self.category_in_df_to_indices(bldg_properties_df, mask)
        )

        #
        # Primitive imputation
        #

        bldg_properties_preliminary, nbrs_G, trainY_G_list = self.geospatial_knn(
            bldg_properties_encoded, mask, bldg_geometries_df
        )

        #
        # Cluster
        #

        cluster_ids, n_cluster = self.clustering(
            bldg_properties_encoded,
            bldg_geometries_df,
            nbldg_per_cluster=500,
            seed=self.seed,
        )

        #
        # set up numpy array
        #

        column_names = bldg_properties_encoded.columns

        bldg_encoded_np = bldg_properties_encoded.values  # table with nan
        bldg_inde_np = bldg_properties_encoded.index  # table of building indices
        bldg_prel_np = bldg_properties_preliminary.values  # table of Primitive
        bldg_geom_np = bldg_geometries_df.values  # table of (lat,lon)

        mask_impu_np = mask.values
        bldg_impu_np = bldg_encoded_np.copy()

        #
        # set up placeholders
        #

        sample_dic = {}  # samples
        mp_dic = {}  # most probable value
        for column in column_names[mask.any(axis=0)]:
            mp_dic[column] = {}
            sample_dic[column] = {}

        #
        # Loop over clusters
        #

        elapseStart = time.time()

        print("Running the main imputation. This may take a while.")
        for ci in range(n_cluster):
            if np.mod(ci, 20) == 19:
                print("Enumerating clusters: {} among {}".format(ci + 1, n_cluster))

            #
            # Compute correlation matrix to select important features
            #

            cluster_idx = np.where(cluster_ids == ci)[0]
            corrMat = np.array(bldg_properties_encoded.iloc[cluster_idx].corr())
            const_idx = np.where(bldg_properties_encoded.iloc[cluster_idx].var() == 0)[
                0
            ]
            corrMat[:, const_idx] = 0
            corrMat[const_idx, :] = 0

            #
            # The Primitive values will be used as train_X
            #

            bldg_prel_subset = bldg_prel_np[cluster_idx, :]  # Primitive
            bldg_inde_subset = bldg_inde_np[cluster_idx]  # building indices

            for npp in range(self.n_pw):
                #
                # Sub dataframes corresponding to the current cluster
                #

                bldg_geom_subset = bldg_geom_np[cluster_idx, :]

                bldg_impu_subset = bldg_impu_np[cluster_idx, :]
                mask_impu_subset = mask_impu_np[cluster_idx, :]

                sample_dic, mp_dic = self.sequential_imputer(
                    sample_dic,
                    mp_dic,
                    bldg_impu_subset,
                    mask_impu_subset,
                    bldg_inde_subset,
                    column_names,
                    corrMat,
                    bldg_prel_subset,
                    bldg_geom_subset,
                    nbrs_G,
                    trainY_G_list,
                    is_category,
                    npp,
                    self.gen_method,
                )

        elapseEnd = (time.time() - elapseStart) / 60
        print("Done imputation. It took {:.2f} mins".format(elapseEnd))

        #
        # update inventory
        #

        output_inventory = self.update_inventory(
            input_inventory, sample_dic, label_encoders, is_category
        )

        return output_inventory

    def update_inventory(self, inventory, sample_dic, label_encoders, is_category):
        output_inventory = deepcopy(inventory)

        for column in sample_dic.keys():
            for bldg_idx in sample_dic[column]:
                if is_category[column]:
                    imputed_value_sample = (
                        label_encoders[column]
                        .inverse_transform(sample_dic[column][bldg_idx])
                        .tolist()
                    )
                else:
                    imputed_value_sample = sample_dic[column][bldg_idx]

                if len(imputed_value_sample) == 1:
                    imputed_value_sample = imputed_value_sample[0]

                output_inventory.add_asset_features(
                    bldg_idx, {column: imputed_value_sample}, overwrite=True
                )

        return output_inventory

    def invetory_to_df(self, inventory):
        """
        Convert inventory class to df

        Args:
            inventory (AssetInventory):
                the id of the asset
        """

        features_json = inventory.get_geojson()["features"]
        bldg_properties = [
            (inventory.inventory[i].features | {"index": i})
            for dummy, i in enumerate(inventory.inventory)
        ]

        [bldg_features["properties"] for bldg_features in features_json]

        nbldg = len(bldg_properties)

        bldg_properties_df = pd.DataFrame(bldg_properties)
        bldg_properties_df.drop(columns=["type"], inplace=True)

        # to be used for spatial interpolation
        lat_values = [
            features_json[idx]["geometry"]["coordinates"][0] for idx in range(nbldg)
        ]
        lon_values = [
            features_json[idx]["geometry"]["coordinates"][1] for idx in range(nbldg)
        ]
        bldg_geometries_df = pd.DataFrame()
        bldg_geometries_df["Lat"] = lat_values
        bldg_geometries_df["Lon"] = lon_values
        bldg_geometries_df["index"] = bldg_properties_df["index"]

        bldg_properties_df = bldg_properties_df.set_index("index")
        bldg_geometries_df = bldg_geometries_df.set_index("index")

        return bldg_properties_df, bldg_geometries_df, nbldg

    def category_in_df_to_indices(self, bldg_properties_df, mask):
        X = bldg_properties_df.copy()

        is_category = {}
        label_encoders = {}

        for column in bldg_properties_df.columns:
            values = bldg_properties_df[column]
            idxs = np.array(mask[column] == False)  # removing nans  # noqa: E712

            # if not is_numeric_dtype(values):
            if math.isnan(sum(pd.to_numeric(values[idxs], errors="coerce"))):
                is_category[column] = True
            elif len(np.unique(values)) < 20:
                is_category[column] = True
            else:
                is_category[column] = False

            if is_category[column]:
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le

        bldg_properties_encoded = X.where(
            ~mask, bldg_properties_df
        )  # encoded to indices, but has nan

        return bldg_properties_encoded, label_encoders, is_category

    def geospatial_knn(self, bldg_properties_encoded, mask, bldg_geometries_df):
        bldg_properties_preliminary = bldg_properties_encoded.copy()
        column_names = bldg_properties_encoded.columns

        nbrs_G = {}
        trainY_G_list = {}
        print(
            "Missing percentages among {} assets".format(len(bldg_properties_encoded))
        )
        for column in column_names[mask.any(axis=0)]:
            colLoc = int(column_names.get_loc(column))

            print(
                "{}: {:.2f}%".format(
                    column,
                    sum(bldg_properties_encoded[column].isnull())
                    / len(bldg_properties_encoded)
                    * 100,
                )
            )
            train_id = mask[column] == False  # noqa: E712
            test_id = mask[column]
            trainX_G = np.array(bldg_geometries_df[["Lat", "Lon"]].loc[train_id])
            trainY_G = np.array(bldg_properties_encoded[column].loc[train_id])
            testX_G = np.array(bldg_geometries_df[["Lat", "Lon"]].loc[test_id])

            n_neighbors = min(self.k_nn, trainX_G.shape[0])
            nbrs_G[column] = NearestNeighbors(
                n_neighbors=n_neighbors, algorithm="ball_tree"
            ).fit(trainX_G)
            trainY_G_list[column] = trainY_G
            distances, indices = nbrs_G[column].kneighbors(testX_G)

            for i, t_id in enumerate(np.where(test_id)[0]):
                bldg_properties_preliminary.iat[t_id, colLoc] = trainY_G[indices[i][0]]

        print("Primitive imputation done.")
        return bldg_properties_preliminary, nbrs_G, trainY_G_list

    def clustering(
        self, bldg_properties_encoded, bldg_geometries_df, nbldg_per_cluster=500, seed=0
    ):
        nbldg = bldg_properties_encoded.shape[0]
        n_cluster = int(nbldg / nbldg_per_cluster)  # 500 bldgs per cluster
        kmeans = KMeans(n_clusters=n_cluster, random_state=seed, n_init="auto").fit(
            bldg_geometries_df[["Lat", "Lon"]]
        )
        cluster_ids = kmeans.predict(bldg_geometries_df[["Lat", "Lon"]])

        return cluster_ids, n_cluster

    def sequential_imputer(
        self,
        sample_dic,
        mp_dic,
        bldg_impu_subset,
        mask_impu_subset,
        bldg_inde_subset,
        column_names,
        corrMat,
        bldg_prel_subset,
        bldg_geom_subset,
        nbrs_G,
        trainY_G_list,
        is_category,
        npp,
        gen_method="sequential",
    ):
        tmpcounter = 0
        while np.max(np.sum(mask_impu_subset, 0)) > 0:
            # for sequential generations
            tmpcounter = tmpcounter + 1

            #
            # Loop over columns
            #

            for column in column_names[mask_impu_subset.any(axis=0)]:
                colLoc = column_names.get_loc(column)
                corrVec = corrMat[colLoc, :]
                corrVec[colLoc] = 0  # don't put itself

                selected_columns = list(column_names[abs(corrVec) > 0.4])
                selected_col_idx = [
                    column_names.get_loc(column) for column in selected_columns
                ]

                #
                # get the train and test set
                #

                existing_idx = np.where(mask_impu_subset[:, colLoc] == False)[0]  # noqa: E712
                missing_idx = np.where(mask_impu_subset[:, colLoc])[0]

                trainX = np.hstack(
                    [
                        bldg_prel_subset[existing_idx][:, selected_col_idx],
                        bldg_geom_subset[existing_idx, :],
                    ]
                )
                testX = np.hstack(
                    [
                        bldg_prel_subset[missing_idx][:, selected_col_idx],
                        bldg_geom_subset[missing_idx, :],
                    ]
                )

                trainY = bldg_prel_subset[existing_idx][:, colLoc]

                #
                # Do KNN
                #

                if trainX.shape[0] > 0 and testX.shape[0] > 0:
                    #
                    # run knn classification
                    #

                    # normalize
                    transformer = Normalizer().fit(trainX)
                    trainX = transformer.transform(trainX)
                    testX = transformer.transform(testX)

                    n_neighbors = min(self.k_nn, trainX.shape[0])
                    nbrs = NearestNeighbors(
                        n_neighbors=n_neighbors, algorithm="ball_tree"
                    ).fit(trainX)
                    distances, building_ids = nbrs.kneighbors(testX)
                    mytrainY = trainY

                elif trainX.shape[0] == 0:
                    #
                    # just adopt the global results
                    #

                    # trainX_G = bldg_geom_np[mask_impu_np[:,colLoc]==False]
                    # trainY_G = bldg_impu_np[mask_impu_np[:,colLoc]==False][:, colLoc]
                    # nbrs_G = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(trainX_G)
                    testX = bldg_geom_subset[missing_idx, :]
                    distances, building_ids = nbrs_G[column].kneighbors(testX)
                    mytrainY = trainY_G_list[column]

                else:
                    # just move on
                    continue

                #
                # sample indices
                #

                invdistance = 1 / distances
                row_sums = invdistance.sum(axis=1)
                weights = invdistance / row_sums[:, np.newaxis]

                #
                # sequential generation
                #

                rnd_index = np.array(range((missing_idx.shape[0])))
                if gen_method == "sequential":
                    np.random.shuffle(rnd_index)
                    rnd_index = rnd_index[0 : int(min(testX.shape[0], self.batch_size))]

                #
                # batch generation
                #

                for nb in rnd_index:
                    # global_nb = cluster_idx[missing_idx[nb]]
                    bldg_idx = bldg_inde_subset[missing_idx[nb]]

                    surrounding_building_sample = np.random.choice(
                        building_ids[nb, :], size=1, replace=True, p=weights[nb, :]
                    )
                    closet_building = building_ids[nb, np.argmin(weights[nb, :])]

                    imputed_index_sample = mytrainY[surrounding_building_sample]
                    imputed_index_mp = mytrainY[closet_building]

                    if is_category[column]:
                        myindex = imputed_index_sample.astype(int)[0]
                    else:
                        myindex = imputed_index_sample[0]

                    # if mask_impu_np[global_nb,colLoc]==False:
                    #     raise Exception("Something wrong")

                    # save for iteration
                    # mask_impu_np[global_nb,colLoc] = False # not anymore missing
                    # bldg_impu_np[global_nb,colLoc] = myindex

                    if not mask_impu_subset[missing_idx[nb], colLoc]:
                        raise Exception(
                            "Something went wrong internally. Please report the bug. (SY01)"
                        )

                    # save for iteration
                    mask_impu_subset[missing_idx[nb], colLoc] = (
                        False  # not anymore missing
                    )
                    bldg_impu_subset[missing_idx[nb], colLoc] = myindex
                    bldg_prel_subset[missing_idx[nb], colLoc] = myindex

                    # save to store
                    if self.n_pw == 1:
                        sample_dic[column][bldg_idx] = [myindex]
                        mp_dic[column][bldg_idx] = [imputed_index_mp]

                    else:
                        if npp == 0:
                            sample_dic[column][bldg_idx] = [None] * self.n_pw
                            mp_dic[column][bldg_idx] = [None] * self.n_pw

                        sample_dic[column][bldg_idx][npp] = myindex
                        mp_dic[column][bldg_idx][npp] = imputed_index_mp

                        if npp == self.n_pw - 1:
                            if len(set(sample_dic[column][bldg_idx])) == 1:
                                sample_dic[column][bldg_idx] = [
                                    sample_dic[column][bldg_idx][0]
                                ]

                # bldg_impu_subset = bldg_impu_np[cluster_idx,:]
                # mask_impu_subset = mask_impu_np[cluster_idx,:]

                # bldg_impu_np[cluster_idx,colLoc] = bldg_impu_subset[:,colLoc]
                # mask_impu_np[cluster_idx,colLoc] = mask_impu_subset[:,colLoc]

        return sample_dic, mp_dic
