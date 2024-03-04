import os
import string

import pandas as pd
import numpy as np
import cv2
from scipy.stats import kurtosis, skew
from xgboost import XGBClassifier
import skimage.morphology as morphology


class SegAD:
    """
TODO
    """

    def __init__(self, args, seed, scale_pos_weight, num_components) -> None:
        super().__init__()
        self.args = args
        self.xgb = XGBClassifier(
            random_state=seed,
            n_estimators=10,
            max_depth=5,
            num_parallel_tree=200,
            learning_rate=0.3,
            objective='binary:logitraw',
            colsample_bynode=0.6,
            colsample_bytree=0.6,
            subsample=0.6,
            reg_alpha=1.0,
            scale_pos_weight=scale_pos_weight,
        )
        self.components, self.list_features = self.get_list_features(num_components)

    def forward(self, features):
        score = self.xgb.predict_proba(features)[:, 1]
        return score

    @staticmethod
    def get_list_features(num_components):
        components = []
        for n in range(num_components):
            components.append(string.ascii_lowercase[n])
        lst_f = ["_q995", "_scewness", "_kurtosis", "_mean"]
        list_features = [c + f for c in components for f in lst_f]
        list_features.append("prediction_an_det")
        return components, list_features

    @staticmethod
    def get_features_from_part(part, selection, df):
        if len(selection):
            df[part + "_q995"] = np.quantile(selection, 0.995)
            df[part + "_scewness"] = skew(selection)
            df[part + "_kurtosis"] = kurtosis(selection)
            df[part + "_mean"] = selection.mean()
        else:
            df[part + "_q995"] = 0.0
            df[part + "_scewness"] = 0.0
            df[part + "_kurtosis"] = 0.0
            df[part + "_mean"] = 0.0

    def get_features(self, df, cls):
        # Get segmentation map
        segm_path = os.path.join(self.args.segm_path, cls,
                                 "bad" if df.label else "good",
                                 os.path.basename(df.an_map_path))
        mask = np.load(segm_path)
        mask = morphology.convex_hull_image(mask)

        # Extract features
        anomaly_map = np.load(df.an_map_path)
        for j, part in enumerate(self.components):
            selection = anomaly_map[mask == j]
            self.get_features_from_part(part, selection, df)

        return df
