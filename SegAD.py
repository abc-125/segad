import os
import string

import numpy as np
from scipy.stats import kurtosis, skew
from xgboost import XGBClassifier


class SegAD:
    def __init__(self, args, seed, scale_pos_weight, num_components, models_list) -> None:
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
        self.models_list = models_list
        self.components, self.list_features = self.get_list_features(num_components)

    def forward(self, features):
        score = self.xgb.predict_proba(features)[:, 1]
        return score

    def get_list_features(self, num_components):
        components = []
        for n in range(num_components):
            components.append(string.ascii_lowercase[n])
        features = ["_q995", "_scewness", "_kurtosis", "_mean"]
        lst = [c + "_" + m for c in components for m in self.models_list]
        list_features = [l + f for l in lst for f in features]
        for model in self.models_list:
            list_features.append("an_det_score_" + model)
        return components, list_features

    @staticmethod
    def get_features_from_part(part, model, selection, df):
        if len(selection):
            df[part + "_" + model + "_q995"] = np.quantile(selection, 0.995)
            df[part + "_" + model + "_scewness"] = skew(selection)
            df[part + "_" + model + "_kurtosis"] = kurtosis(selection)
            df[part + "_" + model + "_mean"] = selection.mean()
        else:
            df[part + "_" + model + "_q995"] = 0.0
            df[part + "_" + model + "_scewness"] = 0.0
            df[part + "_" + model + "_kurtosis"] = 0.0
            df[part + "_" + model + "_mean"] = 0.0

    def get_features(self, df, cls):
        # Load segmentation map
        segm_path = os.path.join(self.args.segm_path, cls,
                                 "bad" if df.label else "good",
                                 os.path.basename(df.an_map_path))
        mask = np.load(segm_path)

        # Load anomaly map and extract features
        for model in self.models_list:
            an_path = os.path.join(self.args.an_path, model, cls,
                                   "anomaly_maps",
                                   "bad" if df.label else "good",
                                   os.path.basename(df.an_map_path))
            anomaly_map = np.load(an_path)
            for j, part in enumerate(self.components):
                selection = anomaly_map[mask == j]
                self.get_features_from_part(part, model, selection, df)
            df["an_det_score_" + model] = np.max(anomaly_map)

        return df
