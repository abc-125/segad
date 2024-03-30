import argparse
import logging
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from SegAD import SegAD


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


# Seeds to reproduce results from the paper
SEEDS = [333, 576, 725, 823, 831, 902, 226, 598, 874, 589]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segm_path", type=str, default="./data/visa_segm", help="Path to the segmentation maps")
    parser.add_argument("--output_path", type=str, default="./data/efficient_ad_output",
                        help="Path to the anomaly maps and dataframes")
    parser.add_argument("--results_path", type=str, default="./results", help="Path to the results")
    parser.add_argument("--bad_parts", type=int, default=10, help="Number of bad parts to use for training")
    return parser.parse_args()


def segad_run():
    if not os.path.exists(args.results_path) or not os.listdir(args.results_path):
        logger.info("Started SegAD training and inference")
        num_components = {
            "candle": 2, "capsules": 2, "cashew": 2, "chewinggum": 2,
            "fryum": 2, "pipe_fryum": 2, "macaroni1": 2, "macaroni2": 2,
            "pcb1": 6, "pcb2": 8, "pcb3": 8, "pcb4": 6,
        }

        mean_auroc = 0
        mean_fpr95tpr = 0
        mean_auroc_an_det = 0
        mean_fpr95tpr_an_det = 0
        results = {}
        results_detailed = {}

        for cls in CATEGORIES:
            auroc = 0
            fpr95tpr = 0
            auroc_an_det = 0
            fpr95tpr_an_det = 0

            df_training_all = pd.read_csv(os.path.join(args.output_path, cls, "df_training.csv"),
                                          index_col=0).reset_index()
            df_testing_all = pd.read_csv(os.path.join(args.output_path, cls, "df_test.csv"), index_col=0).reset_index()
            scale_pos_weight = len(df_training_all.index) / args.bad_parts
            num_comp_cls = num_components[cls]

            for seed in SEEDS:
                # Split bad images from the test set for training and testing
                df_testing_bad = df_testing_all.loc[df_testing_all.label == 1]
                df_training_bad, df_testing_bad = train_test_split(df_testing_bad,
                                                                   test_size=len(df_testing_bad.index) - args.bad_parts,
                                                                   random_state=seed)
                df_training = pd.concat([df_training_all.loc[df_training_all.label == 0], df_training_bad])
                df_testing = pd.concat([df_testing_all.loc[df_testing_all.label == 0], df_testing_bad])

                # Apply SegAD to the training and testing sets
                segad = SegAD(args, seed, scale_pos_weight, num_comp_cls)
                df_training = df_training.apply(lambda row: segad.get_features(row, cls), axis=1)
                segad.xgb.fit(df_training[segad.list_features], df_training.label)
                df_testing = df_testing.apply(lambda row: segad.get_features(row, cls), axis=1)
                predictions = segad.forward(df_testing[segad.list_features])
                df_testing["final_score"] = predictions
                thr_accept = df_testing.loc[df_testing.label == 1, "final_score"].quantile(0.05)

                # Calculate metrics for SegAD and the base anomaly detector, sum for every seed
                auroc = auroc + metrics.roc_auc_score(df_testing.label, predictions)
                fpr95tpr = fpr95tpr + (1 - (df_testing.loc[df_testing.label == 0, "final_score"] < thr_accept).mean())
                auroc_an_det = auroc_an_det \
                                    + metrics.roc_auc_score(df_testing.label, df_testing.prediction_an_det)
                thr_accept = df_testing.loc[df_testing.label == 1, "prediction_an_det"].quantile(0.05)
                fpr95tpr_an_det = fpr95tpr_an_det \
                                    + (1 - (df_testing.loc[df_testing.label == 0, "prediction_an_det"] < thr_accept).mean())

                # Detailed auroc metrics per seed
                results_detailed[cls + str(seed)] = (cls, seed, metrics.roc_auc_score(df_testing.label, predictions))

            # Calculate mean metrics for all seeds
            auroc = auroc / len(SEEDS) * 100
            fpr95tpr = fpr95tpr / len(SEEDS) * 100
            logger.info("SegAD, {}, mean results for all seeds. Cl. AUROC: {}, FPR@95TPR: {}"
                        .format(cls, round(auroc, 1), round(fpr95tpr, 1)))
            results[cls] = (round(auroc, 1), round(fpr95tpr, 1))
            mean_auroc = mean_auroc + auroc
            mean_fpr95tpr = mean_fpr95tpr + fpr95tpr
            auroc_an_det = auroc_an_det / len(SEEDS) * 100
            fpr95tpr_an_det = fpr95tpr_an_det / len(SEEDS) * 100
            mean_auroc_an_det = mean_auroc_an_det + auroc_an_det
            mean_fpr95tpr_an_det = mean_fpr95tpr_an_det + fpr95tpr_an_det

        mean_auroc = round(mean_auroc / len(CATEGORIES), 1)
        mean_fpr95tpr = round(mean_fpr95tpr / len(CATEGORIES), 1)
        mean_auroc_an_det = round(mean_auroc_an_det / len(CATEGORIES), 1)
        mean_fpr95tpr_an_det = round(mean_fpr95tpr_an_det / len(CATEGORIES), 1)
        results["mean"] = (mean_auroc, mean_fpr95tpr)
        os.makedirs(args.results_path, exist_ok=True)
        pd.DataFrame.from_dict(results, orient="index").to_csv(os.path.join(args.results_path, "results.csv"))
        pd.DataFrame.from_dict(results_detailed, orient="index").to_csv(os.path.join(args.results_path,
                                                                                     "results_detailed.csv"))

        logger.info("Finished SegAD training and inference")
        logger.info("EfficientAD, mean Cl. AUROC: {}, mean FPR@95TPR: {}".format(mean_auroc_an_det, mean_fpr95tpr_an_det))
        logger.info("SegAD, mean Cl. AUROC: {}, mean FPR@95TPR: {}".format(mean_auroc, mean_fpr95tpr))
    else:
        logger.info("Folder with results exists and not empty.")


if __name__ == '__main__':
    args = get_args()

    segad_run()
