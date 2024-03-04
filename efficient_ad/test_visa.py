#!/usr/bin/python
# -*- coding: utf-8 -*-
# Based on https://github.com/nelson1425/EfficientAD under the http://www.apache.org/licenses/LICENSE-2.0

from logging import getLogger
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import random
import pandas as pd
from tqdm import tqdm

from efficient_ad.common import ImageFolderWithPath, ImageFolderWithoutTargetTrain, ImageFolderWithPathTrainFinal

on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

seed = 333

logger = getLogger(__name__)


default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def transform(image):
    return default_transform(image), default_transform(image)


@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:]) ** 2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         data_path="", desc='Running inference'):
    y_true = []
    y_score = []
    an_paths = []
    paths = []
    for image, target, path in tqdm(test_set, desc=desc):
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        # map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (image_size, image_size), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        defect_class = os.path.basename(os.path.dirname(path))

        if not os.path.exists(os.path.join(test_output_dir, defect_class)):
            os.makedirs(os.path.join(test_output_dir, defect_class))
        img_nm = os.path.split(path)[1].split('.')[0]
        an_path = os.path.join(test_output_dir, defect_class, img_nm + '.npy')
        np.save(an_path, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
        an_paths.append(an_path)
        paths.append(os.path.join(data_path, defect_class, os.path.basename(path)))

    return y_true, y_score, an_paths, paths


def test_class(args, cls):
    # Paths to dataset
    dataset_path_test = os.path.join(args.visa_path, cls, "test")
    dataset_path_train = os.path.join(args.visa_path, cls, "train")

    # Paths to save anomaly maps and dataframes
    output_path = os.path.join(args.output_path, cls)
    an_maps_path = os.path.join(output_path, "anomaly_maps")
    if not os.path.exists(an_maps_path):
        os.makedirs(an_maps_path)

    # Paths to load weights
    efficient_ad_path = os.path.join(args.models_path, cls)
    weights_path = os.path.join(efficient_ad_path, "weights")

    # Create dataloaders
    train_set = ImageFolderWithoutTargetTrain(dataset_path_train, transform=transforms.Lambda(transform))
    _, val_set = train_test_split(train_set, test_size=0.1, random_state=seed)
    val_loader = DataLoader(val_set, batch_size=1)
    test_set = ImageFolderWithPath(dataset_path_test)

    # Load models
    teacher = torch.load(os.path.join(weights_path, "teacher.pth"))
    student = torch.load(os.path.join(weights_path, "student.pth"))
    autoencoder = torch.load(os.path.join(weights_path, "autoencoder.pth"))
    teacher.eval()
    student.eval()
    autoencoder.eval()
    teacher_mean = torch.from_numpy(np.load(os.path.join(efficient_ad_path, "teacher_mean.npy")))
    teacher_std = torch.from_numpy(np.load(os.path.join(efficient_ad_path, "teacher_std.npy")))
    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        teacher_mean = teacher_mean.cuda()
        teacher_std = teacher_std.cuda()

    # Normalization
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=val_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc="Normalization")

    # Process images for test
    y_true, y_score, an_paths, paths = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=an_maps_path, data_path=dataset_path_test, desc="Testing set")
    logs = pd.DataFrame({"filepath": paths, "an_map_path": an_paths, "label": y_true, "prediction_an_det": y_score})
    logs.to_csv(os.path.join(output_path, "df_test.csv"), index=False)

    # Process images for final training
    # Use the same good images which were used for the normalization earlier (val_set)
    train_set2 = ImageFolderWithPathTrainFinal(dataset_path_train)
    _, val_set = train_test_split(train_set2, test_size=0.1, random_state=seed)
    final_train_set = val_set
    y_true, y_score, an_paths, paths = test(
        test_set=final_train_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=an_maps_path, data_path=dataset_path_train, desc="Training set")
    logs = pd.DataFrame({"filepath": paths, "an_map_path": an_paths, "label": y_true, "prediction_an_det": y_score})
    logs.to_csv(os.path.join(output_path, "df_training.csv"), index=False)


def run(args, categories):
    if not os.path.exists(args.output_path) \
            or not len(os.listdir(args.output_path)) == len(categories):
        logger.info("Started EfficientAD inference")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for cls in categories:
            logger.info("VisA, {}:".format(cls))
            test_class(args, cls)

        logger.info("Finished EfficientAD inference")
