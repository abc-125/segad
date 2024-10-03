# Supervised Anomaly Detection for Complex Industrial Images
Official code for our CVPR 2024 [paper](https://arxiv.org/abs/2405.04953)

[VAD repository](https://github.com/abc-125/vad)

## Get Started 

### Enviroment
```bash
pip install -r requirements.txt
```

### Data
1. Download [segmentation maps for VisA](https://drive.google.com/file/d/1ZVMxtb6PY958qigxAQcLEifWsRdnLaI4/view?usp=sharing).
2. Download [anomaly maps for EfficientAD](https://drive.google.com/file/d/1mknzBIE6Heqfr5_BQIFOojzuPDQG2o_O/view?usp=sharing).
3. Download [anomaly maps for RD4AD](https://drive.google.com/file/d/1Pap5-8x74_AROFRxjcBvIu9XdqvzHMs8/view?usp=sharing).
4. Data structure should look as following:
```shell
data
|-- visa_segm
|-- anomaly_maps
|-----|--efficient_ad
|-----|--rd4ad
```

### Train and evaluate
Only VisA dataset is available for now. List of available models: `["efficient_ad", "rd4ad", "all_ad"]`. `"all_ad"` includes both EfficientAD and RD4AD.
```bash
python main.py --model efficient_ad
```

## Results
Cl. AUROC (image-level) for SegAD with different sources of anomaly maps.

|   model            |  mean  | candle | capsules  | cashew | chewinggum  | fryum  | macaroni1 | macaroni2 | pcb1 | pcb2 | pcb3 | pcb4 | pipe_fryum |
| -------------------| :----: | :----: | :-------: | :----: | :---------: | :----: | :-------: | :-------: | :---:| :---:| :---:| :---:| :--------: |
| RD4AD + SegAD      | 95.3   | 98.5   | 80.2      | 98.9   | 99.4        | 96.1   | 97.4      | 90.7      | 96.4 | 96.3 | 94.1 | 99.9 | 95.8       |
| EfficientAD + SegAD| 98.3   | 98.7   | 89.7      | 98.6   | 99.9        | 98.6   | 99.5      | 98.1      | 99.5 | 99.7 | 98.4 | 99.3 | 99.2       |
| All AD + SegAD     | 98.4   | 99.0   | 90.7      | 99.0   | 99.9        | 98.5   | 99.4      | 98.1      | 99.2 | 99.7 | 98.3 | 99.8 | 99.1       |

## Acknowledgement

We use [EfficientAD](https://github.com/nelson1425/EfficientAD) and [Anomalib](https://github.com/openvinotoolkit/anomalib/tree/main) for baseline anomaly detection models. We are thankful for their amazing work!

## Citation
Please cite this paper if it helps your project:
```
@inproceedings{baitieva2024supervised,
      title={Supervised Anomaly Detection for Complex Industrial Images}, 
      author={Aimira Baitieva and David Hurych and Victor Besnier and Olivier Bernard},
      booktitle={CVPR},
      year={2024} 
}
```
