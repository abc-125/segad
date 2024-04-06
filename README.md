# Work in progress

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
Only VisA dataset is available for now.
```bash
python main.py --model efficient_ad
```