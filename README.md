# Work in progress

## Get Started 

### Enviroment
```bash
pip install -r requirements.txt
```

### Data
1. Download [segmentation maps for VisA](https://drive.google.com/file/d/1ZVMxtb6PY958qigxAQcLEifWsRdnLaI4/view?usp=sharing).
2. Download [anomaly maps for EfficientAD](https://drive.google.com/file/d/1mknzBIE6Heqfr5_BQIFOojzuPDQG2o_O/view?usp=sharing).
3. Data structure should look as following:
```shell
data
|-- visa_segm
|-- efficient_ad_output
```

### Train and evaluate
Only VisA dataset and anomaly detector EffcientAD are available for now.
```bash
python main.py
```