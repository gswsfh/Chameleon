### Codes for paper "Chameleon: Malicious Traffic Detection Based on Dynamic Baselines"

#### Environment Installation
```
pip install -r requirement
```

### File Description
In the project, files prefixed with sbem are primarily used for model-related components:
- sbemEmbed.py implements pattern embedding learning,
- sbemCon.py handles graph contrastive embedding learning, and
- sbemClass.py is responsible for classification.

### Dataset Description
The `cache` file provides a subset of sample data from the CIC-IDS2018 dataset.

### Model Description
We provide the trained model weight files `models` (not the full-size weights due to data privacy constraints). For CIC-IDS2018, we present the optimal detection rules derived from multi-day data, achieving an F1 score of 100% across all evaluated days.

### Traffic Collection 
The `traffic_analysis` directory contains code for capturing network traffic related to Google Search.
