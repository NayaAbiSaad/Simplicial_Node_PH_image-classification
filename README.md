# Simplicial_Node_PH_image-classification
Code for superpixel-based medical image classification using simplicial complexes, node features, and persistent homology, evaluated with Random Forest and XGBoost on brain MRI, breast MRI, and melanoma image datasets.

This repository contains the code associated with the manuscript “Superpixel-based persistent homology for topology-aware medical image classification.”
The study investigates the use of superpixel-derived features and persistent homology for binary classification of medical images using conventional machine learning models.

**Datasets**
The experiments were conducted on three medical image datasets:
Brain tumor MRI: 400 images
Breast cancer MRI: 1,400 images
Melanoma dermoscopic images: approximately 10,000 images
The original image datasets are not included in this repository.

The datasets used in this study are publicly available from their original sources:

The Breast Cancer Patients MRI’s https://www.kaggle.com/datasets/uzairkhan45/breast-cancer-patients-mris
The Brain MRI Images https://www.kaggle.com/datasets/mhantor/mri-based-brain-tumor-images?select=Brain_tumor_images
The Melanoma Skin Cancer Dataset of 10000” https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

**Methodology**
The pipeline consists of:
Image preprocessing and superpixel segmentation.
Construction of a superpixel-based simplicial complex using TopoNetX.
Extraction of superpixel-derived node features.
Computation of persistent homology descriptors.
Construction of different feature sets:
Raw features
Node features
Node + persistent homology (PH) features
Raw + node + PH features
Classification using:
Random Forest (RF)
XGBoost (XGB)
The feature sets are compared using AUC and other classification metrics.

**Software**
The main Python libraries used in the study include:
Python
NumPy
scikit-learn
XGBoost
TopoNetX
GUDHI
scikit-image
Specific package versions are provided in the corresponding requirements/environment file where applicable.
Python version when the code was executed: 3.12.12

**Repository Structure**
```text
.
├── code/ 
├── README.md
└── requirements.txt
```
The exact organization may vary depending on the released version of the code.

**Reproducibility**
The code is provided to support the reproducibility of the methods and experiments described in the manuscript. The datasets themselves are not redistributed here. Users should obtain the corresponding datasets from their original sources and adapt the dataset paths in the scripts as required.

*Citation*
If you use this code or methodology in your research, please cite the associated manuscript:
> N. Abi Saad et al., “Superpixel-based persistent homology for topology-aware medical image classification.”
A DOI will be added here when available.
