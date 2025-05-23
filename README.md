OCT Macular Edema Classification

This repository contains the Jupyter Notebooks to benchmark deep learning models for identifying causes of macular edema (ME) using Optical Coherence Tomography (OCT) images.

| Notebook | Description |
|----------|-------------|
| `kauh-dataset-models-benchmarking.ipynb` | Benchmarking ResNet152, InceptionV3, and MobileNetV2 on a private dataset from King Abdullah University Hospital (KAUH, 2015–2024). |
| `public-dataset-models-benchmarking.ipynb` | Replicates the benchmarking using a publicly available OCT dataset from Kaggle. |
| `xai-performance-on-resnet152.ipynb` | Applies Grad-CAM explainability to ResNet152 outputs, with manual comparison to annotated OCT images. |

Public dataset: Link to source https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8

Private KAUH dataset: The dataset contains retrospective OCT scans and associated patient demographic information. This data is not publicly shareable due to the following reasons:
-It contains sensitive medical information that is protected under privacy laws and ethical research standards.
-The dataset was collected under an IRB (Institutional Review Board) approval that restricts its use to this specific research project only.
-Informed consent for open data sharing was not obtained from patients at the time of data collection

• All experiments in this study were carried out on Kaggle platform with GPU acceleration enabled.

• Dependencies: 
-Python 3.8+
-TensorFlow 2.x
-NumPy
-Pandas
-Albumentations
-Matplotlib
-OpenCV
-Scikit-learn

To install them:
!pip install tensorflow numpy pandas opencv-python matplotlib tqdm scikit-learn albumentations

• Evaluation Metrics:
-Accuracy 
-Precision
-Recall 
-F1-score
-Confusion Matrix

• Citation:
If you use this code, please cite the following :
Sondos Momany, AI-based Identification of Macular Edema Causes in OCT Scans, 2025.
