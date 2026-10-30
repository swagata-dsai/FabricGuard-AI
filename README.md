\# FabricGuard AI



\### An Intelligent Textile Defect Detection \& Automated Quality Assessment System



!\[Python](https://img.shields.io/badge/Python-3.13-blue)

!\[TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)



\## 1. Project Overview



FabricGuard AI is an AI-powered textile quality assessment system that uses Convolutional Neural Networks (CNNs) to classify fabric image patches as normal or defective.



The project uses the AITEX Fabric Image Database to develop an image preprocessing pipeline, train deep learning models, evaluate classification performance, and provide an interactive Streamlit dashboard.



\## 2. Problem Statement \& Objectives



Traditional textile inspection often depends on manual observation, which can be time-consuming and inconsistent.



Objectives:



\* Analyse fabric images using computer vision.

\* Extract image patches and generate defect labels.

\* Train CNN-based classification models.

\* Evaluate model performance.

\* Develop an interactive prediction dashboard.



\## 3. Key Features



\* Automated fabric image preprocessing.

\* Defect-mask-based patch labelling.

\* CNN-based binary classification.

\* Source-image-level dataset splitting.

\* Model performance evaluation.

\* Interactive Streamlit dashboard.

\* Patch-level prediction reports and CSV export.



\## 4. Dataset Description



Dataset: AITEX Fabric Image Database.



| Component             | Images |

| --------------------- | -----: |

| Defect Images         |    106 |

| Normal Images         |    141 |

| Mask Images           |    107 |

| Total Extracted Files |    354 |



\### Processed Dataset



| Category                       | Count |

| ------------------------------ | ----: |

| Defect source images processed |   105 |

| Normal source images processed |   141 |

| Total source images            |   246 |

| Total patches                  | 3,935 |

| Defect patches                 |   185 |

| Normal patches                 | 3,750 |



Images are divided into 256 × 256 patches. Defect masks are processed to generate binary patch-level labels.



\*\*Note:\*\* This project performs patch-level classification, not pixel-level segmentation.



\## 5. Project Workflow



1\. Dataset Collection

2\. Exploratory Data Analysis

3\. Image Preprocessing

4\. Patch Extraction

5\. Mask Processing

6\. Label Generation

7\. Dataset Splitting

8\. CNN Model Development

9\. Model Training

10\. Model Evaluation

11\. Streamlit Dashboard



\## 6. CNN Architecture



Two CNN versions were developed.



\### CNN V1



\* Conv2D: 32 filters

\* MaxPooling2D

\* Conv2D: 64 filters

\* MaxPooling2D

\* Conv2D: 128 filters

\* MaxPooling2D

\* Global Average Pooling

\* Dense: 64

\* Dropout: 0.4

\* Sigmoid Output



\### CNN V2



\* Four convolutional blocks: 32, 64, 128 and 256 filters

\* Batch Normalization

\* ReLU Activation

\* MaxPooling2D

\* Global Average Pooling

\* Dense: 128

\* Dropout: 0.4

\* Sigmoid Output



Training uses Adam optimizer, binary cross-entropy, class weights, data augmentation, early stopping and learning-rate reduction.



\## 7. Technologies Used



| Technology           | Purpose                   |

| -------------------- | ------------------------- |

| Python               | Programming               |

| TensorFlow / Keras   | Deep Learning             |

| NumPy                | Numerical Computing       |

| Pandas               | Data Processing           |

| Matplotlib / Seaborn | Visualization             |

| Scikit-learn         | Model Evaluation          |

| Pillow / OpenCV      | Image Processing          |

| Streamlit            | Dashboard                 |

| Plotly               | Interactive Visualization |

| Git / GitHub         | Version Control           |



\## 8. Project Structure



```text

FabricGuard-AI/

├── dashboard/

│   └── app.py

├── data/

│   ├── raw/

│   └── processed/

├── models/

│   ├── fabricguard\_cnn.keras

│   └── fabricguard\_cnn\_v2.keras

├── notebooks/

├── outputs/

│   ├── figures/

│   ├── reports/

│   └── metrics/

├── src/

│   ├── preprocessing.py

│   ├── split\_dataset.py

│   ├── train\_cnn.py

│   ├── train\_cnn\_v2.py

│   ├── tune\_threshold.py

│   └── evaluate\_tuned\_model.py

├── tests/

├── .gitignore

├── README.md

└── requirements.txt

```



\## 9. Installation \& Setup



\### Clone Repository



```bash

git clone https://github.com/swagata-dsai/FabricGuard-AI.git

cd FabricGuard-AI

```



\### Create Virtual Environment



```bash

python -m venv .venv

```



\### Activate Environment (Windows)



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### Install Dependencies



```bash

pip install -r requirements.txt

```



\### Run Preprocessing



```bash

python src/preprocessing.py

```



\### Split Dataset



```bash

python src/split\_dataset.py

```



\### Train CNN V2



```bash

python src/train\_cnn\_v2.py

```



\### Launch Dashboard



```bash

streamlit run dashboard/app.py

```



\## 10. Model Training \& Evaluation



The dataset is split at source-image level to reduce data leakage.



| Split      | Source Images | Patches |

| ---------- | ------------: | ------: |

| Training   |           172 |   2,752 |

| Validation |            37 |     592 |

| Testing    |            37 |     591 |



Evaluation metrics include Accuracy, Precision, Recall, F1-score, PR-AUC and Confusion Matrix.



\## 11. Actual Model Results



\### CNN V1 — Test Results



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 65.48% |

| Precision |  8.92% |

| Recall    | 65.52% |

| F1-score  | 15.70% |

| PR-AUC    | 10.09% |



\### CNN V2 — Test Results



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 81.05% |

| Precision | 11.93% |

| Recall    | 44.83% |

| F1-score  | 18.84% |

| PR-AUC    | 11.31% |



\### CNN V2 Confusion Matrix



| Actual / Predicted | Normal | Defect |

| ------------------ | -----: | -----: |

| Normal             |    466 |     96 |

| Defect             |     16 |     13 |



CNN V2 improved accuracy, F1-score and PR-AUC compared with CNN V1 at the default threshold of 0.50.



However, defect precision remains low. The model requires further improvement and validation before real-world deployment.



\## 12. Streamlit Dashboard



The interactive dashboard supports:



\* Fabric image upload.

\* Automatic 256 × 256 patch extraction.

\* CNN-based patch classification.

\* Normal and defect patch counts.

\* Prediction probability display.

\* Patch-level results table.

\* CSV report download.



Run the application:



```bash

streamlit run dashboard/app.py

```



\## 13. Threshold Tuning Experiment



CNN V1 threshold was selected using validation data by maximizing F1-score.



| Metric                 | Result |

| ---------------------- | -----: |

| Selected Threshold     | 0.6703 |

| Validation Precision   | 84.62% |

| Validation Recall      | 34.38% |

| Validation F1-score    | 48.89% |

| Held-out Test F1-score |  4.88% |



The validation result did not generalize well to the independent test set. This demonstrates the importance of independent model evaluation.



\## 14. Limitations



\* Limited number of defective patches.

\* Significant class imbalance.

\* Low defect precision.

\* Performance may vary on unseen fabric types.

\* Patch classification does not provide pixel-level segmentation.

\* Further validation is required.



\## 15. Future Improvements



\* Experiment with transfer learning.

\* Explore advanced CNN architectures.

\* Investigate focal loss and imbalance-handling techniques.

\* Develop U-Net-based defect segmentation.

\* Improve defect precision and recall.

\* Test on independent real-world fabric images.



\## 16. Author



\*\*Swagata Das\*\*



MBA — Data Science and Artificial Intelligence

Chandigarh University



GitHub: https://github.com/swagata-dsai



\---



\*\*Disclaimer:\*\* FabricGuard AI is an academic prototype developed for learning and research. Current model performance does not establish production readiness.



