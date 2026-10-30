\# FabricGuard AI – Intelligent Textile Defect Detection \& Quality Assessment System



An end-to-end Deep Learning and Computer Vision project that detects potential textile defects from fabric images using \*\*Convolutional Neural Networks (CNNs)\*\* built with TensorFlow/Keras. The project uses the \*\*AITEX Fabric Image Database\*\*, applies image preprocessing and patch-level classification, evaluates model performance, and provides an interactive \*\*Streamlit dashboard\*\* for fabric quality assessment.



\---



\## 1. Overview



Textile quality inspection is an important part of the textile manufacturing process. Traditional manual inspection can be time-consuming and may produce inconsistent results.



FabricGuard AI explores how deep learning and computer vision can support automated textile defect screening.



The system divides fabric images into smaller patches and classifies each patch as either:



\* \*\*Normal:\*\* No defect detected by the model.

\* \*\*Defect:\*\* Potential fabric defect detected by the model.



> \*\*Accurate Scope Definition:\*\* FabricGuard AI is an experimental, patch-level textile defect classification system. It classifies fabric patches as normal or defective. It does not perform pixel-level defect segmentation or guarantee precise defect localization.



\---



\## 2. Problem Statement



Manual textile inspection requires considerable human effort and may be affected by inspection time, lighting conditions, and individual judgment.



Small fabric defects can be difficult to identify consistently, especially when inspecting large fabric surfaces.



The project investigates whether a CNN-based image classification pipeline can assist textile quality inspection by automatically identifying fabric patches that may contain defects.



\---



\## 3. Objectives



\* Develop an end-to-end textile defect classification pipeline.

\* Explore and analyze the AITEX Fabric Image Database.

\* Preprocess large fabric images into smaller image patches.

\* Generate binary labels using defect masks.

\* Prevent data leakage through source-image-level dataset splitting.

\* Develop and train CNN-based classification models.

\* Evaluate model performance using accuracy, precision, recall, F1-score, PR-AUC, and confusion matrices.

\* Build an interactive Streamlit application for fabric image analysis.

\* Demonstrate the potential application of deep learning in textile quality assessment.



\---



\## 4. Key Features



\* \*\*End-to-End Pipeline:\*\* Dataset auditing, EDA, preprocessing, model training, evaluation, and deployment.

\* \*\*Fabric Image Processing:\*\* Converts large 4096 × 256 fabric images into smaller 256 × 256 patches.

\* \*\*Mask-Based Labelling:\*\* Uses defect masks to identify patches containing defect pixels.

\* \*\*Leakage Prevention:\*\* Splits data at the original source-image level rather than randomly splitting individual patches.

\* \*\*CNN Model Development:\*\* Implements two CNN architectures for binary textile defect classification.

\* \*\*Model Evaluation:\*\* Generates classification reports, confusion matrices, and performance metrics.

\* \*\*Interactive Dashboard:\*\* Upload fabric images and view patch-level predictions.

\* \*\*Prediction Visualization:\*\* Displays normal and potentially defective patch counts, prediction probabilities, and downloadable results.



\---



\## 5. Dataset



This project uses the \*\*AITEX Fabric Image Database\*\* for textile defect detection research.



The dataset contains fabric images, corresponding defect images, and defect masks.



\### Dataset Composition



| Dataset Category  | Number of Images |

| ----------------- | ---------------: |

| Defect Images     |              106 |

| Normal Images     |              141 |

| Mask Images       |              107 |

| Total Image Files |              354 |



\### Image Properties



\* \*\*Original Image Resolution:\*\* Approximately 4096 × 256 pixels.

\* \*\*Image Format:\*\* PNG.

\* \*\*Normal Images:\*\* Fabric samples without annotated defects.

\* \*\*Defect Images:\*\* Fabric samples containing defects.

\* \*\*Mask Images:\*\* Corresponding defect annotations.



\### Dataset Auditing



Before model development, the dataset was audited to check:



\* Image dimensions and file availability.

\* Defect and normal image counts.

\* Corresponding mask availability.

\* Multiple masks associated with individual defect images.

\* Defect-mask alignment and positive mask pixels.



One defect image, `0100\_025\_08.png`, did not have a corresponding mask and was excluded from preprocessing.



Two defect images contained multiple masks, which were merged during preprocessing.



\### Dataset After Preprocessing



| Category                       | Count |

| ------------------------------ | ----: |

| Defect Source Images Processed |   105 |

| Normal Source Images Processed |   141 |

| Total Source Images Processed  |   246 |

| Total Generated Patches        | 3,935 |

| Defect Patches                 |   185 |

| Normal Patches                 | 3,750 |



The defect and normal patches were generated from the original fabric images.



\### Patch Label Distribution



| Patch Category       | Normal (0) | Defect (1) |

| -------------------- | ---------: | ---------: |

| Defect Source Images |      1,494 |        185 |

| Normal Source Images |      2,256 |          0 |

| \*\*Total\*\*            |  \*\*3,750\*\* |    \*\*185\*\* |



> \*\*Important:\*\* The dataset is highly imbalanced. Only 185 out of 3,935 patches contain annotated defect pixels. This imbalance is an important consideration when interpreting model performance.



\---



\## 6. Project Workflow



```text

&#x20;       AITEX Fabric Image Database

&#x20;                   |

&#x20;                   v

&#x20;            Dataset Auditing

&#x20;                   |

&#x20;                   v

&#x20;      Exploratory Data Analysis

&#x20;                   |

&#x20;                   v

&#x20;         Image Preprocessing

&#x20;                   |

&#x20;                   v

&#x20;      Defect Mask-Based Labelling

&#x20;                   |

&#x20;                   v

&#x20;      256 x 256 Patch Generation

&#x20;                   |

&#x20;                   v

&#x20;      Source-Level Dataset Splitting

&#x20;                   |

&#x20;                   v

&#x20;            CNN Model V1

&#x20;                   |

&#x20;                   v

&#x20;            CNN Model V2

&#x20;                   |

&#x20;                   v

&#x20;         Model Performance Evaluation

&#x20;                   |

&#x20;                   v

&#x20;          Threshold Analysis

&#x20;                   |

&#x20;                   v

&#x20;         Trained Model Saving

&#x20;                   |

&#x20;                   v

&#x20;         Streamlit Dashboard

&#x20;                   |

&#x20;                   v

&#x20;      Fabric Patch Classification

```



\---



\## 7. Image Preprocessing



The original AITEX fabric images are wide panoramic images. To make them suitable for CNN training, they are divided into smaller patches.



\### Preprocessing Steps



1\. Load defect and normal fabric images.

2\. Read corresponding defect masks.

3\. Convert masks into grayscale luminance values.

4\. Merge multiple masks where available.

5\. Convert positive mask pixels into binary defect annotations.

6\. Divide fabric images into 256 × 256 patches.

7\. Pad incomplete patches when necessary.

8\. Assign binary labels based on defect-mask pixels.

9\. Save processed patches and their labels.

10\. Generate a CSV file containing patch metadata.



\### Patch Labelling Logic



\* \*\*Label 0:\*\* Patch contains no annotated defect pixels.

\* \*\*Label 1:\*\* Patch contains annotated defect pixels.



The preprocessing pipeline generates:



```text

data/

└── processed/

&#x20;   └── patches/

&#x20;       ├── images/

&#x20;       ├── masks/

&#x20;       └── patch\_labels.csv

```



\---



\## 8. Dataset Splitting



To prevent data leakage, the dataset is divided at the original source-image level.



This ensures that patches originating from the same fabric image cannot appear in different dataset splits.



\### Split Configuration



| Dataset    | Source Images |   Patches |

| ---------- | ------------: | --------: |

| Training   |           172 |     2,752 |

| Validation |            37 |       592 |

| Testing    |            37 |       591 |

| \*\*Total\*\*  |       \*\*246\*\* | \*\*3,935\*\* |



\### Split Strategy



\* Training: 70%

\* Validation: 15%

\* Testing: 15%

\* Random State: 42



The split uses source-image grouping and stratification by source category.



A leakage verification check was performed and passed.



\---



\## 9. CNN Model Architecture



Two CNN models were developed to investigate textile defect classification performance.



\### CNN V1



The first model uses a relatively lightweight CNN architecture.



| Layer                  | Configuration      | Purpose                        |

| ---------------------- | ------------------ | ------------------------------ |

| Input                  | 128 × 128 × 3      | Receives RGB image             |

| Conv2D                 | 32 filters, 3 × 3  | Extracts basic visual features |

| MaxPooling2D           | 2 × 2              | Reduces spatial dimensions     |

| Conv2D                 | 64 filters, 3 × 3  | Learns intermediate features   |

| MaxPooling2D           | 2 × 2              | Downsampling                   |

| Conv2D                 | 128 filters, 3 × 3 | Learns complex patterns        |

| MaxPooling2D           | 2 × 2              | Downsampling                   |

| GlobalAveragePooling2D | -                  | Reduces feature maps           |

| Dense                  | 64 neurons         | Feature learning               |

| Dropout                | 0.4                | Regularization                 |

| Output                 | 1 neuron, Sigmoid  | Binary classification          |



\### CNN V2



The second model introduces additional convolutional blocks and batch normalization.



| Layer                  | Configuration     | Purpose                         |

| ---------------------- | ----------------- | ------------------------------- |

| Input                  | 128 × 128 × 3     | RGB image input                 |

| Conv Block 1           | 32 filters        | Basic feature extraction        |

| Conv Block 2           | 64 filters        | Intermediate feature extraction |

| Conv Block 3           | 128 filters       | Complex feature extraction      |

| Conv Block 4           | 256 filters       | Higher-level feature extraction |

| Batch Normalization    | After convolution | Training stabilization          |

| ReLU                   | Activation        | Non-linearity                   |

| MaxPooling2D           | 2 × 2             | Spatial downsampling            |

| GlobalAveragePooling2D | -                 | Feature aggregation             |

| Dense                  | 128 neurons, ReLU | Classification features         |

| Dropout                | 0.4               | Regularization                  |

| Output                 | 1 neuron, Sigmoid | Defect probability              |



\### Training Configuration



| Parameter        |              CNN V1 |              CNN V2 |

| ---------------- | ------------------: | ------------------: |

| Input Size       |           128 × 128 |           128 × 128 |

| Batch Size       |                  32 |                  32 |

| Maximum Epochs   |                  25 |                  15 |

| Optimizer        |                Adam |                Adam |

| Learning Rate    |               0.001 |              0.0003 |

| Loss Function    | Binary Crossentropy | Binary Crossentropy |

| Class Weighting  |            Balanced |            Balanced |

| Early Stopping   |                 Yes |                 Yes |

| Model Checkpoint |                 Yes |                 Yes |



CNN V2 also uses data augmentation through random horizontal/vertical flipping and random contrast adjustment.



\---



\## 10. Technologies Used



| Category                | Technologies              |

| ----------------------- | ------------------------- |

| Programming Language    | Python                    |

| Deep Learning           | TensorFlow, Keras         |

| Data Processing         | NumPy, Pandas             |

| Machine Learning        | Scikit-learn              |

| Image Processing        | Pillow, OpenCV            |

| Visualization           | Matplotlib, Seaborn       |

| Dashboard               | Streamlit, Plotly         |

| Development Environment | Jupyter Notebook, VS Code |

| Version Control         | Git, GitHub               |



\---



\## 11. Project Structure



```text

FabricGuard-AI/

│

├── dashboard/

│   ├── app.py

│   └── pages/

│

├── data/

│   ├── raw/

│   ├── processed/

│   │   └── patches/

│   │       ├── images/

│   │       ├── masks/

│   │       └── patch\_labels.csv

│   └── sample\_images/

│

├── models/

│   ├── fabricguard\_cnn.keras

│   └── fabricguard\_cnn\_v2.keras

│

├── notebooks/

│

├── outputs/

│   ├── figures/

│   ├── reports/

│   └── metrics/

│       ├── test\_metrics.json

│       ├── tuned\_test\_metrics.json

│       ├── confusion\_matrix.csv

│       └── selected\_threshold.txt

│

├── src/

│   ├── preprocessing.py

│   ├── split\_dataset.py

│   ├── train\_cnn.py

│   ├── tune\_threshold.py

│   ├── evaluate\_tuned\_model.py

│   └── train\_cnn\_v2.py

│

├── tests/

│

├── .gitignore

├── README.md

└── requirements.txt

```



\---



\## 12. Installation \& Setup



\### Step 1: Clone the Repository



```bash

git clone https://github.com/swagata-dsai/FabricGuard-AI.git

cd FabricGuard-AI

```



\### Step 2: Create a Virtual Environment



Windows PowerShell:



```powershell

python -m venv .venv

```



\### Step 3: Activate the Environment



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks script execution:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\\.venv\\Scripts\\Activate.ps1

```



\### Step 4: Install Dependencies



```bash

pip install -r requirements.txt

```



\---



\## 13. Dataset Setup



The AITEX dataset is not automatically downloaded by cloning this repository.



Download and extract the AITEX Fabric Image Database separately.



The preprocessing scripts expect the dataset to be organized into these directories:



```text

AITEX\_Dataset/

│

├── Defect\_images/

├── NODefect\_images/

└── Mask\_images/

```



Update the dataset directory configuration in the preprocessing script if your local dataset is stored in a different location.



After dataset setup, run the preprocessing pipeline to generate image patches and labels.



> Large raw datasets and generated patch images are excluded from GitHub through `.gitignore`. Users must prepare the dataset locally before running the complete training pipeline.



\---



\## 14. Model Training \& Evaluation



\### Run Image Preprocessing



```powershell

python src/preprocessing.py

```



This generates the image patches, masks, and patch label CSV.



\### Run Dataset Splitting



```powershell

python src/split\_dataset.py

```



This creates source-level training, validation, and testing splits.



\### Train CNN V1



```powershell

python src/train\_cnn.py

```



The trained model is saved as:



```text

models/fabricguard\_cnn.keras

```



\### Train CNN V2



```powershell

python src/train\_cnn\_v2.py

```



The trained model is saved as:



```text

models/fabricguard\_cnn\_v2.keras

```



\### Tune Classification Threshold



```powershell

python src/tune\_threshold.py

```



This selects a classification threshold using validation data.



\### Evaluate the Tuned Model



```powershell

python src/evaluate\_tuned\_model.py

```



The evaluation script measures performance on the held-out test set.



\---



\## 15. Streamlit Web Application



FabricGuard AI includes an interactive Streamlit dashboard for testing fabric images.



\### Run the Dashboard



From the project root:



```powershell

streamlit run dashboard/app.py

```



The application opens in a browser at:



```text

http://localhost:8501

```



\### Dashboard Features



\* Upload fabric images in PNG, JPG, JPEG, or BMP format.

\* Automatically divide uploaded images into 256 × 256 patches.

\* Resize patches to the CNN input size of 128 × 128.

\* Generate patch-level defect probabilities.

\* Display normal and potentially defective patch counts.

\* View prediction results in a table.

\* Visualize prediction probabilities.

\* Download prediction results as a CSV file.



\### Prediction Output



The dashboard provides:



| Output             | Description                                 |

| ------------------ | ------------------------------------------- |

| Total Patches      | Number of generated image patches           |

| Normal Patches     | Patches classified as normal                |

| Defect Patches     | Patches classified as potentially defective |

| Defect Probability | Model-generated defect score                |

| Classification     | Normal or Defect                            |



> \*\*Important:\*\* Predictions are experimental screening results. A defect classification indicates a potentially defective patch, not a verified manufacturing defect.



\---



\## 16. Actual Model Performance



The project evaluates two CNN models on the same held-out test split containing 591 patches.



\### CNN V1 – Test Results



At the default classification threshold of 0.50:



| Metric    | Measured Value |

| --------- | -------------: |

| Accuracy  |         65.48% |

| Precision |          8.92% |

| Recall    |         65.52% |

| F1-Score  |         15.70% |

| PR-AUC    |         10.09% |



\*\*CNN V1 Confusion Matrix\*\*



```text

&#x20;                Predicted

&#x20;             Normal  Defect

Actual Normal   368     194

Actual Defect    10      19

```



\### CNN V2 – Test Results



At the default classification threshold of 0.50:



| Metric    | Measured Value |

| --------- | -------------: |

| Accuracy  |         81.05% |

| Precision |         11.93% |

| Recall    |         44.83% |

| F1-Score  |         18.84% |

| PR-AUC    |         11.31% |



\*\*CNN V2 Confusion Matrix\*\*



```text

&#x20;                Predicted

&#x20;             Normal  Defect

Actual Normal   466      96

Actual Defect    16      13

```



\### Model Comparison



| Metric    | CNN V1 | CNN V2 |

| --------- | -----: | -----: |

| Accuracy  | 65.48% | 81.05% |

| Precision |  8.92% | 11.93% |

| Recall    | 65.52% | 44.83% |

| F1-Score  | 15.70% | 18.84% |

| PR-AUC    | 10.09% | 11.31% |



\### Performance Interpretation



CNN V2 improved accuracy, precision, F1-score, and PR-AUC compared with CNN V1 at the default threshold.



However, defect detection performance remains limited:



\* CNN V2 detects 13 out of 29 actual defect patches.

\* Its defect precision is approximately 11.93%.

\* The model still produces a considerable number of false-positive predictions.



The results indicate that the model has learned some defect-related visual patterns, but its classification performance is not yet reliable enough for production use.



\---



\## 17. Classification Threshold Experiment



A threshold-tuning experiment was conducted for CNN V1.



The threshold was selected using validation data rather than the test set.



\### Selected Validation Threshold



```text

Selected Threshold: 0.6703

```



\### Validation Results



| Metric    |  Value |

| --------- | -----: |

| Precision | 84.62% |

| Recall    | 34.38% |

| F1-Score  | 48.89% |

| PR-AUC    | 46.42% |



\### Held-Out Test Results at the Selected Threshold



| Metric    |  Value |

| --------- | -----: |

| Accuracy  | 93.40% |

| Precision |  8.33% |

| Recall    |  3.45% |

| F1-Score  |  4.88% |

| PR-AUC    | 10.09% |



\### Key Observation



The validation-selected threshold did not generalize well to the held-out test data.



Although the validation precision improved, test recall dropped substantially.



This experiment highlights the importance of evaluating threshold selection on independent data and not relying on accuracy alone when dealing with highly imbalanced datasets.



\---



\## 18. Limitations



1\. \*\*Class Imbalance:\*\* The dataset contains substantially fewer defect patches than normal patches.

2\. \*\*Limited Dataset Size:\*\* Only 246 source images were processed.

3\. \*\*Patch-Level Classification:\*\* The model identifies potentially defective patches but does not produce pixel-level defect segmentation.

4\. \*\*False Positives:\*\* The CNN V2 model produces many false-positive defect predictions.

5\. \*\*Limited Generalization:\*\* Performance on new fabric types, textures, and imaging conditions has not been established.

6\. \*\*Experimental Prototype:\*\* The system is intended for academic demonstration and further research, not autonomous industrial quality control.



\---



\## 19. Future Improvements



\* \*\*Larger Dataset:\*\* Train and validate using more diverse textile images.

\* \*\*Class Imbalance Handling:\*\* Investigate focal loss, improved sampling strategies, and additional defect examples.

\* \*\*Advanced CNN Architectures:\*\* Experiment with transfer learning and lightweight computer vision models.

\* \*\*Defect Segmentation:\*\* Implement U-Net or similar segmentation architectures for pixel-level localization.

\* \*\*Explainable AI:\*\* Integrate Grad-CAM to visualize image regions influencing predictions.

\* \*\*Real-Time Inspection:\*\* Explore camera-based fabric inspection.

\* \*\*Model Optimization:\*\* Improve precision and recall through systematic validation-based experimentation.

\* \*\*Industrial Testing:\*\* Evaluate the model on independent real-world fabric samples.



\---



\## 20. Author \& Academic Information



\* \*\*Project Name:\*\* FabricGuard AI – Intelligent Textile Defect Detection \& Quality Assessment System

\* \*\*Domain:\*\* Deep Learning / Computer Vision / Artificial Intelligence

\* \*\*Dataset:\*\* AITEX Fabric Image Database

\* \*\*Models:\*\* Convolutional Neural Networks (CNN V1 and CNN V2)

\* \*\*Framework:\*\* TensorFlow / Keras

\* \*\*Deployment:\*\* Streamlit

\* \*\*Author:\*\* Swagata Das

\* \*\*Academic Program:\*\* MBA – Data Science and Artificial Intelligence

\* \*\*Institution:\*\* Chandigarh University



\---



\## 21. GitHub Repository



\*\*Project Repository:\*\*

https://github.com/swagata-dsai/FabricGuard-AI



\---



\*FabricGuard AI is an academic deep learning project demonstrating the application of computer vision to textile defect screening and quality assessment.\*



