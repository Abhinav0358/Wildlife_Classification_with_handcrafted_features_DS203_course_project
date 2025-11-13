# Wildlife Detection (Block-Level Classifier)

This repository contains a pipeline for detecting wildlife in aerial images by operating on an **8×8 grid** over each image. The system uses classical computer-vision features (HOG, LBP, and color histograms) combined with a Random Forest classifier. Optional GPU acceleration is supported through OpenCV CUDA for preprocessing.

---

## How the System Works

### 1. Preprocessing
Each image is converted to LAB color space and CLAHE is applied on the L-channel.  
If CUDA is available, CLAHE runs on the GPU; otherwise it falls back to CPU.

### 2. Block Extraction
Images are expected to be of size **800×600**.  
Each image is divided into **64 blocks** in an 8×8 layout.

### 3. Feature Extraction
For each block, the following features are computed:

- HOG descriptor  
- LBP histogram (uniform pattern)  
- BGR color histograms  

All features are concatenated into a single feature vector.

### 4. Sliding Window Construction
Rather than classify blocks independently, the code constructs **3×3 sliding windows** across the grid (49 windows per image).  
Each window is assigned a label based on the average of its 9 block labels.

### 5. Model Training
All window features from all training images are scaled and used to train a  
`RandomForestClassifier` (300 trees, class-balanced).  
The trained model and the scaler are stored using `joblib`.

### 6. Prediction
During inference:

1. Features are extracted exactly as done for training.  
2. The classifier outputs a probability for every 3×3 window.  
3. These probabilities are mapped back onto the 8×8 grid using **max-pooling**, so each block receives its final score.  
4. Heatmaps and binary prediction masks are generated and saved.

---

## Usage

### Train the Model
```bash
python main.py --mode train
