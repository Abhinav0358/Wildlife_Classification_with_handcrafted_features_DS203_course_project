import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Configuration 
TRAIN_FOLDER = "train_images"
TEST_FOLDER = "test_images"
TRAIN_XL = "1_400_annotated.xlsx"
TEST_XL = "400_465_annotated.xlsx"
MODEL_FILE = "rf_3x3_last.pkl"
SCALER_FILE = "scaler_last.pkl"
PRED_DIR = "predlast2_train"

GRID_H, GRID_W = 8, 8
EXPECTED_SIZE = (800, 600)  # width, height

# HOG params
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')

# LBP params
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS

# Color histogram bins/channel
COLOR_BINS = 16

# Labels & thresholds
WINDOW_POS_THRESH = 0.50  # stricter window label
BLOCK_PROB_THRESH = 0.50  # threshold for reporting window preds from prob grid

# Speed/Hardware toggles
USE_GPU = True      # will auto-fallback if CUDA not available
N_JOBS = -1         # CPU parallelism for feature extraction (use all cores)

# -------------------- Helpers --------------------
def cuda_available():
    try:
        return USE_GPU and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

def find_image_file(img_dir: str, base_name: str):
    """Return the first existing image path for a basename with common extensions."""
    for ext in ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'):
        p = os.path.join(img_dir, base_name + ext)
        if os.path.exists(p):
            return p
    return None

#  Pre-processing (GPU-aware) 
def preprocess_image(img, clahe_clip=2.0, clahe_grid=(8, 8), equalize_hsv=False):
    """
    Apply CLAHE on L channel (LAB). If CUDA is available, use GPU path for speed.
    Returns a BGR image (uint8). If preprocessing is not desired, just return img.
    """
    if img is None or img.size == 0:
        raise ValueError("Empty image to preprocess")

    if cuda_available():
        try:
            print(" Using GPU for preprocessing (cv2.cuda)")
            # Upload to GPU
            g_img = cv2.cuda_GpuMat()
            g_img.upload(img)

            # BGR -> LAB
            g_lab = cv2.cuda.cvtColor(g_img, cv2.COLOR_BGR2LAB)
            l, a, b = [cv2.cuda_GpuMat() for _ in range(3)]
            cv2.cuda.split(g_lab, [l, a, b])

            # CLAHE on L
            clahe = cv2.cuda.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
            l2 = clahe.apply(l)

            g_lab2 = cv2.cuda.merge([l2, a, b])
            g_bgr = cv2.cuda.cvtColor(g_lab2, cv2.COLOR_LAB2BGR)
            out = g_bgr.download()

            if equalize_hsv:
                # HSV V-equalization (CPU fallback—cheap)
                hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                v = cv2.equalizeHist(v)
                hsv = cv2.merge([h, s, v])
                out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return out
        except Exception as e:
            print(f" GPU preprocessing failed ({e}); falling back to CPU.")

    # CPU fallback
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    if equalize_hsv:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv2 = cv2.merge((h, s, v))
        out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    return out

#  Feature Extraction (block) 
def extract_block_features(block, hog_params=HOG_PARAMS):
    """Concatenate HOG + LBP + Color histogram for one block (BGR)."""
    if block is None or block.size == 0:
        print(" Empty block detected, returning zeros")
        return np.zeros(10, dtype=float)

    gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)

    # HOG
    hog_feat = hog(gray, **hog_params)

    # LBP
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_POINTS + 3), range=(0, LBP_N_POINTS + 2))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-8)

    # Color histograms (B, G, R)
    ch_hists = []
    for ch in range(3):
        h = cv2.calcHist([block], [ch], None, [COLOR_BINS], [0, 256]).flatten()
        if h.sum() > 0:
            h = h / (h.sum() + 1e-8)
        ch_hists.append(h)
    color_hist = np.hstack(ch_hists)

    return np.concatenate([hog_feat, lbp_hist, color_hist])

#  Image → Blocks 
def image_to_blocks(image, grid_h=GRID_H, grid_w=GRID_W):
    """Split the image into (grid_h x grid_w) blocks using exact edges (no loss)."""
    print(" Splitting image into blocks...")
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to image_to_blocks")
    H, W = image.shape[:2]
    print(f" Input image shape: {H}x{W}")

    y_edges = np.linspace(0, H, grid_h + 1, dtype=int)
    x_edges = np.linspace(0, W, grid_w + 1, dtype=int)

    blocks = []
    for i in range(grid_h):
        for j in range(grid_w):
            y0, y1 = y_edges[i], y_edges[i + 1]
            x0, x1 = x_edges[j], x_edges[j + 1]
            block = image[y0:y1, x0:x1].copy()
            if block.size > 0:
                blocks.append(block)
            else:
                print(f"⚠️ Empty block at ({i},{j})")
    print(f" Created {len(blocks)} blocks")
    if len(blocks) > 0:
        print("🔹 Example block shape:", blocks[0].shape)
    return blocks

#  Build Training Windows 
def _win_feature_from_grid(blocks_grid, i, j):
    # concatenate features of 3x3 window starting at (i,j)
    nine = [extract_block_features(blocks_grid[i+r, j+c]) for r in range(3) for c in range(3)]
    return np.concatenate(nine)

def build_windows_from_blocks(blocks, labels):
    print("🧩 Building 3x3 windows from 8x8 blocks...")
    if not isinstance(blocks, list):
        blocks = list(blocks)
    if len(blocks) != GRID_H * GRID_W:
        raise ValueError(f"build_windows_from_blocks: expected 64 blocks, got {len(blocks)}")

    # manual object grid to avoid NumPy stacking surprises
    blocks_grid = np.empty((GRID_H, GRID_W), dtype=object)
    idx = 0
    for i in range(GRID_H):
        for j in range(GRID_W):
            blocks_grid[i, j] = blocks[idx]
            idx += 1

    labels_grid = np.array(labels, dtype=int).reshape(GRID_H, GRID_W)

    # parallel extract window features (49 windows per image)
    coords = [(i, j) for i in range(GRID_H - 2) for j in range(GRID_W - 2)]
    X_list = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_win_feature_from_grid)(blocks_grid, i, j) for (i, j) in coords
    )

    y_list = []
    for i, j in coords:
        win_labels = labels_grid[i:i+3, j:j+3]
        y_list.append(1 if win_labels.mean() >= WINDOW_POS_THRESH else 0)

    print(f" Created {len(X_list)} windows | Positives: {sum(y_list)} / {len(y_list)}")
    return np.vstack(X_list), np.array(y_list, dtype=int)

#  Training Pipeline
def load_and_build_train(train_folder, train_excel):
    print("📘 Loading training data...")
    df = pd.read_excel(train_excel)
    print(f" Loaded {len(df)} training entries")
    X_all, y_all = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building training windows"):
        base = str(row['image_name']).strip()
        img_p = find_image_file(train_folder, base)
        if img_p is None:
            print(f" Missing image: {base}")
            continue

        img = cv2.imread(img_p)
        if img is None:
            print(f" Failed to read image: {img_p}")
            continue

        H, W = img.shape[:2]
        if (W, H) != EXPECTED_SIZE:
            print(f" Skipped: wrong size {img.shape}")
            continue

        print(f" Processing {img_p}")
        # Preprocess with GPU if available
        img_proc = preprocess_image(img, clahe_clip=2.0, clahe_grid=(8, 8), equalize_hsv=False)
        blocks = image_to_blocks(img_proc)

        try:
            block_labels = [int(row[f'block_{k}']) for k in range(64)]
        except Exception as e:
            print(f" Label error for {base}: {e}")
            continue

        Xw, yw = build_windows_from_blocks(blocks, block_labels)
        X_all.append(Xw)
        y_all.append(yw)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    print(f" Final training data: X={X_all.shape}, y={y_all.shape}")
    return X_all, y_all

#  Predict & Aggregate (Max Pooling)
def predict_windows_and_aggregate(img, model, scaler):
    print(" Predicting windows and aggregating probabilities...")

    # Preprocess with GPU if available
    img_proc = preprocess_image(img, clahe_clip=2.0, clahe_grid=(8, 8), equalize_hsv=False)

    # Split into 8×8 blocks
    blocks = image_to_blocks(img_proc)
    if len(blocks) != GRID_H * GRID_W:
        raise ValueError(f"predict_windows_and_aggregate: expected 64 blocks, got {len(blocks)}")

    # Manual object grid
    blocks_grid = np.empty((GRID_H, GRID_W), dtype=object)
    idx = 0
    for i in range(GRID_H):
        for j in range(GRID_W):
            blocks_grid[i, j] = blocks[idx]
            idx += 1
    print("Successfully reshaped into (8x8) grid")

    # Batch window features (parallel)
    coords = [(i, j) for i in range(GRID_H - 2) for j in range(GRID_W - 2)]
    window_feats = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_win_feature_from_grid)(blocks_grid, i, j) for (i, j) in coords
    )
    window_feats = np.vstack(window_feats)
    print(" Computed all window features:", window_feats.shape)

    # Predict window probabilities
    window_probs = model.predict_proba(scaler.transform(window_feats))[:, 1]

    # Max-pool aggregation to blocks
    block_probs = np.zeros((GRID_H, GRID_W), dtype=float)
    for (i, j), p in zip(coords, window_probs):
        for r in range(3):
            for c in range(3):
                block_probs[i+r, j+c] = max(block_probs[i+r, j+c], p)

    print("Finished aggregating predictions.")
    return block_probs

#  Visualization
def save_visualization(img_path, prob_grid, save_dir=PRED_DIR):
    print(f" Saving visualization for {img_path}")
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original: {os.path.basename(img_path)}")
    for k in range(GRID_H + 1):
        axes[0].axhline(k * (H / GRID_H), color='green', linewidth=0.6)
        axes[0].axvline(k * (W / GRID_W), color='green', linewidth=0.6)
    axes[0].axis('off')

    im = axes[1].imshow(prob_grid, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_title('Predicted Wildlife Probability (8x8)')
    for i in range(GRID_H):
        for j in range(GRID_W):
            axes[1].text(j, i, f"{prob_grid[i, j]:.2f}", ha='center', va='center', color='black')

    plt.colorbar(im, ax=axes[1])
    out = os.path.join(save_dir, os.path.basename(img_path).split('.')[0] + '_pred.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(' Saved:', out)

#  Train/Save 
def train_and_save():
    print(' Building training features...')
    X_train, y_train = load_and_build_train(TRAIN_FOLDER, TRAIN_XL)
    print(' Training samples:', X_train.shape, y_train.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        class_weight='balanced_subsample',
        random_state=42
    )
    print(' Training RandomForest...')
    clf.fit(X_train_scaled, y_train)

    joblib.dump(clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(' Model & scaler saved.')

# -------------------- Predict All --------------------
def predict_all_and_save():
    print(' Loading model & scaler...')
    clf = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    df = pd.read_excel(TEST_XL)
    all_true_windows, all_pred_windows = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Predicting test images'):
        name = str(row['image_name']).strip()
        img_p = find_image_file(TEST_FOLDER, name)
        if not img_p:
            print(f" Missing test image: {name}")
            continue

        print(f"📷 Predicting for {img_p}")
        img = cv2.imread(img_p)
        if img is None:
            print(f" Failed to read test image: {img_p}")
            continue

        prob_grid = predict_windows_and_aggregate(img, clf, scaler)

        # Apply adaptive threshold
        threshold = prob_grid.mean() + 0.5 * prob_grid.std()
        labels_grid = (prob_grid >= threshold).astype(int)
        print(f"Adaptive threshold: {threshold:.3f}")
        print(f"Marked {labels_grid.sum()} / {labels_grid.size} blocks as positive")

        # Save both heatmap & binary mask
        save_visualization(img_p, prob_grid, save_dir=PRED_DIR)
        save_visualization(img_p, labels_grid, save_dir=os.path.join(PRED_DIR, "binary"))



        # window-level metrics from block prob-grid
        block_labels = np.array([int(row[f'block_{k}']) for k in range(64)]).reshape(GRID_H, GRID_W)
        for i in range(GRID_H - 2):
            for j in range(GRID_W - 2):
                win_true = 1 if block_labels[i:i+3, j:j+3].mean() >= WINDOW_POS_THRESH else 0
                win_prob = prob_grid[i:i+3, j:j+3].mean()
                win_pred = 1 if win_prob >= BLOCK_PROB_THRESH else 0
                all_true_windows.append(win_true)
                all_pred_windows.append(win_pred)

    # ---------------- Evaluation & Reporting ----------------
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, precision_recall_curve
    )
    import seaborn as sns

    print("\n" + "="*70)
    print("MODEL EVALUATION SUMMARY (Window-Level)")
    print("="*70)

    # --- Compute basic metrics ---
    report_dict = classification_report(all_true_windows, all_pred_windows, digits=4, output_dict=True)
    cm = confusion_matrix(all_true_windows, all_pred_windows)
    accuracy = report_dict["accuracy"]
    macro_avg = report_dict["macro avg"]
    weighted_avg = report_dict["weighted avg"]

    # --- ROC & PR curves ---
    fpr, tpr, _ = roc_curve(all_true_windows, all_pred_windows)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(all_true_windows, all_pred_windows)
    pr_auc = auc(recall, precision)

    print(f"\nROC AUC Score: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    print("\nConfusion Matrix:\n", cm)

    # --- Plot Confusion Matrix ---
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: No Wildlife', 'Pred: Wildlife'],
                yticklabels=['True: No Wildlife', 'True: Wildlife'])
    plt.title("Confusion Matrix (Window-Level)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=300)
    plt.savefig("reports/confusion_matrix.pdf")
    plt.close()

    # --- Plot ROC Curve ---
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=300)
    plt.savefig("reports/roc_curve.pdf")
    plt.close()

    # --- Plot Precision-Recall Curve ---
    plt.figure(figsize=(5,4))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("reports/precision_recall_curve.png", dpi=300)
    plt.savefig("reports/precision_recall_curve.pdf")
    plt.close()

    # --- Print tabular summary ---
    print("\nClassification Metrics Summary:")
    print(f"{'Metric':<25}{'Value':>10}")
    print("-"*35)
    print(f"{'Accuracy':<25}{accuracy:>10.4f}")
    print(f"{'ROC AUC':<25}{roc_auc:>10.4f}")
    print(f"{'PR AUC':<25}{pr_auc:>10.4f}")
    print(f"{'Macro Avg F1':<25}{macro_avg['f1-score']:>10.4f}")
    print(f"{'Weighted Avg F1':<25}{weighted_avg['f1-score']:>10.4f}")
    print("="*70)

    # --- Save numerical report ---
    with open("reports/extended_model_metrics.txt", "w") as f:
        f.write("=== Model Evaluation Summary (Window-Level) ===\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_true_windows, all_pred_windows, digits=4))
    print("📝 Saved all plots and metrics → 'reports/' folder")


# -------------------- CLI --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        train_and_save()
    else:
        predict_all_and_save()
