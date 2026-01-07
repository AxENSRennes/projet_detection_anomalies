# Anomaly Detection on Radio Signals - Research Summary

## Task Overview
- **Training data**: 30,000 signals from classes 0-5 (SNR: 0, 10, 20, 30 dB)
- **Test data**: 2,000 signals from classes 0-8 (SNR: 10, 20, 30 dB)
- **Goal**: Return 0 for known classes (0-5), return 1 for anomalies (6-8)
- **Metrics**: Precision and Recall

---

## Research Summary: Anomaly Detection Approaches

### 1. DCASE Challenge Approaches (Anomalous Sound Detection)

**Autoencoder-Based Method (Recommended as Baseline)**
- Train autoencoder to reconstruct normal signals (log-mel spectrograms or raw IQ)
- Anomaly score = reconstruction error (MSE)
- Higher reconstruction error → likely anomaly
- Architecture: Dense layers with bottleneck (8 units), batch normalization

**Classifier-Based Method (MobileNetV2-style)**
- Train classifier on known classes (0-5)
- Use negative log-probability or entropy of softmax as anomaly score
- Low confidence predictions → likely anomaly

**Evaluation Metrics from DCASE:**
- **AUC** (Area Under ROC Curve) - overall discrimination
- **pAUC** (Partial AUC at FPR ≤ 10%) - performance at low false-positive rates

### 2. Scikit-learn Outlier/Novelty Detection Methods

Apply these in the **latent space** of a trained neural network:

| Method | Best For | Key Idea |
|--------|----------|----------|
| **One-Class SVM** | Novelty detection | Learns boundary around normal data |
| **Isolation Forest** | High-dimensional data | Anomalies are easier to isolate |
| **Local Outlier Factor (LOF)** | Density-based anomalies | Compares local density to neighbors |
| **Elliptic Envelope** | Gaussian latent space | Mahalanobis distance from center |

**Recommended**: Isolation Forest or One-Class SVM on latent features from a classifier/encoder.

### 3. Deep Learning Approaches (Recent Research)

**Variational Autoencoder (VAE)**
- KL divergence regularizes latent space (smooth, centered distribution)
- Better generalization, easier to distinguish normal vs anomalous
- Anomaly score: reconstruction error + KL divergence

**Hybrid: Autoencoder + Isolation Forest**
- Train autoencoder on normal data
- Extract latent representations
- Apply Isolation Forest in latent space
- Achieves near-perfect detection in recent benchmarks

**Structured Latent Space (SVDD)**
- Support Vector Data Description in latent space
- Learn compact hypersphere around normal data
- Points outside hypersphere are anomalies

---

## Proposed Implementation Strategy

### Phase 1: Baseline - Classical Signal Analysis (No Deep Learning)

**Feature Extraction from IQ Signals:**
- **Time-domain features**: Mean, variance, skewness, kurtosis of I and Q channels
- **Frequency-domain features**: FFT magnitude, spectral centroid, bandwidth, peak frequency
- **Statistical features**: Higher-order statistics, cyclostationary features
- **Energy-based features**: Signal power, crest factor

**Novelty Detection Methods:**
1. **Isolation Forest** on extracted features
2. **One-Class SVM** on extracted features
3. **Mahalanobis distance** (Elliptic Envelope) if features are Gaussian

This baseline requires no neural network training and provides a reference for comparison.

### Phase 2: Deep Learning - Autoencoder + Classifier

**Option A: Autoencoder Reconstruction Error**
1. Train autoencoder on training signals (classes 0-5)
2. Architecture: Conv1D encoder → bottleneck → Conv1D decoder
3. Anomaly score = reconstruction MSE
4. Threshold-based classification

**Option B: Classifier with Latent Space Detection**
1. Train CNN classifier on classes 0-5
2. Extract latent features (before final softmax)
3. Apply Isolation Forest or One-Class SVM on latent features
4. Also compare: softmax entropy/max-prob as anomaly score

### Phase 3: Advanced - VAE-based Anomaly Detection

1. Train Variational Autoencoder on normal signals (classes 0-5)
2. Architecture: Conv1D encoder → μ, σ → reparameterization → Conv1D decoder
3. Anomaly score = reconstruction error + β × KL divergence
4. Threshold-based classification with ROC curve analysis

---

## Jupyter Notebook Structure

### 1. Data Exploration & Visualization
- Load data using existing `data_utils.py`
- Visualize sample IQ signals (time domain)
- Plot FFT/spectrograms for each class
- Statistics: class distribution, SNR distribution
- Difficulty assessment

### 2. Phase 1: Classical Signal Analysis Baseline
- Extract handcrafted features (time + frequency domain)
- Train Isolation Forest / One-Class SVM on features
- Evaluate: precision, recall, AUC, pAUC
- Analyze results by SNR and anomaly class

### 3. Phase 2: Autoencoder + Classifier
- Build Conv1D autoencoder
- Train on normal signals (classes 0-5)
- Reconstruction error as anomaly score
- Optional: Classifier with latent space detection
- Compare with Phase 1

### 4. Phase 3: VAE-based Detection
- Build VAE architecture
- Train on normal signals
- Anomaly score = reconstruction + KL
- Final comparison of all methods

### 5. Results & Analysis
- ROC curves, precision-recall curves
- Comparison table of all methods
- Qualitative analysis: success/failure cases
- Analysis by SNR and anomaly subclass
- Critical discussion

---

## Files to Create/Modify
- `anomaly_detection.ipynb` - Main notebook (new)
- `data_utils.py` - May add feature extraction functions

---

## Sources

- [DCASE 2022 - Unsupervised Anomalous Sound Detection](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
- [DCASE 2023 - First-Shot Anomalous Sound Detection](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
- [Scikit-learn Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Enhancing Anomaly Detection Through Latent Space Manipulation](https://www.mdpi.com/2076-3417/15/1/286)
- [Hybrid Autoencoder and Isolation Forest for IoT Anomaly](https://etasr.com/index.php/ETASR/article/download/15288/6110/77171)
- [Deep Learning Advancements in Anomaly Detection Survey](https://arxiv.org/html/2503.13195v1)
