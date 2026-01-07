# Exploratory Data Analysis Report
## Radio Signal Anomaly Detection

This document summarizes the exploratory data analysis performed on the radio signal dataset for the anomaly detection task.

---

## 1. Objective

Detect during test phase whether certain radio signals were **not seen during training**:
- Training: Classes 0-5 (known)
- Test: Classes 0-5 (known) + Classes 6-8 (anomalies)
- Goal: Return 0 if signal is from classes 0-5, return 1 otherwise

---

## 2. Analysis Process

### Step 1: Dataset Structure Inspection

Loaded both HDF5 files and examined their structure:

| Dataset | Samples | Classes | SNR (dB) | Signal Shape |
|---------|---------|---------|----------|--------------|
| `train.hdf5` | 30,000 | 0-5 | 0, 10, 20, 30 | (2048, 2) |
| `test_anomalies.hdf5` | 2,000 | 0-8 | 10, 20, 30 | (2048, 2) |

**Key observation**: Training has SNR=0 dB but test does NOT.

### Step 2: Class and SNR Distribution Analysis

**Training distribution:**
- ~5,000 samples per class (balanced)
- ~7,500 samples per SNR level

**Test distribution:**
- Known classes (0-5): 1,028 samples (51.4%)
- Anomaly classes (6-8): 972 samples (48.6%)
- SNR distribution: 29% at 10dB, 33% at 20dB, 38% at 30dB

### Step 3: Time Domain Visualization

Plotted raw IQ (In-phase/Quadrature) signals for all classes:
- All signals show rapid oscillations typical of modulated radio
- Difficult to distinguish classes visually in time domain
- Generated: `plots_train_classes.png`, `plots_anomaly_classes.png`

### Step 4: IQ Constellation Analysis

Plotted I vs Q scatter plots revealing modulation patterns:
- **Class 0**: QAM-like with 4 quadrant clusters
- **Class 1, 2**: PSK-like with central concentration
- **Class 3**: Four distinct QPSK-like clusters
- **Class 4, 5**: Spread patterns with some structure
- **Anomaly 6**: Horizontal spread (BPSK-like)
- **Anomaly 7**: Ring/annular pattern
- **Anomaly 8**: Four corner clusters

Generated: `plots_iq_constellation.png`

### Step 5: Frequency Domain Analysis

Computed power spectra (FFT) and spectrograms:
- Classes show different spectral bandwidths
- Class 3 has notably narrower bandwidth
- Anomaly classes have distinct spectral signatures
- Signals are largely stationary over time

Generated: `plots_power_spectrum.png`, `plots_spectrograms.png`

### Step 6: Statistical Feature Extraction

Computed 20 features for each signal:
- Amplitude statistics: mean, std, max, min
- Power statistics: mean, std
- Instantaneous frequency: mean, std
- I/Q channel statistics: mean, std, correlation
- Higher-order moments: kurtosis, skewness
- Spectral features: centroid, spread, flatness

Generated: `train_features.csv`, `test_features.csv`

### Step 7: Feature Distribution Comparison

Compared feature distributions between known and anomaly classes:
- Anomalies occupy only the **low-amplitude/low-power** region
- Known classes have bimodal distribution (two groups)
- Anomalies resemble classes 3, 4, 5 statistically

Generated: `plots_feature_distributions.png`

### Step 8: Dimensionality Reduction

**PCA Analysis:**
- PC1 (53.9% variance): Separates high-power (0,1,2) from low-power (3,4,5)
- All anomalies fall in the low-power cluster (PC1 < 0)
- Cannot distinguish anomalies from classes 3,4,5 using PCA

**t-SNE Analysis:**
- Classes 0, 1, 2 form separate cluster (no anomalies)
- Anomalies 6 & 8 form their own cluster (detectable)
- Anomaly 7 mixed with classes 3, 4, 5 (hardest to detect)

Generated: `plots_pca.png`, `plots_tsne.png`

### Step 9: SNR Impact Analysis

Analyzed how features change with SNR:
- SNR=0 dB has dramatically different statistics:
  - Mean amplitude: 1.62 (vs ~0.86 at higher SNR)
  - Mean power: 3.33 (vs ~0.84 at higher SNR)
- Features stabilize for SNR >= 10 dB
- Class separation maintained across SNR levels

Generated: `plots_snr_impact.png`

### Step 10: Baseline Anomaly Detection

Tested classical anomaly detection on hand-crafted features:

| Method | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Isolation Forest | 71.0% | 22.1% | 33.7% |
| One-Class SVM | 51.2% | 13.1% | 20.8% |
| Local Outlier Factor | 44.0% | 9.4% | 15.4% |

**Per-anomaly detection rates (Isolation Forest):**
- Anomaly 6: 19.6%
- Anomaly 7: 17.9%
- Anomaly 8: 28.7%

---

## 3. Key Findings

### Finding 1: Two Distinct Groups in Training Data

| Group | Classes | Mean Amplitude | Mean Power |
|-------|---------|----------------|------------|
| High Power | 0, 1, 2 | ~0.95 | ~1.0 |
| Low Power | 3, 4, 5 | ~0.77 | ~0.67 |

### Finding 2: Anomalies Resemble Low-Power Group

All three anomaly classes (6, 7, 8) have statistics similar to classes 3, 4, 5:
- Mean amplitude: 0.74-0.78
- Mean power: 0.67
- This makes detection challenging using simple features

### Finding 3: SNR=0 dB Distribution Shift

| Metric | SNR=0 dB | SNR>=10 dB |
|--------|----------|------------|
| Mean Amplitude | 1.62 | 0.85-0.95 |
| Mean Power | 3.33 | 0.84-1.08 |
| Std Amplitude | 0.83 | 0.31-0.42 |

**Implication**: Since test has no SNR=0, training on it may hurt performance.

### Finding 4: Detection Difficulty Varies by Anomaly Class

| Anomaly | Difficulty | Reason |
|---------|------------|--------|
| Class 8 | Easiest | Forms distinct cluster in t-SNE |
| Class 6 | Medium | Clusters with class 8 |
| Class 7 | Hardest | Mixed with classes 3, 4, 5 |

### Finding 5: Hand-crafted Features Are Insufficient

Baseline methods achieve only 22% recall with 71% precision. Deep learning representations are needed to capture subtle modulation differences.

---

## 4. Generated Files

### Data Files
| File | Description | Size |
|------|-------------|------|
| `train_features.csv` | 30,000 samples x 20 features | 7.6 MB |
| `test_features.csv` | 2,000 samples x 20 features | 509 KB |

### Visualization Files
| File | Content |
|------|---------|
| `plots_train_classes.png` | Time-domain IQ signals for classes 0-5 |
| `plots_anomaly_classes.png` | Time-domain IQ signals for classes 6-8 |
| `plots_iq_constellation.png` | IQ scatter plots for all classes |
| `plots_power_spectrum.png` | FFT power spectra for all classes |
| `plots_spectrograms.png` | Time-frequency spectrograms |
| `plots_feature_distributions.png` | Feature histograms: known vs anomaly |
| `plots_pca.png` | PCA projection (2D) |
| `plots_tsne.png` | t-SNE projection (2D) |
| `plots_snr_impact.png` | Feature variation with SNR |

### Notebook
| File | Description |
|------|-------------|
| `01_exploratory_analysis.ipynb` | Complete interactive EDA notebook |

---

## 5. Recommendations for Modeling

### Data Preprocessing
1. **Filter SNR=0 dB** from training data (or use SNR-conditioned normalization)
2. **Normalize** signals per-sample or use batch normalization
3. Consider **frequency-domain representation** (FFT/spectrogram) as input

### Modeling Approaches

**Approach 1: Autoencoder**
- Train autoencoder on classes 0-5
- Use reconstruction error as anomaly score
- Anomalies should have higher reconstruction error

**Approach 2: Deep Classifier + Latent Space**
- Train classifier on classes 0-5
- Extract embeddings from penultimate layer
- Apply novelty detection (Isolation Forest, One-Class SVM) in latent space

**Approach 3: Contrastive Learning**
- Learn representations where same-class samples are close
- Anomalies should be far from all known class centroids

### Evaluation Strategy
- Primary metrics: Precision and Recall
- Analyze performance by:
  - Anomaly class (6, 7, 8 separately)
  - SNR level (10, 20, 30 dB)
- Use threshold tuning on validation set

---

## 6. Conclusion

The exploratory analysis reveals that this anomaly detection task is **non-trivial**:

1. Anomalies are not outliers in simple feature space
2. They resemble half of the known classes (3, 4, 5)
3. Class 7 is particularly challenging (mixed with known classes in t-SNE)
4. Deep learning is necessary to capture subtle modulation differences

The baseline F1 score of 34% with hand-crafted features provides a reference point. A successful deep learning approach should significantly exceed this.
