# Deep Dive Analysis Report
## Advanced Signal Transforms for Anomaly Detection

---

## Executive Summary

This deep-dive analysis explored **advanced signal processing transforms** to find discriminative features that can separate anomaly classes (6,7,8) from known classes (0-5). The key insight is that **anomalies are not simple outliers** - they have statistical properties similar to half the training classes.

### Key Discoveries

| Finding | Impact |
|---------|--------|
| Data splits into TWO groups: High-power (0,1,2) and Low-power (3,4,5) | Anomalies fall in low-power region |
| Class 7 is hardest (AUC=0.94 separable from 3,4,5 using amp_kurtosis) | Requires specialized detection |
| Higher-order cumulants (C42) achieve AUC=0.79 for anomaly detection | Key modulation signature |
| Symbol timing (ACF) separates the two groups clearly | Two-stage approach possible |

---

## 1. Feature Categories Explored

### 1.1 Higher-Order Cumulants
**Purpose**: Capture modulation-specific signatures that power/amplitude cannot reveal.

| Cumulant | Formula | Best AUC | Insight |
|----------|---------|----------|---------|
| C42 | E[|x|⁴] - |E[x²]|² - 2E[|x|²]² | 0.787 | Separates modulation types |
| C40 | E[x⁴] - 3E[x²]² | 0.534 | Useful for Class 7 (AUC=0.67) |

**Key Finding**: C42 is the **single best modulation discriminator** for overall anomaly detection.

### 1.2 Phase Features
**Purpose**: Capture phase dynamics and modulation characteristics.

| Feature | AUC (Overall) | Insight |
|---------|---------------|---------|
| Phase linearity | 0.647 | Class 7 has lowest value (6.88) |
| Phase diff kurtosis | 0.540 | Varies with modulation type |

### 1.3 Entropy Features
**Purpose**: Measure signal complexity and predictability.

| Feature | Best For | Note |
|---------|----------|------|
| Spectral entropy | General | Higher for anomalies |
| Spectral flatness | Class 7 | 0.24 vs 0.21 for known |

### 1.4 Constellation Geometry
**Purpose**: Analyze IQ plane structure via clustering.

| Feature | AUC | Insight |
|---------|-----|---------|
| Cluster inertia (k=4) | 0.764 | Anomalies have different clustering |
| Radius std | 0.765 | Lower for classes 4,5,7 |

### 1.5 Cyclostationary / Symbol Timing
**Purpose**: Reveal symbol rate and timing information.

| Feature | AUC | Insight |
|---------|-----|---------|
| N ACF peaks | 0.752 | Classes 0,1,2: ~30-250, Classes 3-8: ~600 |
| ACF @ lag 50 | 0.724 | Positive for 0,1,2; Negative for 3-8 |
| ACF @ lag 100 | 0.724 | Same pattern |

**Critical Insight**: Symbol timing clearly separates the two regimes!

### 1.6 AM/FM Modulation Features
**Purpose**: Quantify amplitude and frequency modulation characteristics.

| Feature | AUC | Insight |
|---------|-----|---------|
| AM Index | 0.765 | Higher for classes 3,6,8 |
| AM Depth | 0.501 | Classes 4,5,7 have lower depth (~0.6) |

### 1.7 Bispectrum (Higher-Order Spectrum)
**Purpose**: Capture non-Gaussian and nonlinear signal properties.

| Class Type | Bispectrum Max | Note |
|------------|----------------|------|
| High-power (0,1,2) | 16,000-21,000 | Higher energy |
| Low-power (3,4,5) | 7,700-10,500 | Lower energy |
| Anomalies (6,7,8) | 7,400-10,700 | Similar to low-power |

---

## 2. Best Discriminative Features

### 2.1 Overall Anomaly Detection (Known vs Anomaly)

| Rank | Feature | AUC | Category |
|------|---------|-----|----------|
| 1 | amp_mean | 0.803 | Basic |
| 2 | C42 | 0.787 | Cumulant |
| 3 | radius_std | 0.765 | Constellation |
| 4 | am_index | 0.765 | Modulation |
| 5 | cluster_inertia | 0.764 | Constellation |
| 6 | n_acf_peaks | 0.752 | Symbol timing |
| 7 | acf_50 | 0.724 | Symbol timing |
| 8 | acf_100 | 0.724 | Symbol timing |

### 2.2 Detecting Class 7 (Hardest Anomaly)

| Rank | Feature | AUC (7 vs 3,4,5) | Note |
|------|---------|------------------|------|
| 1 | amp_kurtosis | **0.936** | Class 7 has unique low value (-0.65) |
| 2 | radius_kurtosis | **0.936** | Same as amp_kurtosis |
| 3 | acf_50 | 0.719 | Timing difference |
| 4 | acf_100 | 0.716 | Timing difference |
| 5 | C40_abs | 0.674 | Lower for Class 7 |

**Key Discovery**: Amplitude kurtosis achieves **93.6% AUC** for separating Class 7 from similar known classes!

---

## 3. Data Structure Insights

### 3.1 Two Distinct Regimes

The training data naturally splits into two groups based on multiple features:

| Property | High-Power Group (0,1,2) | Low-Power Group (3,4,5) |
|----------|--------------------------|-------------------------|
| Mean amplitude | ~0.93-0.95 | ~0.74-0.78 |
| ACF peaks | 30-250 | 585-635 |
| ACF @ lag 100 | +0.04 to +0.07 | -0.17 to -0.32 |
| Bispectrum max | 16,000-21,000 | 7,700-10,500 |

**Anomalies (6,7,8) fall entirely in the Low-Power regime.**

### 3.2 Per-Class Characteristics

| Class | Mean Amp | Amp Kurtosis | C42 | ACF Peaks | Detection Difficulty |
|-------|----------|--------------|-----|-----------|---------------------|
| 0 | 0.93 | -0.80 | -0.44 | 33 | N/A (known) |
| 1 | 0.95 | +0.78 | -0.67 | 247 | N/A (known) |
| 2 | 0.95 | +0.76 | -0.67 | 250 | N/A (known) |
| 3 | 0.75 | +0.14 | -0.16 | 587 | N/A (known) |
| 4 | 0.78 | -0.08 | -0.47 | 631 | N/A (known) |
| 5 | 0.78 | -0.08 | -0.46 | 627 | N/A (known) |
| **6** | 0.74 | +0.08 | -0.16 | 585 | Medium |
| **7** | 0.77 | **-0.65** | -0.45 | 636 | **Hardest** |
| **8** | 0.74 | +0.15 | -0.16 | 595 | Medium |

### 3.3 Why Class 7 is Hardest

Class 7 is difficult because:
1. **Same mean amplitude** as classes 4,5 (~0.77)
2. **Similar C42** to classes 4,5 (~-0.45)
3. **Similar symbol timing** (ACF peaks, lag correlation)

**However**, Class 7 has a **unique amplitude kurtosis of -0.65** (vs ~0 for classes 3,4,5), which provides excellent separation.

---

## 4. Modulation Hypothesis

Based on constellation patterns and statistical features:

| Class | Likely Modulation | Evidence |
|-------|-------------------|----------|
| 0 | QAM-like | 4-quadrant clusters, high power |
| 1 | PSK-like | Central concentration |
| 2 | PSK-like | Central concentration |
| 3 | QPSK-like | 4 distinct clusters, low C42 |
| 4 | Similar to 5 | Spread pattern |
| 5 | Similar to 4 | Spread pattern |
| 6 (anomaly) | BPSK-like | Horizontal spread |
| 7 (anomaly) | Similar to 4,5 | But lower kurtosis |
| 8 (anomaly) | Four-corner QAM | 4 corner clusters |

---

## 5. Recommendations for Modeling

### 5.1 Two-Stage Detection Approach

```
Stage 1: Is sample from High-Power (0,1,2) or Low-Power (3,4,5) region?
         Features: amp_mean, n_acf_peaks, acf_100
         If High-Power → Definitely NOT anomaly (return 0)

Stage 2: For Low-Power region, apply anomaly detection
         Features: amp_kurtosis, C42, C40_abs, phase_linearity
         Focus on Class 7 separation using kurtosis
```

### 5.2 Feature Engineering for Deep Learning

The network should learn to capture:

1. **Higher-order statistics** (cumulants)
   - Use higher-order pooling layers
   - Or explicit cumulant computation layers

2. **Multi-scale temporal patterns**
   - Multi-scale convolutions
   - Dilated convolutions for symbol timing

3. **Constellation geometry**
   - Attention on IQ relationships
   - Cluster-aware representations

### 5.3 Specific Architecture Suggestions

1. **Autoencoder approach**:
   - Train on classes 0-5
   - Expect higher reconstruction error for anomalies
   - Add kurtosis/cumulant auxiliary loss

2. **Classifier + Latent Space**:
   - Train 6-class classifier
   - Extract embeddings
   - Apply anomaly detection in latent space with focus on features listed above

3. **Contrastive Learning**:
   - Learn representations where same-class samples are close
   - Anomalies should be far from all centroids
   - Use amplitude kurtosis as auxiliary signal

---

## 6. Generated Visualizations

| File | Content |
|------|---------|
| `plots_comprehensive_summary.png` | Overall feature ranking and scatter plots |
| `plots_class7_comparison.png` | Class 7 vs 3,4,5 feature comparison |
| `plots_bispectrum.png` | Higher-order spectrum for all classes |
| `plots_instantaneous_freq.png` | Instantaneous frequency distributions |
| `plots_symbol_timing_acf.png` | Autocorrelation revealing symbol timing |
| `plots_am_fm_features.png` | AM/FM modulation features |
| `plots_iq_density.png` | IQ constellation density maps |

---

## 7. Conclusion

This deep-dive analysis reveals that **simple features are insufficient** for anomaly detection because anomalies statistically resemble half the known classes. However, we identified **powerful discriminative features**:

1. **amp_kurtosis** (AUC=0.94 for Class 7) - Best single feature for hardest case
2. **C42 cumulant** (AUC=0.79) - Captures modulation signatures
3. **Symbol timing features** (AUC=0.72-0.75) - Separates two regimes
4. **Constellation geometry** (AUC=0.76) - Cluster structure differences

A deep learning model that learns to compute these features implicitly should significantly outperform the baseline 34% F1 score.

**Predicted achievable performance with right architecture**:
- Precision: 75-85%
- Recall: 60-75%
- F1: 65-80%
