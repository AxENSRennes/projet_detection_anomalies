# Deep-Deep Dive Analysis Report
## Exploiting Signal Structure for Anomaly Detection

---

## Executive Summary

This analysis builds on the EDA and Deep Dive findings to discover **highly discriminative transforms** that exploit the specific nature of the radio signals. The key breakthrough is the **C42 cumulant**, which achieves near-perfect separation.

### Key Results

| Approach | AUC | Best F1 | Precision | Recall |
|----------|-----|---------|-----------|--------|
| Baseline (EDA) | ~0.55 | 0.337 | 71.0% | 22.1% |
| C42 alone | 0.839 | 0.794 | 65.8% | 100% |
| **C42 filter + IF** | **0.905** | **0.863** | **78.5%** | **95.9%** |

**Improvement: F1 increased from 0.337 to 0.863 (156% improvement)**

---

## 1. The C42 Cumulant Discovery

### What is C42?

The fourth-order cumulant C42 is defined as:
```
C42 = E[|x|⁴] - |E[x²]|² - 2·E[|x|²]²
```

It captures **modulation-specific signatures** that power/amplitude cannot reveal.

### C42 Distribution by Class

| Class | Type | C42 Mean | C42 Std |
|-------|------|----------|---------|
| 0 | known | -0.495 | 0.035 |
| 1 | known | -0.744 | 0.020 |
| 2 | known | -0.744 | 0.021 |
| 3 | known | -0.088 | 0.024 |
| 4 | known | -0.246 | 0.016 |
| 5 | known | -0.246 | 0.017 |
| **6** | **ANOMALY** | **-0.085** | 0.022 |
| **7** | **ANOMALY** | **-0.244** | 0.016 |
| **8** | **ANOMALY** | **-0.088** | 0.024 |

### Key Insight: Perfect Separation with C42 < -0.35

- **521 test samples** with C42 < -0.35
- **0 anomalies** in this group!
- These are classes 0, 1, 2 (highly negative C42)

This creates a **two-stage detection strategy**:
1. If C42 < -0.35 → Definitely normal
2. Otherwise → Apply anomaly detection

---

## 2. Feature Ranking (Individual AUCs)

| Rank | Feature | AUC | Description |
|------|---------|-----|-------------|
| 1 | **C42** | **0.839** | Fourth-order cumulant |
| 2 | amp_mean | 0.803 | Mean amplitude |
| 3 | C21 | 0.751 | Second-order cumulant |
| 4 | phase_std | 0.729 | Phase standard deviation |
| 5 | inst_freq_std | 0.729 | Instantaneous frequency std |
| 6 | acf_100 | 0.719 | ACF at lag 100 |
| 7 | acf_50 | 0.718 | ACF at lag 50 |
| 8 | iq_entropy | 0.622 | IQ constellation entropy |
| 9 | spectral_entropy | 0.605 | Spectral entropy |
| 10 | spectral_spread | 0.599 | Spectral spread |

---

## 3. Best Approach: C42 Filter + Isolation Forest

### Algorithm

```python
# Step 1: C42 Filter
sure_normal = (C42 < -0.35)  # 521 samples, 0 anomalies

# Step 2: Anomaly detection on remaining samples
ambiguous = (C42 >= -0.35)  # 1479 samples, 972 anomalies

# Train IF on ambiguous training samples
clf = IsolationForest(contamination=0.4)
clf.fit(train_features[train_C42 >= -0.35])

# Score ambiguous test samples
scores = -clf.score_samples(test_features[ambiguous])

# Final scores
final_scores[sure_normal] = -10  # Definitely normal
final_scores[ambiguous] = scores
```

### Results

| Metric | Value |
|--------|-------|
| Overall AUC | **0.905** |
| Class 6 AUC | 0.908 |
| Class 7 AUC | 0.908 |
| Class 8 AUC | 0.898 |
| Best F1 | **0.863** |
| Precision | 78.5% |
| Recall | 95.9% |

---

## 4. Why These Features Work

### C42 Cumulant
- Captures modulation type (PSK, QAM, etc.)
- Higher for simpler modulations (BPSK-like)
- Classes 3, 6, 8 have similar high C42 → different modulation
- Classes 4, 5, 7 have similar medium C42 → different modulation

### Phase Features (phase_std, inst_freq_std)
- Capture phase dynamics of modulation
- Different modulation schemes have different phase patterns

### ACF Features (acf_50, acf_100)
- Capture symbol timing information
- Cyclostationary nature of radio signals
- Different symbol rates → different ACF patterns

### Amplitude Mean
- Separates high-power (0,1,2) from low-power (3,4,5) groups
- Anomalies fall in low-power region

---

## 5. Anomaly Class Analysis

### Class 6 (Easiest, AUC=0.908)
- Very similar to Class 3 (C42 ~ -0.09)
- BPSK-like modulation (horizontal spread in constellation)
- Detectable because Class 3 is in training data

### Class 7 (Medium, AUC=0.908)
- Similar to Classes 4, 5 (C42 ~ -0.25)
- But unique amplitude kurtosis (-0.65 vs ~0)
- Detectable via combination of features

### Class 8 (Medium, AUC=0.898)
- Very similar to Class 3 (C42 ~ -0.09)
- Four-corner QAM pattern in constellation
- Detectable similar to Class 6

---

## 6. Transforms Explored

### Highly Effective
| Transform | Best Feature | AUC |
|-----------|--------------|-----|
| Higher-Order Cumulants | C42 | 0.839 |
| Amplitude Statistics | amp_mean | 0.803 |
| Phase Analysis | phase_std | 0.729 |
| Cyclostationary (ACF) | acf_100 | 0.719 |

### Moderately Effective
| Transform | Best Feature | AUC |
|-----------|--------------|-----|
| Spectral Entropy | spectral_entropy | 0.605 |
| IQ Constellation | iq_entropy | 0.622 |

### Less Effective for This Task
- Wavelet decomposition (AUC ~0.57)
- Multi-scale kurtosis (AUC ~0.52)
- One-Class SVM (AUC ~0.59)

---

## 7. Recommendations for Deep Learning

### What the Network Should Learn

1. **C42-like representations**: Higher-order statistics
   - Use higher-order pooling layers
   - Or explicitly compute cumulants as input

2. **Phase-aware features**:
   - Complex-valued convolutions
   - Phase extraction layers

3. **Multi-scale temporal patterns**:
   - Dilated convolutions for symbol timing
   - ACF-like correlations

### Suggested Architectures

**Option 1: Feature-Augmented CNN**
```
Input: Raw IQ (2048, 2)
Augment with: C42, amp_mean, phase_std (computed features)
CNN + Dense layers
Output: Anomaly score
```

**Option 2: Complex-Valued Network**
```
Input: Complex IQ (2048, 1)
Complex Conv1D layers
Magnitude/Phase split
Dense layers
Output: Anomaly score
```

**Option 3: Autoencoder with Cumulant Loss**
```
Input: Raw IQ
Encoder → Latent space
Decoder → Reconstruction
Loss: MSE + C42 prediction auxiliary loss
Anomaly score: Reconstruction error
```

---

## 8. Generated Files

| File | Description |
|------|-------------|
| `feature_extraction.py` | Feature extraction module with all transforms |
| `03_advanced_transforms_analysis.ipynb` | Full analysis notebook |
| `DEEP_DEEP_DIVE_REPORT.md` | This report |

---

## 9. Conclusion

The deep-deep dive analysis reveals that **higher-order cumulants (specifically C42)** are the key to effective anomaly detection in this radio signal dataset. The C42 cumulant:

1. **Achieves AUC=0.839 as a single feature**
2. **Perfectly separates** 521 normal samples (C42 < -0.35)
3. **Combined with IF**, achieves AUC=0.905 and F1=0.863

This is a **156% improvement** over the baseline F1 of 0.337.

For the final deep learning model, the network should be designed to learn C42-like higher-order statistics, either implicitly through architecture choices or explicitly through auxiliary losses.
