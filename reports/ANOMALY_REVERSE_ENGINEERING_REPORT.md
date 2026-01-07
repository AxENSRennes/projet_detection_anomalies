# Signal Analysis & Anomaly Reverse Engineering Report

## Executive Summary
Following the deep-dive analysis of the radio signal dataset, three distinct anomaly classes (6, 7, and 8) have been identified based on their IQ constellations and statistical signatures. This report outlines a "reverse engineering" of these signals and proposes four advanced transformations designed to maximize the discrimination between these anomalies and the known training classes (0-5).

---

## 1. Identified Anomaly Characteristics

Based on IQ plane observations and higher-order statistics, we have identified the likely modulation schemes for the anomalies:

| Anomaly | Likely Modulation | Key Evidence |
| :--- | :--- | :--- |
| **Class 6** | **BPSK** (Binary Phase Shift Keying) | Horizontal spread in IQ plane; 180° phase transitions. |
| **Class 7** | **Constant Envelope** (e.g., GMSK, 8-PSK) | Annular (ring) pattern; uniquely low amplitude kurtosis (-0.65). |
| **Class 8** | **QPSK / 4-QAM** | Four distinct clusters in the corners of the constellation. |

**Common Parameter:** All anomalies share a symbol rate of approximately **3.4 samples per symbol**, derived from Autocorrelation Function (ACF) peak analysis.

---

## 2. Proposed Discriminative Transformations

To enhance anomaly detection performance, we propose four physical signal transformations that exploit the identified symmetries and timing.

### 2.1 Nonlinear Power Transforms ($x^2, x^4$)
Nonlinear mappings are highly effective at "unmasking" the phase structure of digital modulations by collapsing states into a single phase point.
*   **Square Transform ($y = x^2$):** For BPSK (Class 6), this maps the $\{0, \pi\}$ states to a single point at $0$ rad. In the frequency domain, this creates a high-intensity spectral line at twice the carrier frequency.
*   **Quad Transform ($y = x^4$):** For QPSK (Class 8), this collapses the four quadrants into a single point, revealing a spectral line at four times the carrier frequency.
*   **Discriminative Power:** Known classes (0-5) lack this specific symmetry or use amplitude shaping (QAM) that prevents the emergence of such sharp spectral lines.

### 2.2 Symbol-Lagged Correlation (Delay $\tau \approx 3.4$)
By utilizing the retro-engineered symbol rate, we can apply a lagged conjugate product:
$$y(t) = x(t) \cdot x^*(t - \tau) \quad \text{where } \tau \approx 3.4$$
*   **Mechanism:** This transformation highlights the phase relationship between consecutive symbols.
*   **Discriminative Power:** For Constant Envelope signals (Class 7), the angle of $y(t)$ will remain remarkably stable ("phase plateaus") during the symbol duration. This contrasts sharply with classes 4 and 5, which exhibit continuous phase fluctuations due to linear pulse shaping.

### 2.3 Instantaneous Frequency Analysis ($d\phi/dt$)
Moving from the phase domain to the frequency derivative reveals the underlying frequency shift logic.
*   **Mechanism:** Calculated as $f_{inst} = \frac{1}{2\pi} \frac{d\phi}{dt}$.
*   **Discriminative Power:** If Class 7 is a Frequency Shift Keying (FSK) variant, it will display discrete, stable frequency levels ("staircase" patterns). In contrast, Phase Shift Keying (PSK) classes (0-5) will show erratic spikes during transitions, providing a clear boundary for anomaly classification.

### 2.4 Cyclostationary Feature Extraction (SCF)
Radio signals are not stationary; they are cyclostationary (their statistics vary periodically with the symbol rate).
*   **Mechanism:** Compute the **Spectral Correlation Function (SCF)** or look for peaks in the spectrum of the instantaneous power $|x(t)|^2$.
*   **Discriminative Power:** Anomalies exhibit cyclic energy peaks at $\alpha = 1/3.4$. Since the known classes may have different symbol rates or pulse shapes, their "cyclic signatures" will be shifted or possess different magnitudes, allowing for near-perfect separation in the cyclic frequency domain.

---

## 3. Implementation Recommendation
For a deep learning architecture, we recommend incorporating these transformations as **"Physical Priors"** (e.g., pass $x^2$ and $|x|^2$ as additional input channels or use them in an auxiliary loss function). This allows the network to bypass "learning" the physics from scratch and focus on the residual differences in the transformed space.
