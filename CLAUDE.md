# Anomaly Detection on Radio Signals

MVA 2025 course project: Deep Learning and Signal Processing.

## Project Guidelines

The mini-project is to be done in pairs.

The mini-project results in a structured Jupyter notebook that you will send and present during a 15-minute defense (plan for 12 minutes of speaking) via videoconference. You can present directly from your notebook and/or make slides if you wish, but it's not required.

The objective is to work approximately 10 hours on the project (per person, so 20 hours per pair).

For each subject you are invited to take initiatives, notably to:
- Analyze the data (high-level statistics, visualization, difficulty assessment)
- Start from the course or a related article that you identified in a (quick) literature review
- Define one or more evaluation metrics
- Define and implement a baseline method whose performance will serve as reference
- Implement at least one and ideally two methods addressing the problem. At least one must be a deep learning approach. The second can be a classical signal processing approach, another network architecture, the same architecture with data augmentation or data engineering strategy...
- Compare approaches quantitatively and analyze your results qualitatively, success cases and failure cases

If you haven't achieved convincing results within this time, you are invited to critically analyze your results and hypothesize what didn't work (approach type, architecture, data quality...).

A good analysis of your results and rigorous methodology will be highly valued in the evaluation, as will your initiatives. Feel free to be creative with original ideas!

## Anomaly Detection Task

The objective is to detect during the test phase that certain signals were not seen during training.

In the training phase, you start from the train.hdf5 dataset (from TP3) which contains signals:
- Labeled 0, 1, 2, 3, 4, 5
- With SNR 30, 20, 10, or 0 dB

For testing, you have test_anomalies.hdf5 with:
- Signals from classes 0-5 (seen during training)
- Signals from classes 6, 7, 8 (NOT represented in training)

The goal is to build an algorithm trained on classes 0-5 and adapt it to detect if a signal was seen or not during training.

**Test objective:**
- Return 0 if signal comes from classes 0-5
- Return 1 otherwise

Performance evaluated in terms of **precision** and **recall**.

All novelty classes can be grouped into the same macro class. You can analyze results conditionally on novelty subclass but you're not required to exploit this information. The decision can depend on a threshold you can vary.

**Suggested approaches:**
- DCASE challenge for anomalous sound detection
- Scikit-learn outlier/novelty detection in a latent space from a deep network
- Start from TP3 networks or retrain with a strategy appropriate for anomaly detection

## Environment

```bash
# Use this Python environment
/home/axel/wsl_venv/bin/python

# Or activate it
source /home/axel/wsl_venv/bin/activate
```

## Data

| File | Samples | Classes | SNR (dB) |
|------|---------|---------|----------|
| `train.hdf5` | 30,000 | 0-5 | 0, 10, 20, 30 |
| `test_anomalies.hdf5` | 2,000 | 0-8 | 10, 20, 30 |

HDF5 structure:
- `signaux`: (N, 2048, 2) float32 - IQ radio signals
- `labels`: (N,) int8 - class labels
- `snr`: (N,) int16 - Signal-to-Noise Ratio

## Code

- `data_utils.py` - Data loading utilities with PyTorch Dataset/DataLoader support

## References

- DCASE challenge for anomalous sound detection
- Scikit-learn outlier detection methods
- Can build on classifier from TP3 and use latent space for anomaly detection
