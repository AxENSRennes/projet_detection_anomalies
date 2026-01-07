"""
Data utilities for the Anomaly Detection project.
Loads radio signal data from HDF5 files and provides PyTorch Dataset/DataLoader support.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).parent
KNOWN_CLASSES = set(range(6))  # Classes 0-5 are known
ANOMALY_CLASSES = {6, 7, 8}  # Classes 6-8 are anomalies


def load_hdf5_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from an HDF5 file.

    Args:
        filepath: Path to the HDF5 file

    Returns:
        Tuple of (signals, labels, snr)
        - signals: (N, 2048, 2) float32 - IQ radio signals
        - labels: (N,) int8 - class labels
        - snr: (N,) int16 - Signal-to-Noise Ratio in dB
    """
    with h5py.File(filepath, "r") as f:
        signals = f["signaux"][:]
        labels = f["labels"][:]
        snr = f["snr"][:]
    return signals, labels, snr


def load_train_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data from train.hdf5 (classes 0-5)."""
    return load_hdf5_data(DATA_DIR / "train.hdf5")


def load_test_anomalies() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test anomalies data (classes 0-8, with 6-8 being anomalies)."""
    return load_hdf5_data(DATA_DIR / "test_anomalies.hdf5")


def create_binary_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert class labels to binary labels for anomaly detection.

    Args:
        labels: Original class labels (0-8)

    Returns:
        Binary labels: 0 for known classes (0-5), 1 for anomalies (6-8)
    """
    return (labels >= 6).astype(np.int64)


def filter_by_snr(
    signals: np.ndarray, labels: np.ndarray, snr: np.ndarray, target_snr: int | list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter data by SNR value(s).

    Args:
        signals: Signal array
        labels: Label array
        snr: SNR array
        target_snr: Single SNR value or list of SNR values to keep

    Returns:
        Filtered (signals, labels, snr)
    """
    if isinstance(target_snr, int):
        target_snr = [target_snr]
    mask = np.isin(snr, target_snr)
    return signals[mask], labels[mask], snr[mask]


class RadioSignalDataset(Dataset):
    """PyTorch Dataset for radio signals."""

    def __init__(
        self, signals: np.ndarray, labels: np.ndarray, snr: np.ndarray, binary_labels: bool = False
    ):
        """
        Args:
            signals: (N, 2048, 2) signal array
            labels: (N,) label array
            snr: (N,) SNR array
            binary_labels: If True, convert labels to binary (0=known, 1=anomaly)
        """
        # Convert to (N, 2, 2048) for Conv1d compatibility (channels first)
        self.signals = torch.from_numpy(signals).permute(0, 2, 1).float()

        if binary_labels:
            self.labels = torch.from_numpy(create_binary_labels(labels)).long()
        else:
            self.labels = torch.from_numpy(labels.astype(np.int64)).long()

        self.snr = torch.from_numpy(snr.astype(np.int64)).long()

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx], self.snr[idx]


def get_train_loader(batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
    """
    Create a DataLoader for training data.

    Args:
        batch_size: Batch size
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader for training data
    """
    signals, labels, snr = load_train_data()
    dataset = RadioSignalDataset(signals, labels, snr, binary_labels=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_test_loader(batch_size: int = 32, binary_labels: bool = True, **kwargs) -> DataLoader:
    """
    Create a DataLoader for test anomalies data.

    Args:
        batch_size: Batch size
        binary_labels: If True, use binary labels (0=known, 1=anomaly)
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader for test data
    """
    signals, labels, snr = load_test_anomalies()
    dataset = RadioSignalDataset(signals, labels, snr, binary_labels=binary_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("Loading test anomalies data...")
    signals, labels, snr = load_test_anomalies()
    print(f"  Signals shape: {signals.shape}")
    print(f"  Labels: unique={np.unique(labels)}, counts={np.bincount(labels)}")
    print(f"  SNR: unique={np.unique(snr)}")

    binary = create_binary_labels(labels)
    print(f"  Binary labels: {np.bincount(binary)} (known / anomaly)")

    print("\nCreating test DataLoader...")
    loader = get_test_loader(batch_size=16)
    batch = next(iter(loader))
    print(f"  Batch signals: {batch[0].shape}")
    print(f"  Batch labels: {batch[1].shape}")
    print(f"  Batch SNR: {batch[2].shape}")
