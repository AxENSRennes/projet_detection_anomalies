"""Tests for data_utils module."""

import numpy as np
import pytest
import torch

from src.data_utils import (
    RadioSignalDataset,
    create_binary_labels,
    filter_by_snr,
    load_test_anomalies,
    load_train_data,
)


class TestLoadData:
    """Tests for data loading functions."""

    def test_load_train_data_shape(self):
        """Train data should have expected shape."""
        signals, labels, snr = load_train_data()
        assert signals.shape == (30000, 2048, 2)
        assert labels.shape == (30000,)
        assert snr.shape == (30000,)

    def test_load_train_data_classes(self):
        """Train data should only contain classes 0-5."""
        _, labels, _ = load_train_data()
        unique_labels = np.unique(labels)
        assert set(unique_labels) == {0, 1, 2, 3, 4, 5}

    def test_load_train_data_snr_values(self):
        """Train data should have SNR in {0, 10, 20, 30}."""
        _, _, snr = load_train_data()
        unique_snr = np.unique(snr)
        assert set(unique_snr) == {0, 10, 20, 30}

    def test_load_test_anomalies_shape(self):
        """Test anomalies data should have expected shape."""
        signals, labels, snr = load_test_anomalies()
        assert signals.shape == (2000, 2048, 2)
        assert labels.shape == (2000,)
        assert snr.shape == (2000,)

    def test_load_test_anomalies_classes(self):
        """Test anomalies data should contain classes 0-8."""
        _, labels, _ = load_test_anomalies()
        unique_labels = np.unique(labels)
        assert set(unique_labels) == {0, 1, 2, 3, 4, 5, 6, 7, 8}

    def test_load_test_anomalies_has_anomalies(self):
        """Test data should contain anomaly classes (6, 7, 8)."""
        _, labels, _ = load_test_anomalies()
        anomaly_count = np.sum(labels >= 6)
        assert anomaly_count > 0


class TestCreateBinaryLabels:
    """Tests for binary label creation."""

    def test_known_classes_are_zero(self):
        """Classes 0-5 should map to 0."""
        labels = np.array([0, 1, 2, 3, 4, 5])
        binary = create_binary_labels(labels)
        assert np.all(binary == 0)

    def test_anomaly_classes_are_one(self):
        """Classes 6-8 should map to 1."""
        labels = np.array([6, 7, 8])
        binary = create_binary_labels(labels)
        assert np.all(binary == 1)

    def test_mixed_labels(self):
        """Mixed labels should be correctly converted."""
        labels = np.array([0, 6, 3, 7, 5, 8])
        binary = create_binary_labels(labels)
        expected = np.array([0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(binary, expected)

    def test_output_dtype(self):
        """Output should be int64."""
        labels = np.array([0, 6])
        binary = create_binary_labels(labels)
        assert binary.dtype == np.int64


class TestFilterBySNR:
    """Tests for SNR filtering."""

    def test_filter_single_snr(self):
        """Filter by single SNR value."""
        signals = np.random.randn(100, 2048, 2).astype(np.float32)
        labels = np.random.randint(0, 6, 100).astype(np.int8)
        snr = np.array([10, 20, 30] * 33 + [10])

        filtered_signals, _, filtered_snr = filter_by_snr(signals, labels, snr, target_snr=20)

        assert np.all(filtered_snr == 20)
        assert len(filtered_signals) == 33

    def test_filter_multiple_snr(self):
        """Filter by multiple SNR values."""
        signals = np.random.randn(100, 2048, 2).astype(np.float32)
        labels = np.random.randint(0, 6, 100).astype(np.int8)
        snr = np.array([10, 20, 30] * 33 + [10])

        _, _, filtered_snr = filter_by_snr(signals, labels, snr, target_snr=[10, 30])

        assert set(np.unique(filtered_snr)) == {10, 30}
        assert 20 not in filtered_snr


class TestRadioSignalDataset:
    """Tests for PyTorch Dataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        signals = np.random.randn(50, 2048, 2).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8] * 5 + [0, 1, 2, 3, 4]).astype(np.int8)
        snr = np.array([10, 20, 30] * 16 + [10, 20]).astype(np.int16)
        return signals, labels, snr

    def test_dataset_length(self, sample_data):
        """Dataset length should match input."""
        signals, labels, snr = sample_data
        dataset = RadioSignalDataset(signals, labels, snr)
        assert len(dataset) == 50

    def test_dataset_output_shape(self, sample_data):
        """Dataset should return correctly shaped tensors."""
        signals, labels, snr = sample_data
        dataset = RadioSignalDataset(signals, labels, snr)
        sig, lab, s = dataset[0]

        assert sig.shape == (2, 2048)  # Channels first
        assert lab.shape == ()
        assert s.shape == ()

    def test_dataset_output_types(self, sample_data):
        """Dataset should return correct tensor types."""
        signals, labels, snr = sample_data
        dataset = RadioSignalDataset(signals, labels, snr)
        sig, lab, s = dataset[0]

        assert sig.dtype == torch.float32
        assert lab.dtype == torch.int64
        assert s.dtype == torch.int64

    def test_dataset_binary_labels(self, sample_data):
        """Binary labels should be 0 or 1."""
        signals, labels, snr = sample_data
        dataset = RadioSignalDataset(signals, labels, snr, binary_labels=True)

        all_labels = [dataset[i][1].item() for i in range(len(dataset))]
        assert set(all_labels) == {0, 1}

    def test_dataset_channels_first(self, sample_data):
        """Signals should be permuted to channels-first format."""
        signals, labels, snr = sample_data
        dataset = RadioSignalDataset(signals, labels, snr)

        # Original: (N, 2048, 2) -> Dataset: (2, 2048)
        sig, _, _ = dataset[0]
        assert sig.shape == (2, 2048)
