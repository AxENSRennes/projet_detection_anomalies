"""Tests for eval_utils module."""

import numpy as np
import pytest
import torch

from src.eval_utils import (
    DEFAULT_PAUC_MAX_FPR,
    apply_threshold,
    compare_models,
    compute_binary_metrics,
    compute_pauc,
    compute_precision_recall_auc,
    compute_roc_auc,
    evaluate_by_anomaly_class,
    evaluate_by_snr,
    find_optimal_threshold,
    generate_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_score_distribution,
    to_numpy,
)


class TestToNumpy:
    """Tests for numpy conversion helper."""

    def test_numpy_passthrough(self):
        """Numpy arrays should pass through unchanged."""
        arr = np.array([1, 2, 3])
        result = to_numpy(arr)
        np.testing.assert_array_equal(result, arr)
        assert result is arr  # Same object

    def test_torch_tensor_conversion(self):
        """Torch tensors should be converted to numpy."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_torch_tensor_detach(self):
        """Tensors with gradients should be detached."""
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_cuda_tensor(self):
        """CUDA tensors should be moved to CPU."""
        tensor = torch.tensor([1.0, 2.0]).cuda()
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])


class TestBinaryMetrics:
    """Tests for compute_binary_metrics."""

    def test_perfect_predictions(self):
        """Perfect predictions should give all 1.0 scores."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        metrics = compute_binary_metrics(y_true, y_pred)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["accuracy"] == 1.0

    def test_all_wrong_predictions(self):
        """All wrong predictions should give 0.0 scores."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        metrics = compute_binary_metrics(y_true, y_pred)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["accuracy"] == 0.0

    def test_mixed_predictions(self):
        """Mixed predictions should give intermediate scores."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])  # 1 FP, 2 TP, 1 TN
        metrics = compute_binary_metrics(y_true, y_pred)

        assert metrics["precision"] == pytest.approx(2 / 3)  # TP / (TP + FP)
        assert metrics["recall"] == 1.0  # TP / (TP + FN)
        assert metrics["accuracy"] == 0.75

    def test_accepts_torch_tensor(self):
        """Should accept torch tensors."""
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 0, 1, 1])
        metrics = compute_binary_metrics(y_true, y_pred)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_no_positives(self):
        """Should handle case with no positives (zero_division=0)."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = compute_binary_metrics(y_true, y_pred)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestROCMetrics:
    """Tests for ROC-based metrics."""

    @pytest.fixture
    def perfect_separation_data(self):
        """Data with perfect separation between classes."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        return y_true, scores

    @pytest.fixture
    def random_data(self):
        """Random data for testing edge cases."""
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        scores = np.random.rand(100)
        return y_true, scores

    def test_perfect_roc_auc(self, perfect_separation_data):
        """Perfect separation should give AUC = 1.0."""
        y_true, scores = perfect_separation_data
        auc = compute_roc_auc(y_true, scores)
        assert auc == 1.0

    def test_random_roc_auc(self, random_data):
        """Random scores should give AUC around 0.5."""
        y_true, scores = random_data
        auc = compute_roc_auc(y_true, scores)
        assert 0.3 < auc < 0.7  # Should be roughly 0.5

    def test_pauc_bounds(self, perfect_separation_data):
        """pAUC should be between 0 and 1."""
        y_true, scores = perfect_separation_data
        pauc = compute_pauc(y_true, scores)
        assert 0.0 <= pauc <= 1.0

    def test_pauc_perfect_separation(self, perfect_separation_data):
        """Perfect separation should give pAUC = 1.0."""
        y_true, scores = perfect_separation_data
        pauc = compute_pauc(y_true, scores)
        assert pauc == 1.0

    def test_pauc_configurable_fpr(self, perfect_separation_data):
        """pAUC should accept different max_fpr values."""
        y_true, scores = perfect_separation_data
        pauc_10 = compute_pauc(y_true, scores, max_fpr=0.1)
        pauc_05 = compute_pauc(y_true, scores, max_fpr=0.05)
        # Both should work without error
        assert 0.0 <= pauc_10 <= 1.0
        assert 0.0 <= pauc_05 <= 1.0

    def test_pauc_default_fpr(self):
        """Default pAUC max_fpr should be 0.1."""
        assert DEFAULT_PAUC_MAX_FPR == 0.1

    def test_pr_auc_perfect(self, perfect_separation_data):
        """Perfect separation should give high PR AUC."""
        y_true, scores = perfect_separation_data
        pr_auc = compute_precision_recall_auc(y_true, scores)
        assert pr_auc > 0.99


class TestThresholdSelection:
    """Tests for threshold selection utilities."""

    @pytest.fixture
    def separable_data(self):
        """Data with clear separation."""
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        scores = np.concatenate(
            [
                np.random.normal(0.3, 0.1, 50),
                np.random.normal(0.7, 0.1, 50),
            ]
        )
        return y_true, scores

    def test_optimal_threshold_f1(self, separable_data):
        """F1-optimal threshold should be between class centers."""
        y_true, scores = separable_data
        threshold = find_optimal_threshold(y_true, scores, method="f1")
        assert 0.3 < threshold < 0.7

    def test_optimal_threshold_youden(self, separable_data):
        """Youden threshold should be between class centers."""
        y_true, scores = separable_data
        threshold = find_optimal_threshold(y_true, scores, method="youden")
        assert 0.3 < threshold < 0.7

    def test_apply_threshold(self):
        """Threshold should correctly binarize scores."""
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        y_pred = apply_threshold(scores, threshold=0.5)
        np.testing.assert_array_equal(y_pred, [0, 0, 1, 1])

    def test_apply_threshold_edge_case(self):
        """Scores equal to threshold should be classified as 1."""
        scores = np.array([0.5])
        y_pred = apply_threshold(scores, threshold=0.5)
        assert y_pred[0] == 1

    def test_invalid_method_raises(self, separable_data):
        """Invalid method should raise ValueError."""
        y_true, scores = separable_data
        with pytest.raises(ValueError, match="Unknown method"):
            find_optimal_threshold(y_true, scores, method="invalid")


class TestStratifiedEvaluation:
    """Tests for stratified evaluation functions."""

    @pytest.fixture
    def stratified_data(self):
        """Data with multiple SNR values and classes."""
        np.random.seed(42)
        n = 60
        y_true = np.array([0] * 30 + [1] * 30)
        y_pred = np.array([0] * 25 + [1] * 5 + [1] * 25 + [0] * 5)
        snr = np.array([10, 20, 30] * 20)
        original_labels = np.array([0, 1, 2, 3, 4, 5] * 5 + [6, 7, 8] * 10)
        scores = np.random.rand(n)
        return y_true, y_pred, snr, original_labels, scores

    def test_evaluate_by_snr(self, stratified_data):
        """Should compute metrics for each SNR value."""
        y_true, y_pred, snr, _, scores = stratified_data
        results = evaluate_by_snr(y_true, y_pred, snr, scores)

        assert set(results.keys()) == {10, 20, 30}
        for snr_val in results:
            assert "precision" in results[snr_val]
            assert "recall" in results[snr_val]
            assert "f1" in results[snr_val]

    def test_evaluate_by_snr_without_scores(self, stratified_data):
        """Should work without scores."""
        y_true, y_pred, snr, _, _ = stratified_data
        results = evaluate_by_snr(y_true, y_pred, snr)

        for snr_val in results:
            assert "roc_auc" not in results[snr_val]

    def test_evaluate_by_anomaly_class(self, stratified_data):
        """Should compute metrics for each anomaly class."""
        y_true, y_pred, _, original_labels, scores = stratified_data
        results = evaluate_by_anomaly_class(y_true, y_pred, original_labels, scores)

        assert set(results.keys()) == {6, 7, 8}
        for cls in results:
            assert "precision" in results[cls]
            assert "n_samples" in results[cls]


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def plot_data(self):
        """Data for plotting tests."""
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        scores = np.concatenate(
            [
                np.random.normal(0.3, 0.1, 50),
                np.random.normal(0.7, 0.1, 50),
            ]
        )
        y_pred = (scores > 0.5).astype(int)
        return y_true, y_pred, scores

    def test_roc_curve_runs(self, plot_data):
        """ROC curve plotting should complete without error."""
        y_true, _, scores = plot_data
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        ax = plot_roc_curve(y_true, scores)
        assert ax is not None

    def test_pr_curve_runs(self, plot_data):
        """PR curve plotting should complete without error."""
        y_true, _, scores = plot_data
        import matplotlib

        matplotlib.use("Agg")

        ax = plot_precision_recall_curve(y_true, scores)
        assert ax is not None

    def test_confusion_matrix_runs(self, plot_data):
        """Confusion matrix plotting should complete without error."""
        y_true, y_pred, _ = plot_data
        import matplotlib

        matplotlib.use("Agg")

        ax = plot_confusion_matrix(y_true, y_pred)
        assert ax is not None

    def test_confusion_matrix_normalized(self, plot_data):
        """Normalized confusion matrix should work."""
        y_true, y_pred, _ = plot_data
        import matplotlib

        matplotlib.use("Agg")

        ax = plot_confusion_matrix(y_true, y_pred, normalize=True)
        assert ax is not None

    def test_score_distribution_runs(self, plot_data):
        """Score distribution plotting should complete without error."""
        y_true, _, scores = plot_data
        import matplotlib

        matplotlib.use("Agg")

        ax = plot_score_distribution(scores, y_true)
        assert ax is not None

    def test_roc_curve_with_label(self, plot_data):
        """ROC curve with custom label should work."""
        y_true, _, scores = plot_data
        import matplotlib

        matplotlib.use("Agg")

        ax = plot_roc_curve(y_true, scores, label="Test Model")
        assert ax is not None


class TestReportGeneration:
    """Tests for report generation functions."""

    @pytest.fixture
    def report_data(self):
        """Data for report generation tests."""
        np.random.seed(42)
        n = 100
        y_true = np.array([0] * 60 + [1] * 40)
        y_pred = np.array([0] * 55 + [1] * 5 + [1] * 35 + [0] * 5)
        scores = np.random.rand(n)
        snr = np.array([10, 20, 30] * 33 + [10])
        original_labels = np.array([0, 1, 2, 3, 4, 5] * 10 + [6] * 15 + [7] * 15 + [8] * 10)
        return y_true, y_pred, scores, snr, original_labels

    def test_generate_report_basic(self, report_data):
        """Basic report generation should work."""
        y_true, y_pred, _, _, _ = report_data
        report = generate_report(y_true, y_pred)

        assert "Precision" in report
        assert "Recall" in report
        assert "F1 Score" in report

    def test_generate_report_with_scores(self, report_data):
        """Report with scores should include AUC metrics."""
        y_true, y_pred, scores, _, _ = report_data
        report = generate_report(y_true, y_pred, scores=scores)

        assert "ROC AUC" in report
        assert "pAUC" in report
        assert "PR AUC" in report

    def test_generate_report_with_snr(self, report_data):
        """Report with SNR should include SNR breakdown."""
        y_true, y_pred, _, snr, _ = report_data
        report = generate_report(y_true, y_pred, snr=snr)

        assert "SNR" in report
        assert "10 dB" in report
        assert "20 dB" in report
        assert "30 dB" in report

    def test_generate_report_with_model_name(self, report_data):
        """Report should include model name in header."""
        y_true, y_pred, _, _, _ = report_data
        report = generate_report(y_true, y_pred, model_name="My Model")

        assert "My Model" in report

    def test_compare_models(self):
        """Model comparison should return DataFrame."""
        results = {
            "Model A": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
            "Model B": {"precision": 0.9, "recall": 0.6, "f1": 0.72},
        }
        df = compare_models(results)

        assert df.shape == (2, 3)
        assert "Model A" in df.index
        assert "Model B" in df.index
        assert "precision" in df.columns
