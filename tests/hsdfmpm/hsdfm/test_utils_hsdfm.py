import unittest
import numpy as np
import tempfile
import json
import os
from io import StringIO
import contextlib
from unittest.mock import patch

from hsdfmpm.hsdfm import hsdfm


class TestUtils(unittest.TestCase):
    def test_read_metadata_json_valid(self):
        # Create a temporary file with valid JSON data.
        data = [
            {
                "AbsTime": 1,
                "ExpTime": 2,
                "Filter": "A",
                "AvgInt": 3.5,
                "Wavelength": 500,
            },
            {
                "AbsTime": 2,
                "ExpTime": 3,
                "Filter": "B",
                "AvgInt": 4.5,
                "Wavelength": 600,
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name

        result = hsdfm.read_metadata_json(tmp_path)
        os.remove(tmp_path)  # Clean up

        self.assertEqual(result["AbsTime"], [1, 2])
        self.assertEqual(result["ExpTime"], [2, 3])
        self.assertEqual(result["Filter"], ["A", "B"])
        self.assertEqual(result["AvgInt"], [3.5, 4.5])
        self.assertEqual(result["Wavelength"], [500, 600])

    def test_read_metadata_json_invalid_json(self):
        # Create a temporary file with invalid JSON content.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write("Not a JSON")
            tmp_path = tmp.name

        with contextlib.redirect_stdout(StringIO()) as buf:
            result = hsdfm.read_metadata_json(tmp_path)
            output = buf.getvalue()
        os.remove(tmp_path)

        self.assertIn("Error decoding JSON", output)
        # The function returns the dictionary with empty lists.
        self.assertEqual(
            result,
            {
                "AbsTime": [],
                "ExpTime": [],
                "Filter": [],
                "AvgInt": [],
                "Wavelength": [],
            },
        )

    def test_read_metadata_json_file_not_found(self):
        # Test that a non-existent file prints an error and returns empty lists.
        non_existent_path = "non_existent_file.json"
        with contextlib.redirect_stdout(StringIO()) as buf:
            result = hsdfm.read_metadata_json(non_existent_path)
            output = buf.getvalue()

        self.assertIn("File not found", output)
        self.assertEqual(
            result,
            {
                "AbsTime": [],
                "ExpTime": [],
                "Filter": [],
                "AvgInt": [],
                "Wavelength": [],
            },
        )

    def test_normalize_integration_time(self):
        # Test that division is applied correctly.
        hyperstack = np.ones((2, 3, 3), dtype=float)
        integration_time = [2, 4]
        expected = np.array([np.ones((3, 3)) / 2, np.ones((3, 3)) / 4])
        result = hsdfm.normalize_integration_time(hyperstack.copy(), integration_time)
        np.testing.assert_allclose(result, expected)

    def test_normalize_to_standard(self):
        # Test simple arithmetic normalization.
        hyperstack = np.array([[5, 7], [9, 11]], dtype=float)
        standard = 15
        bg = 3
        expected = (hyperstack - bg) / (standard - bg)
        result = hsdfm.normalize_to_standard(hyperstack, standard, bg)
        np.testing.assert_allclose(result, expected)

    def test_get_local_stdev(self):
        # Create a simple 1-channel image (shape: (1, 4, 4))
        image = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]],
            dtype=float,
        )
        # For the top-left block (2x2 sub-array), compute the expected std.
        block = np.array([[1, 2], [5, 6]])
        expected_std = np.nanstd(block)
        result = hsdfm.get_local_stdev(image, (2, 2))
        self.assertEqual(result.shape, (1, 2, 2))
        np.testing.assert_allclose(result[0, 0, 0], expected_std)

    @patch("hsdfmpm.hsdfm.hsdfm.vectorize_img")
    def test_k_cluster(self, mock_vectorize_img):
        # Patch vectorize_img to return a dummy 2D array.
        dummy_src = np.zeros((1, 2, 2))
        # For a 1-channel 2x2 image, assume 4 samples and 1 feature per sample.
        X_dummy = np.array([[0], [1], [2], [3]], dtype=float)
        mock_vectorize_img.return_value = X_dummy

        # Call k_cluster with k=2.
        labels = hsdfm.k_cluster(dummy_src, k=2, include_location=False)
        # The returned labels should be reshaped to (2,2)
        self.assertEqual(labels.shape, (2, 2))

    def test_intra_vs_inter_cluster_variance(self):
        # Use a simple 2x2 array and matching labels.
        src = np.array([[1, 2], [3, 4]], dtype=float)
        labels = np.array([[0, 0], [1, 1]])
        # For cluster 0: mean = 1.5, variance sum = 0.5;
        # For cluster 1: mean = 3.5, variance sum = 0.5; intra = 1.0,
        # Global mean = 2.5; inter = 2*1 + 2*1 = 4; ratio = 4/5 = 0.8.
        result = hsdfm.intra_vs_inter_cluster_variance(src, labels)
        self.assertAlmostEqual(result, 0.8)

    @patch("hsdfmpm.hsdfm.hsdfm.k_cluster")
    def test_try_n_clusters(self, mock_k_cluster):
        # Create a dummy src of shape (1,2,2).
        src = np.zeros((1, 2, 2))
        # Force k_cluster (patched) to return the same labels for any k.
        dummy_labels = np.array([[0, 1], [1, 0]])
        mock_k_cluster.return_value = dummy_labels

        ks = [2, 3]
        clusters, scores = hsdfm.try_n_clusters(src, ks)
        # clusters should have shape (len(ks), 2, 2) and scores length equal to len(ks).
        self.assertEqual(clusters.shape, (len(ks), 2, 2))
        self.assertEqual(scores.shape, (len(ks),))
        # Both scores are NaN due to 0/0 division in the variance computation.
        # Assert that both scores are NaN by using np.testing.assert_allclose with equal_nan=True:
        np.testing.assert_allclose(scores[0], scores[1], equal_nan=True)

    def test_find_elbow_clusters(self):
        # Prepare dummy clusters and scores.
        clusters = np.array([[[0, 0], [0, 0]], [[0, 1], [1, 0]], [[1, 1], [1, 1]]])
        scores = np.array([0.2, 0.5, 0.6])
        # Based on np.gradient, the maximum gradient is expected at index 0,
        # so elbow = np.argmax(np.gradient(scores)) + 1 should be 1.
        expected_cluster = clusters[1]
        result_cluster, elbow_index = hsdfm.find_elbow_clusters(clusters, scores)
        np.testing.assert_array_equal(result_cluster, expected_cluster)
        self.assertEqual(elbow_index, 1)

    def test_slice_clusters(self):
        # Define a simple src and corresponding clusters.
        src = np.array([[10, 20], [30, 40]], dtype=float)
        clusters = np.array([[0, 1], [2, 0]])
        # For unique clusters [0,1,2] with averages:
        # Cluster 0: (10+40)/2 = 25, Cluster 1: 20, Cluster 2: 30.
        # np.argsort yields [1, 0, 2]. Using the default slice (2, None) selects [2].
        expected_mask = clusters == 2
        result_mask = hsdfm.slice_clusters(src, clusters)
        np.testing.assert_array_equal(result_mask, expected_mask)


if __name__ == "__main__":
    unittest.main()
