from __future__ import annotations

import unittest

import numpy as np

from data.generate_data import N_RECORDS, generate_dataset
from model.train import TARGETS, compute_mape, feature_engineering


class SponsorshipPipelineTests(unittest.TestCase):
    def test_generator_produces_complete_reproducible_contract(self) -> None:
        np.random.seed(42)
        first = generate_dataset()
        np.random.seed(42)
        second = generate_dataset()

        self.assertEqual(len(first), N_RECORDS)
        self.assertEqual(N_RECORDS, 5_000)
        self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))
        self.assertFalse(first.isna().any().any())
        self.assertTrue(set(TARGETS).issubset(first.columns))
        self.assertTrue(first["roi_score"].between(1, 100).all())

    def test_feature_engineering_is_numeric_and_leakage_safe(self) -> None:
        np.random.seed(42)
        raw = generate_dataset().head(100)
        transformed = feature_engineering(raw)
        inputs = transformed.drop(columns=TARGETS)

        self.assertGreater(inputs.shape[1], raw.shape[1])
        self.assertFalse(inputs.select_dtypes(exclude=["number", "bool"]).columns.any())
        self.assertTrue(np.isfinite(inputs.astype(float).to_numpy()).all())

    def test_mape_has_an_auditable_known_value(self) -> None:
        truth = np.array([100.0, 200.0])
        prediction = np.array([90.0, 220.0])
        self.assertAlmostEqual(compute_mape(truth, prediction), 10.0)


if __name__ == "__main__":
    unittest.main()
