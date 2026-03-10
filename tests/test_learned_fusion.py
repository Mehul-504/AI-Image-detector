import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.learned_fusion import (
    MODEL_PATH_ENV,
    extract_fusion_features,
    load_learned_fusion_model,
    predict_with_learned_fusion,
)
from aidetector.schemas import MediaType


class LearnedFusionTests(unittest.TestCase):
    def test_model_is_disabled_when_env_not_set(self):
        previous = os.environ.pop(MODEL_PATH_ENV, None)
        try:
            self.assertIsNone(load_learned_fusion_model())
        finally:
            if previous is not None:
                os.environ[MODEL_PATH_ENV] = previous

    def test_extract_features_includes_compression(self):
        features = extract_fusion_features({}, compression_score=0.72)
        self.assertIn("compression_score", features)
        self.assertAlmostEqual(features["compression_score"], 0.72, places=6)
        self.assertIn("confidence_mean", features)

    def test_predict_uses_temp_model_artifact(self):
        artifact = {
            "model_id": "test_model",
            "feature_names": ["compression_score", "confidence_mean"],
            "means": [0.5, 0.5],
            "stds": [0.2, 0.2],
            "weights": [1.0, -0.2],
            "bias": 0.1,
            "thresholds": {
                "default": {"risk_low": 0.4, "risk_high": 0.6, "min_confidence": 0.3},
                "high": {"risk_low": 0.45, "risk_high": 0.65, "min_confidence": 0.25},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = pathlib.Path(tmpdir) / "fusion.json"
            model_path.write_text(json.dumps(artifact), encoding="utf-8")
            previous = os.environ.get(MODEL_PATH_ENV)
            os.environ[MODEL_PATH_ENV] = str(model_path)
            try:
                prediction = predict_with_learned_fusion(
                    media_type=MediaType.IMAGE,
                    category_scores={},
                    aux_features={"compression_score": 0.8},
                )
            finally:
                if previous is None:
                    os.environ.pop(MODEL_PATH_ENV, None)
                else:
                    os.environ[MODEL_PATH_ENV] = previous

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertGreaterEqual(prediction.risk_01, 0.0)
        self.assertLessEqual(prediction.risk_01, 1.0)
        self.assertEqual(prediction.bucket, "high")
        self.assertIn(prediction.predicted_class, {"real", "ai_generated", "ai_edited"})
        self.assertAlmostEqual(sum(prediction.class_probabilities.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
