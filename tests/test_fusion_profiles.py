import pathlib
import sys
import unittest
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.fusion import fuse_to_verdict
from aidetector.schemas import Category, CategoryScore, MediaType, OverrideHints, Verdict


class FusionProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_model_path = os.environ.pop("AIDETECTOR_FUSION_MODEL_PATH", None)

    def tearDown(self) -> None:
        if self._old_model_path is not None:
            os.environ["AIDETECTOR_FUSION_MODEL_PATH"] = self._old_model_path

    def test_camera_profile_trends_authentic(self):
        response = fuse_to_verdict(
            media_type=MediaType.IMAGE,
            category_scores={
                Category.METADATA: CategoryScore(risk=12.0, confidence=85.0),
                Category.CLIP_ANOMALY: CategoryScore(risk=44.8, confidence=32.68),
                Category.FORENSIC_TRANSFORMER: CategoryScore(risk=40.93, confidence=48.91),
            },
            override_hints=OverrideHints(),
        )
        self.assertEqual(response.verdict, Verdict.LIKELY_AUTHENTIC)

    def test_ai_generated_profile_trends_synthetic(self):
        response = fuse_to_verdict(
            media_type=MediaType.IMAGE,
            category_scores={
                Category.METADATA: CategoryScore(risk=50.0, confidence=15.0),
                Category.CLIP_ANOMALY: CategoryScore(risk=87.87, confidence=79.4),
                Category.FORENSIC_TRANSFORMER: CategoryScore(risk=87.85, confidence=87.36),
            },
            override_hints=OverrideHints(),
        )
        self.assertEqual(response.verdict, Verdict.LIKELY_SYNTHETIC_OR_EDITED)

    def test_mixed_signal_profile_can_be_likely_authentic_after_calibration(self):
        response = fuse_to_verdict(
            media_type=MediaType.IMAGE,
            category_scores={
                Category.METADATA: CategoryScore(risk=43.0, confidence=45.0),
                Category.CLIP_ANOMALY: CategoryScore(risk=21.91, confidence=58.13),
                Category.FORENSIC_TRANSFORMER: CategoryScore(risk=52.43, confidence=32.82),
            },
            override_hints=OverrideHints(),
        )
        self.assertEqual(response.verdict, Verdict.LIKELY_AUTHENTIC)


if __name__ == "__main__":
    unittest.main()
