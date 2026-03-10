import pathlib
import sys
import unittest
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.fusion import fuse_to_verdict
from aidetector.schemas import Category, CategoryScore, MediaType, OverrideHints, Verdict


class FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_model_path = os.environ.pop("AIDETECTOR_FUSION_MODEL_PATH", None)

    def tearDown(self) -> None:
        if self._old_model_path is not None:
            os.environ["AIDETECTOR_FUSION_MODEL_PATH"] = self._old_model_path

    def test_hard_override_synthetic_from_provenance(self):
        result = fuse_to_verdict(
            media_type=MediaType.IMAGE,
            category_scores={
                Category.PROVENANCE: CategoryScore(risk=5, confidence=90),
                Category.FORENSIC_TRANSFORMER: CategoryScore(risk=10, confidence=85),
            },
            override_hints=OverrideHints(provenance_declares_synthetic_or_edited=True),
        )
        self.assertEqual(result.verdict, Verdict.LIKELY_SYNTHETIC_OR_EDITED)
        self.assertEqual(result.applied_override, "trusted_provenance_says_synthetic_or_edited")

    def test_low_confidence_defaults_to_suspicious(self):
        result = fuse_to_verdict(
            media_type=MediaType.IMAGE,
            category_scores={
                Category.CLIP_ANOMALY: CategoryScore(risk=20, confidence=20),
                Category.FORENSIC_TRANSFORMER: CategoryScore(risk=20, confidence=20),
            },
            override_hints=OverrideHints(),
        )
        self.assertEqual(result.verdict, Verdict.SUSPICIOUS)

    def test_high_risk_high_confidence_video_is_synthetic_or_edited(self):
        result = fuse_to_verdict(
            media_type=MediaType.VIDEO,
            category_scores={
                Category.FORENSIC_TRANSFORMER: CategoryScore(risk=92, confidence=90),
                Category.AUDIO_VISUAL_TRANSFORMER: CategoryScore(risk=88, confidence=86),
                Category.CLIP_ANOMALY: CategoryScore(risk=81, confidence=80),
            },
            override_hints=OverrideHints(),
        )
        self.assertEqual(result.verdict, Verdict.LIKELY_SYNTHETIC_OR_EDITED)
        self.assertGreaterEqual(result.overall_risk, 65)
        self.assertGreaterEqual(result.overall_confidence, 45)

    def test_weight_redistribution_when_signals_missing(self):
        result = fuse_to_verdict(
            media_type=MediaType.IMAGE,
            category_scores={
                Category.CLIP_ANOMALY: CategoryScore(risk=80, confidence=90),
                Category.FORENSIC_TRANSFORMER: CategoryScore(risk=80, confidence=90),
            },
            override_hints=OverrideHints(),
        )
        total_eff_weight = sum(contrib.effective_weight for contrib in result.category_contributions)
        self.assertAlmostEqual(total_eff_weight, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
