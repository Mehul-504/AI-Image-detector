import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.detectors.base import DetectionContext
from aidetector.detectors.image_forensic_transformer_layer import adjust_forensic_with_consensus
from aidetector.detectors.prnu_layer import PrnuLayerDetector
from aidetector.schemas import Category, CategoryScore, MediaType, OverrideHints


class ConsensusCalibrationTests(unittest.TestCase):
    def test_forensic_conflict_is_downweighted(self):
        risk, conf, note = adjust_forensic_with_consensus(
            risk=85.42,
            confidence=81.27,
            clip_score=CategoryScore(risk=48.31, confidence=24.07),
            frequency_score=CategoryScore(risk=25.93, confidence=56.06),
            spatial_score=CategoryScore(risk=28.02, confidence=42.98),
            tamper_score=CategoryScore(risk=36.86, confidence=51.75),
            watermark_score=CategoryScore(risk=11.99, confidence=38.32),
            metadata_score=CategoryScore(risk=43.0, confidence=45.0),
        )
        self.assertLess(risk, 45.0)
        self.assertLess(conf, 45.0)
        self.assertEqual(note, "cross_signal_conflict_strong")

    def test_forensic_consensus_support_can_strengthen_signal(self):
        risk, conf, note = adjust_forensic_with_consensus(
            risk=82.0,
            confidence=70.0,
            clip_score=CategoryScore(risk=89.0, confidence=82.0),
            frequency_score=CategoryScore(risk=74.0, confidence=62.0),
            spatial_score=CategoryScore(risk=70.0, confidence=58.0),
            tamper_score=CategoryScore(risk=68.0, confidence=61.0),
            watermark_score=CategoryScore(risk=50.0, confidence=20.0),
            metadata_score=CategoryScore(risk=50.0, confidence=15.0),
        )
        self.assertGreater(risk, 80.0)
        self.assertGreater(conf, 70.0)
        self.assertEqual(note, "cross_signal_support")

    def test_prnu_conflict_is_downweighted(self):
        detector = PrnuLayerDetector()
        context = DetectionContext(
            media_type=MediaType.IMAGE,
            media_uri=None,
            provided_scores={
                Category.FREQUENCY: CategoryScore(risk=25.93, confidence=56.06),
                Category.SPATIAL: CategoryScore(risk=28.02, confidence=42.98),
                Category.TAMPER_LOCALIZATION: CategoryScore(risk=36.86, confidence=51.75),
                Category.CLIP_ANOMALY: CategoryScore(risk=48.31, confidence=24.07),
                Category.METADATA: CategoryScore(risk=43.0, confidence=45.0),
                Category.WATERMARK: CategoryScore(risk=11.99, confidence=38.32),
            },
            override_hints=OverrideHints(),
        )
        adjusted = detector._apply_consensus_adjustment(
            CategoryScore(
                risk=77.52,
                confidence=76.67,
                rationale="prnu-like baseline",
            ),
            context,
        )
        self.assertLess(adjusted.risk, 45.0)
        self.assertLess(adjusted.confidence, 45.0)
        self.assertIn("cross_signal_conflict_strong", adjusted.rationale or "")


if __name__ == "__main__":
    unittest.main()
