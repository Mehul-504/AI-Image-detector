import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.detectors.image_forensic_transformer_layer import adjust_forensic_with_metadata
from aidetector.schemas import CategoryScore


class ForensicMetadataAdjustmentTests(unittest.TestCase):
    def test_camera_metadata_reduces_forensic_risk(self):
        risk, conf, note = adjust_forensic_with_metadata(
            risk=82.0,
            confidence=70.0,
            metadata_score=CategoryScore(
                risk=12.0,
                confidence=85.0,
                rationale="camera exif present",
            ),
        )
        self.assertLess(risk, 50.0)
        self.assertLess(conf, 70.0)
        self.assertEqual(note, "camera_metadata_support")

    def test_ai_metadata_boosts_forensic_risk(self):
        risk, conf, note = adjust_forensic_with_metadata(
            risk=58.0,
            confidence=52.0,
            metadata_score=CategoryScore(
                risk=95.0,
                confidence=82.0,
                rationale="ai generator metadata",
            ),
        )
        self.assertGreater(risk, 60.0)
        self.assertGreater(conf, 52.0)
        self.assertEqual(note, "ai_metadata_support")


if __name__ == "__main__":
    unittest.main()
