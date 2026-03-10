import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.detectors.image_forensic_transformer_layer import (
    DEFAULT_IMAGE_FORENSIC_MODEL_ID,
    ImageForensicTransformerDetector,
)


class DetectorDefaultsTests(unittest.TestCase):
    def test_forensic_detector_uses_default_model_id_when_not_set(self):
        detector = ImageForensicTransformerDetector(
            model_id_or_path=None,
            preferred_device="cpu",
        )
        self.assertEqual(detector.model_id_or_path, DEFAULT_IMAGE_FORENSIC_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
