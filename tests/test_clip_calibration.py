import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.detectors.image_clip_layer import calibrate_ai_probability


class ClipCalibrationTests(unittest.TestCase):
    def test_extreme_ai_probability_is_compressed(self):
        risk, confidence = calibrate_ai_probability(ai_prob=0.99, real_prob=0.01)
        self.assertLess(risk, 90.0)
        self.assertLessEqual(confidence, 80.0)
        self.assertGreaterEqual(confidence, 20.0)

    def test_near_boundary_probability_stays_neutralish(self):
        risk, confidence = calibrate_ai_probability(ai_prob=0.55, real_prob=0.45)
        self.assertGreaterEqual(risk, 48.0)
        self.assertLessEqual(risk, 60.0)
        self.assertLess(confidence, 40.0)


if __name__ == "__main__":
    unittest.main()
