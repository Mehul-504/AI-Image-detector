import os
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.fusion import fuse_to_verdict
from aidetector.schemas import Category, CategoryScore, MediaType, ModeProfile, OverrideHints, Verdict


class ModeProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_model_path = os.environ.pop("AIDETECTOR_FUSION_MODEL_PATH", None)

    def tearDown(self) -> None:
        if self._old_model_path is not None:
            os.environ["AIDETECTOR_FUSION_MODEL_PATH"] = self._old_model_path

    def test_strict_profile_is_more_conservative_than_balanced(self):
        scores = {
            Category.CLIP_ANOMALY: CategoryScore(risk=38.0, confidence=90.0),
            Category.FORENSIC_TRANSFORMER: CategoryScore(risk=38.0, confidence=90.0),
        }
        balanced = fuse_to_verdict(
            media_type=MediaType.IMAGE,
            category_scores=scores,
            override_hints=OverrideHints(),
            mode_profile=ModeProfile.BALANCED,
        )
        strict = fuse_to_verdict(
            media_type=MediaType.IMAGE,
            category_scores=scores,
            override_hints=OverrideHints(),
            mode_profile=ModeProfile.STRICT,
        )
        self.assertEqual(balanced.verdict, Verdict.LIKELY_AUTHENTIC)
        self.assertIn(strict.verdict, {Verdict.SUSPICIOUS, Verdict.LIKELY_SYNTHETIC_OR_EDITED})


if __name__ == "__main__":
    unittest.main()
