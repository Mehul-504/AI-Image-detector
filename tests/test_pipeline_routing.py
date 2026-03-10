import pathlib
import sys
import unittest
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.pipeline import AuthenticityPipeline
from aidetector.schemas import AnalysisRequest, Category, CategoryScore, MediaType


class PipelineRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_model_path = os.environ.pop("AIDETECTOR_FUSION_MODEL_PATH", None)
        self.pipeline = AuthenticityPipeline()

    def tearDown(self) -> None:
        if self._old_model_path is not None:
            os.environ["AIDETECTOR_FUSION_MODEL_PATH"] = self._old_model_path

    def test_image_precomputed_transformer_scores_are_used(self):
        request = AnalysisRequest(
            media_type=MediaType.IMAGE,
            provided_scores={
                Category.CLIP_ANOMALY: CategoryScore(risk=88, confidence=90, rationale="test"),
                Category.FORENSIC_TRANSFORMER: CategoryScore(risk=79, confidence=84, rationale="test"),
            },
        )
        response = self.pipeline.analyze(request)
        self.assertEqual(response.category_scores[Category.CLIP_ANOMALY].risk, 88)
        self.assertEqual(response.category_scores[Category.FORENSIC_TRANSFORMER].risk, 79)

    def test_image_without_media_uri_falls_back_to_neutral_model_scores(self):
        request = AnalysisRequest(media_type=MediaType.IMAGE)
        response = self.pipeline.analyze(request)
        self.assertIn(Category.CLIP_ANOMALY, response.category_scores)
        self.assertIn(Category.FORENSIC_TRANSFORMER, response.category_scores)
        self.assertEqual(response.category_scores[Category.CLIP_ANOMALY].risk, 50.0)
        self.assertEqual(response.category_scores[Category.FORENSIC_TRANSFORMER].risk, 50.0)

    def test_video_keeps_placeholder_video_transformer_scores(self):
        request = AnalysisRequest(media_type=MediaType.VIDEO)
        response = self.pipeline.analyze(request)
        self.assertIn(Category.AUDIO_VISUAL_TRANSFORMER, response.category_scores)
        self.assertIn(Category.FORENSIC_TRANSFORMER, response.category_scores)
        self.assertEqual(response.category_scores[Category.AUDIO_VISUAL_TRANSFORMER].risk, 50.0)


if __name__ == "__main__":
    unittest.main()
