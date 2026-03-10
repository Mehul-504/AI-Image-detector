import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.api import analyze_payload
from aidetector.run_logger import RunLogger


class RunLoggingTests(unittest.TestCase):
    def test_analyze_payload_persists_run_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_log_dir = os.environ.get("AIDETECTOR_LOG_DIR")
            os.environ["AIDETECTOR_LOG_DIR"] = tmpdir
            try:
                result = analyze_payload({"media_type": "image", "provided_scores": {}}, source="test")
                self.assertIn("run_id", result)
                self.assertIn("log_file", result)

                log_file = pathlib.Path(result["log_file"])
                self.assertTrue(log_file.exists())
                content = json.loads(log_file.read_text(encoding="utf-8"))
                self.assertEqual(content["run_id"], result["run_id"])
                self.assertEqual(content["source"], "test")
                self.assertEqual(content["payload"]["media_type"], "image")

                index_file = pathlib.Path(tmpdir) / "runs_index.jsonl"
                self.assertTrue(index_file.exists())
                lines = index_file.read_text(encoding="utf-8").splitlines()
                self.assertGreaterEqual(len(lines), 1)
            finally:
                if old_log_dir is None:
                    os.environ.pop("AIDETECTOR_LOG_DIR", None)
                else:
                    os.environ["AIDETECTOR_LOG_DIR"] = old_log_dir

    def test_store_upload_uses_given_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(root=tmpdir)
            path = logger.store_upload("sample image.png", b"abc", run_id="run123")
            self.assertTrue(path.exists())
            self.assertIn("run123_sample_image.png", str(path))


if __name__ == "__main__":
    unittest.main()
