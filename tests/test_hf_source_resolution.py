import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aidetector.detectors.utils import resolve_transformers_source, transformers_source_candidates


class HFSourceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
        self._old_hf_home = os.environ.get("HF_HOME")
        self._old_local_only = os.environ.get("AIDETECTOR_HF_LOCAL_ONLY")

    def tearDown(self) -> None:
        if self._old_hf_cache is None:
            os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
        else:
            os.environ["HUGGINGFACE_HUB_CACHE"] = self._old_hf_cache

        if self._old_hf_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = self._old_hf_home

        if self._old_local_only is None:
            os.environ.pop("AIDETECTOR_HF_LOCAL_ONLY", None)
        else:
            os.environ["AIDETECTOR_HF_LOCAL_ONLY"] = self._old_local_only

    def test_resolves_explicit_local_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = pathlib.Path(tmpdir) / "clip-local"
            model_dir.mkdir(parents=True)
            source, kwargs, note = resolve_transformers_source(str(model_dir))
            self.assertEqual(source, str(model_dir.resolve()))
            self.assertEqual(kwargs.get("local_files_only"), True)
            self.assertEqual(note, "local_path")

    def test_resolves_hf_cache_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = pathlib.Path(tmpdir)
            snap = hub / "models--openai--clip-vit-base-patch32" / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}", encoding="utf-8")
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
            os.environ.pop("HF_HOME", None)
            source, kwargs, note = resolve_transformers_source("openai/clip-vit-base-patch32")
            self.assertEqual(source, str(snap))
            self.assertEqual(kwargs.get("local_files_only"), True)
            self.assertEqual(note, "hf_cache_snapshot")

    def test_remote_id_uses_local_only_when_flag_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(pathlib.Path(tmpdir) / "none")
            os.environ["AIDETECTOR_HF_LOCAL_ONLY"] = "1"
            source, kwargs, note = resolve_transformers_source("openai/clip-vit-base-patch32")
            self.assertEqual(source, "openai/clip-vit-base-patch32")
            self.assertEqual(kwargs.get("local_files_only"), True)
            self.assertEqual(note, "remote_id_local_only")

    def test_candidates_include_cache_then_remote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = pathlib.Path(tmpdir)
            snap = hub / "models--openai--clip-vit-base-patch32" / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}", encoding="utf-8")
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
            os.environ.pop("HF_HOME", None)
            os.environ.pop("AIDETECTOR_HF_LOCAL_ONLY", None)
            candidates = transformers_source_candidates("openai/clip-vit-base-patch32")
            self.assertGreaterEqual(len(candidates), 2)
            self.assertEqual(candidates[0][0], str(snap))
            self.assertEqual(candidates[0][2], "hf_cache_snapshot")
            self.assertEqual(candidates[-1][0], "openai/clip-vit-base-patch32")
            self.assertEqual(candidates[-1][2], "remote_id")


if __name__ == "__main__":
    unittest.main()
