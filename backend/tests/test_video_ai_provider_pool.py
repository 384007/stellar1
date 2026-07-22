import unittest
from unittest.mock import patch

from services import gemini_service as gs


class TestVideoAiProviderPool(unittest.TestCase):
    def setUp(self):
        gs._nvidia_rr_cursor = 0

    def test_collects_patentpaper_nvidia_key_names(self):
        env = {
            "NVIDIA_API_KEY": "k1",
            "NVIDIA_API_KEY_2": "k2",
            "NVIDIA_KEY_3": "k3",
            "NVIDIA_API_KEYS": "k4,k2\nk5",
        }
        with patch.dict("os.environ", env, clear=True):
            out = gs._collect_nvidia_api_keys()
        self.assertEqual([x[1] for x in out], ["k1", "k2", "k3", "k4", "k5"])

    def test_nvidia_video_default_ignores_text_model(self):
        with patch.dict("os.environ", {"NVIDIA_MODEL": "nvidia/llama-3.3-nemotron-super-49b-v1"}, clear=True):
            self.assertEqual(gs._nvidia_video_model(), gs.NVIDIA_VIDEO_MODEL_DEFAULT)

    def test_nvidia_default_models_are_hosted_video_candidates(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "k1"}, clear=True):
            models = [p["model"] for p in gs._ordered_video_ai_providers()]
        self.assertEqual(models, list(gs.NVIDIA_VIDEO_MODEL_CANDIDATES))
        self.assertNotIn("qwen/qwen3.6-35b-a3b", models)

    def test_known_unhosted_nvidia_model_env_falls_back_to_candidates(self):
        env = {
            "NVIDIA_API_KEY": "k1",
            "NVIDIA_VIDEO_MODEL": "qwen/qwen3.6-35b-a3b",
        }
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(gs._nvidia_video_models(), list(gs.NVIDIA_VIDEO_MODEL_CANDIDATES))

    def test_explicit_nvidia_video_model_uses_only_that_model(self):
        env = {
            "NVIDIA_API_KEY": "k1",
            "NVIDIA_VIDEO_MODEL": "nvidia/nemotron-nano-12b-v2-vl",
        }
        with patch.dict("os.environ", env, clear=True):
            models = [p["model"] for p in gs._ordered_video_ai_providers()]
        self.assertEqual(models, ["nvidia/nemotron-nano-12b-v2-vl"])

    def test_nvidia_provider_order_round_robins(self):
        env = {
            "NVIDIA_API_KEY": "k1",
            "NVIDIA_API_KEY_2": "k2",
            "NVIDIA_API_KEY_3": "k3",
        }
        with patch.dict("os.environ", env, clear=True):
            first = [p["label"] for p in gs._ordered_video_ai_providers()[:3]]
            second = [p["label"] for p in gs._ordered_video_ai_providers()[:3]]
        self.assertEqual(first, ["nvidia_key", "nvidia_key2", "nvidia_key3"])
        self.assertEqual(second, ["nvidia_key2", "nvidia_key3", "nvidia_key"])


if __name__ == "__main__":
    unittest.main()
