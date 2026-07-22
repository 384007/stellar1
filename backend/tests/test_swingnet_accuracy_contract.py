import sys
import types

import numpy as np
import pytest


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    class _FakeTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=np.float32)

        @property
        def shape(self):
            return self.arr.shape

        def permute(self, *axes):
            return _FakeTensor(np.transpose(self.arr, axes))

        def view(self, *shape):
            return _FakeTensor(self.arr.reshape(shape))

        def unsqueeze(self, axis):
            return _FakeTensor(np.expand_dims(self.arr, axis))

        def __sub__(self, other):
            other_arr = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr - other_arr)

        def __truediv__(self, other):
            other_arr = other.arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self.arr / other_arr)

    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_f = types.ModuleType("torch.nn.functional")

    class _Module:
        pass

    fake_nn.Module = _Module
    fake_nn.Sequential = object
    fake_nn.LSTM = object
    fake_nn.Linear = object
    fake_nn.Dropout = object
    fake_nn.Conv2d = object
    fake_nn.BatchNorm2d = object
    fake_nn.ReLU6 = object
    fake_f.softmax = lambda x, dim=0: x
    fake_torch.nn = fake_nn
    fake_torch.Tensor = _FakeTensor
    fake_torch.from_numpy = lambda arr: _FakeTensor(arr)
    fake_torch.tensor = lambda arr: _FakeTensor(arr)
    fake_torch.device = lambda name: name
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda: None, __exit__=lambda *args: False)
    fake_torch.load = lambda *_args, **_kwargs: {}

    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.nn.functional"] = fake_f

from services import golfdb_swingnet_service as swingnet
from services.lite_ab_mirror import a_adapter


class _DummyCap:
    def __init__(self, total: int = 1000, fps: float = 240.0):
        self.total = total
        self.fps = fps

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == swingnet.cv2.CAP_PROP_FRAME_COUNT:
            return self.total
        if prop == swingnet.cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def set(self, _prop, _value):
        return True

    def release(self):
        return None


def test_swingnet_low_sample_cap_is_raised(monkeypatch):
    monkeypatch.setenv("STELLAR_SWINGNET_MIN_ACCURATE_FRAMES", "64")
    monkeypatch.setattr(swingnet.cv2, "VideoCapture", lambda _path: _DummyCap())
    monkeypatch.setattr(
        swingnet,
        "_read_letterbox_rgb",
        lambda _cap, _input_size: np.zeros((160, 160, 3), dtype=np.uint8),
    )

    batch, sample_indices, _fps, total = swingnet._video_to_batch("demo.mp4", max_frames=64)

    assert total == 1000
    assert batch.shape[1] >= 480
    assert len(sample_indices) >= 480


def test_swingnet_viterbi_does_not_argmax_fallback_on_invalid_rows():
    probs = np.ones((7, 8), dtype=np.float32) / 8.0
    sample_indices = np.arange(7, dtype=np.int64)

    with pytest.raises(RuntimeError, match="swingnet_viterbi_too_few_rows"):
        swingnet._keyframes_from_probs_viterbi(probs, sample_indices)


def test_lite_adapter_raises_low_env_cap_to_accuracy_floor(monkeypatch):
    captured = {}

    def fake_extract(*_args, **kwargs):
        captured["max_extract_frames"] = kwargs["max_extract_frames"]
        return [{"event_name": str(i), "frame_index": i, "confidence": 0.9} for i in range(8)]

    monkeypatch.setenv("STELLAR_SWINGNET_LITE_MAX_FRAMES", "64")
    monkeypatch.setattr(a_adapter, "swingnet_enabled", lambda: True)
    monkeypatch.setattr(a_adapter, "swingnet_checkpoint_path", lambda: "/models/swingnet_1800.pth.tar")
    monkeypatch.setattr(a_adapter.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(a_adapter, "run_swingnet_extract", fake_extract)

    out = a_adapter.infer_lite_a_candidates(
        [],
        analysis_video="analysis.mp4",
        analysis_id="lite_test",
        preprocess_meta={"analysis_fps": 240},
    )

    assert len(out) == 8
    assert captured["max_extract_frames"] == 480


def test_lite_adapter_allows_higher_explicit_cap(monkeypatch):
    captured = {}

    def fake_extract(*_args, **kwargs):
        captured["max_extract_frames"] = kwargs["max_extract_frames"]
        return [{"event_name": str(i), "frame_index": i, "confidence": 0.9} for i in range(8)]

    monkeypatch.setenv("STELLAR_SWINGNET_LITE_MAX_FRAMES", "2400")
    monkeypatch.setattr(a_adapter, "swingnet_enabled", lambda: True)
    monkeypatch.setattr(a_adapter, "swingnet_checkpoint_path", lambda: "/models/swingnet_1800.pth.tar")
    monkeypatch.setattr(a_adapter.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(a_adapter, "run_swingnet_extract", fake_extract)

    out = a_adapter.infer_lite_a_candidates(
        [],
        analysis_video="analysis.mp4",
        analysis_id="lite_test",
        preprocess_meta={"analysis_fps": 240},
    )

    assert len(out) == 8
    assert captured["max_extract_frames"] == 2400
