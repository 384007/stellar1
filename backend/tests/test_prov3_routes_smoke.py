"""
Pro v3 路由冒烟：防止 ``include_router`` 顺序错误等导致 /pro-v3 未挂到 app 上。

CI 只跑本文件即可快速发现「OpenAPI 里没有 pro-v3」类回归。
"""

from __future__ import annotations

import os
import unittest

# main 导入前需要 JWT_SECRET（auth 模块在 import 时会读）
os.environ.setdefault("JWT_SECRET", "test-secret-prov3-smoke")
# CI / 本地冒烟无 R2：跳过「必须 durable」门禁，否则空文件用例会在读 body 前 503
os.environ.setdefault("STELLAR_PROV3_REQUIRE_R2", "0")

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

EXPECTED_PRO_V3_PATHS = frozenset(
    {
        "/pro-v3/analyze",
        "/pro-v3/analyze/cancel",
        "/pro-v3/media/{analysis_id}/{filename}",
        "/pro-v3/keyframes/analyze",
        "/pro-v3/keyframes/extract",
        "/pro-v3/keyframes/preprocess",
        "/pro-v3/keyframes/refine",
    }
)


class TestProv3RoutesSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_openapi_has_exactly_seven_pro_v3_paths(self) -> None:
        schema = app.openapi()
        paths = schema.get("paths") or {}
        prov3 = {p for p in paths if p.startswith("/pro-v3")}
        self.assertEqual(
            prov3,
            EXPECTED_PRO_V3_PATHS,
            f"expected exactly {sorted(EXPECTED_PRO_V3_PATHS)}, got {sorted(prov3)}",
        )

    def test_analyze_missing_file_returns_422(self) -> None:
        """未带 multipart ``file`` 时 FastAPI 校验失败 → 422（与「空字节上传」400 区分）。"""
        r = self.client.post("/pro-v3/analyze")
        self.assertEqual(r.status_code, 422)
        body = r.json()
        self.assertIn("detail", body)

    def test_analyze_empty_file_returns_400(self) -> None:
        """0 字节文件：路由已注册且进入 handler，业务层拒绝空文件 → 400。"""
        r = self.client.post(
            "/pro-v3/analyze",
            files={"file": ("empty.mp4", b"", "video/mp4")},
            data={"screen_mode": "false"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("detail", r.json())


if __name__ == "__main__":
    unittest.main()
