from __future__ import annotations

from dataclasses import asdict
from typing import Any

from services.pro_analysis_chain_service import ProAnalysisArtifacts, ProAnalysisChainSettings
from services.pro_minimal_public_result_service import pack_pro_minimal_public_result
from services.pro_unified_api_service import ProUnifiedApiBundle, ProUnifiedApiService


class StellarProApiService(ProUnifiedApiService):
    """Canonical unified Pro API service entrypoint.

    Naming is product-first:
    - frontends should think in terms of one Pro API only
    - backend may use many internal technologies, but they stay hidden
    """

    def __init__(self, settings: ProAnalysisChainSettings | None = None):
        super().__init__(settings=settings)

    def build_minimal_public_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        return pack_pro_minimal_public_result(raw_result)

    def build_bundle(
        self,
        input_video_path: str,
        work_dir: str,
        *,
        rough_impact_time_s: float | None = None,
        raw_result: dict[str, Any] | None = None,
    ) -> ProUnifiedApiBundle:
        bundle = super().build_bundle(
            input_video_path,
            work_dir,
            rough_impact_time_s=rough_impact_time_s,
            raw_result=None,
        )
        if raw_result is not None:
            bundle.public_result = self.build_minimal_public_result(raw_result)
        return bundle

    @staticmethod
    def snapshot(bundle: ProUnifiedApiBundle) -> dict[str, Any]:
        return {
            'artifacts': asdict(bundle.artifacts),
            'impact_refine': bundle.impact_refine,
            'public_result': bundle.public_result,
        }

    async def analyze_full(
        self,
        input_video_path: str,
        work_dir: str,
        *,
        rough_impact_time_s: float | None = None,
        region: str = 'global',
    ) -> dict[str, Any]:
        """FFmpeg 240fps prep → motion chain → contact sheet → text-only report → minimal public JSON."""
        from services.stellar_pro_video_analysis_service import run_stellar_pro_video_analysis

        return await run_stellar_pro_video_analysis(
            input_video_path,
            work_dir,
            rough_impact_time_s=rough_impact_time_s,
            region=region,
            chain_settings=self.settings,
        )


def create_stellar_pro_api_service(
    settings: ProAnalysisChainSettings | None = None,
) -> StellarProApiService:
    return StellarProApiService(settings=settings)
