from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from services.opencv_impact_refine_service import select_best_impact_candidate
from services.pro_analysis_chain_service import (
    ProAnalysisArtifacts,
    ProAnalysisChainSettings,
    prepare_pro_analysis_artifacts,
)
from services.pro_public_result_service import pack_pro_public_result

logger = logging.getLogger(__name__)


@dataclass
class ProUnifiedApiBundle:
    artifacts: ProAnalysisArtifacts
    impact_refine: dict[str, Any] | None = None
    public_result: dict[str, Any] | None = None


class ProUnifiedApiService:
    """Facade for the refactored Pro API.

    Goal:
    - frontends call one Pro API only
    - backend may use many technologies internally
    - technical/provider/debug fields do not leak to the frontend payload
    """

    def __init__(self, settings: ProAnalysisChainSettings | None = None):
        self.settings = settings or ProAnalysisChainSettings()

    def prepare_inputs(
        self,
        input_video_path: str,
        work_dir: str,
        *,
        rough_impact_time_s: float | None = None,
    ) -> ProAnalysisArtifacts:
        artifacts = prepare_pro_analysis_artifacts(
            input_video_path,
            work_dir,
            rough_impact_time_s=rough_impact_time_s,
            settings=self.settings,
        )
        logger.info(
            "[ROLE=PRO_API] prepare_inputs analysis_video=%s frontend_video=%s impact_window=%s",
            artifacts.analysis_video_path,
            artifacts.frontend_video_path,
            artifacts.impact_window_video_path,
        )
        return artifacts

    def refine_impact(
        self,
        artifacts: ProAnalysisArtifacts,
        *,
        rough_impact_time_s: float | None,
    ) -> dict[str, Any] | None:
        if rough_impact_time_s is None:
            return None
        target_video = artifacts.impact_window_video_path or artifacts.analysis_video_path
        refine = select_best_impact_candidate(
            target_video,
            around_time_s=float(rough_impact_time_s),
        )
        logger.info(
            "[ROLE=PRO_API] refine_impact target_video=%s status=%s",
            target_video,
            refine.get('status'),
        )
        return refine

    def build_public_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        public = pack_pro_public_result(raw_result)
        logger.info(
            "[ROLE=PRO_API] build_public_result analysis_id=%s keyframes=%s",
            public.get('analysis_id'),
            len(list(public.get('keyframes') or [])),
        )
        return public

    def build_bundle(
        self,
        input_video_path: str,
        work_dir: str,
        *,
        rough_impact_time_s: float | None = None,
        raw_result: dict[str, Any] | None = None,
    ) -> ProUnifiedApiBundle:
        artifacts = self.prepare_inputs(
            input_video_path,
            work_dir,
            rough_impact_time_s=rough_impact_time_s,
        )
        impact_refine = self.refine_impact(
            artifacts,
            rough_impact_time_s=rough_impact_time_s,
        )
        public_result = self.build_public_result(raw_result or {}) if raw_result is not None else None
        logger.info(
            "[ROLE=PRO_API] bundle_ready work_dir=%s", artifacts.work_dir,
        )
        return ProUnifiedApiBundle(
            artifacts=artifacts,
            impact_refine=impact_refine,
            public_result=public_result,
        )

    @staticmethod
    def debug_snapshot(bundle: ProUnifiedApiBundle) -> dict[str, Any]:
        return {
            'artifacts': asdict(bundle.artifacts),
            'impact_refine': bundle.impact_refine,
            'public_result': bundle.public_result,
        }
