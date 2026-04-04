from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class KeyframeCandidate(BaseModel):
    event_name: str
    frame_index: int
    confidence: float


class KeyframeSelection(BaseModel):
    event_name: str
    frame_index: int
    confidence: float
    top_k_candidates: List[KeyframeCandidate] = Field(default_factory=list)


class PreprocessMeta(BaseModel):
    source_fps: float
    analysis_fps: int = 240
    stabilized: bool = True
    denoised: bool = True
    cropped_single_swing: bool = True
    screen_mode_corrected: bool = False


class PreprocessResult(BaseModel):
    analysis_id: str
    analysis_video: str
    preprocess_meta: PreprocessMeta
    analysis_frames: List[Dict[str, Any]]
    enhanced_local_frames: List[Dict[str, Any]] = Field(default_factory=list)


class ExtractResult(BaseModel):
    analysis_id: str
    keyframes: List[KeyframeSelection]
    a_status: Literal["pass", "fail"]
    fail_reasons: List[str] = Field(default_factory=list)


class RefineResult(BaseModel):
    analysis_id: str
    refined_keyframes: List[KeyframeSelection]
    b_status: Literal["pass", "low_trust"]
    fail_reasons: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: Literal["pass", "low_trust"]
    trust_level: Literal["high", "medium", "low"]
    keyframes: List[KeyframeSelection]
    fail_reasons: List[str] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    analysis_id: str
    analysis_video: str
    preprocess_meta: Dict[str, Any]
    analysis_frames: List[Dict[str, Any]]
    enhanced_local_frames: List[Dict[str, Any]] = Field(default_factory=list)


class RefineRequest(BaseModel):
    analysis_id: str
    analysis_video: str
    preprocess_meta: Dict[str, Any]
    analysis_frames: List[Dict[str, Any]]
    enhanced_local_frames: List[Dict[str, Any]]
    keyframes: List[Dict[str, Any]]
    confidence: Dict[str, float]
    fail_reasons: List[str]
