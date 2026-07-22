"""
Lite Modal worker surface.

Product video analysis uses ``POST /analyze/lite`` only (club + main AI in one server-side run).
``/analyze/club-detect*`` stay available for standalone callers (e.g. tools, legacy clients) on the
same Lite Modal origin — they are optional and not used by the main Lite analyze path.
"""

from fastapi import APIRouter

from routers import analyze as analyze_mod

router = APIRouter()

router.add_api_route("/lite", analyze_mod.analyze_lite, methods=["POST"])
router.add_api_route("/recalculate", analyze_mod.recalculate_prediction, methods=["POST"])
router.add_api_route("/vision-classic", analyze_mod.vision_classic_multipart, methods=["POST"])
router.add_api_route("/club-detect", analyze_mod.analyze_club_detect_multipart, methods=["POST"])
router.add_api_route("/club-detect-batch", analyze_mod.analyze_club_detect_batch_multipart, methods=["POST"])
