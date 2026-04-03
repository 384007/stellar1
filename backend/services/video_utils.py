import cv2
import subprocess
import json
import logging
import shutil
from typing import Any, Optional

logger = logging.getLogger(__name__)
_FFPROBE_MISSING_LOGGED = False
_FFPROBE_AVAILABLE: Optional[bool] = None


def _has_ffprobe() -> bool:
    global _FFPROBE_AVAILABLE, _FFPROBE_MISSING_LOGGED
    if _FFPROBE_AVAILABLE is None:
        _FFPROBE_AVAILABLE = shutil.which("ffprobe") is not None
    if not _FFPROBE_AVAILABLE and not _FFPROBE_MISSING_LOGGED:
        logger.warning("ffprobe not found, skipping")
        _FFPROBE_MISSING_LOGGED = True
    return bool(_FFPROBE_AVAILABLE)


def read_frame_pose_pipeline(
    cap: cv2.VideoCapture,
    frame_idx: int,
    rotation: int,
) -> Optional[Any]:
    """Single canonical frame read for the pose + keyframe pipeline.

    ``extract_poses_from_video`` and smart keyframes **must** use this exact
    sequence (``CAP_PROP_POS_FRAMES`` + one ``read`` + ``apply_rotation``) so
    skeleton, labels, and JPEG thumbnails always refer to the same decoder
    output for a given ``frame_index``.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return None
    target = int(min(max(int(frame_idx), 0), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return apply_rotation(frame, rotation)


def get_video_rotation(video_path: str) -> int:
    """Detect video rotation from metadata. Returns 0, 90, 180, or 270."""

    # Method 1: OpenCV orientation metadata (requires OpenCV 4.5+)
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            prop = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            cap.release()
            if prop in (90, 180, 270):
                logger.info(f"Rotation from OpenCV metadata: {int(prop)}")
                return int(prop)
    except Exception:
        pass

    if not _has_ffprobe():
        logger.warning(f"No rotation metadata found for {video_path}")
        return 0

    # Method 2: ffprobe — check both tags.rotate and Display Matrix
    for tool_args in [
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", "-select_streams", "v:0", video_path],
    ]:
        try:
            result = subprocess.run(
                tool_args, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)

                # Check stream-level tags
                for stream in info.get("streams", []):
                    tags = stream.get("tags", {})
                    rot = tags.get("rotate", "0")
                    r = int(rot) % 360
                    if r in (90, 180, 270):
                        logger.info(f"Rotation from ffprobe stream tags: {r}")
                        return r

                    side_data = stream.get("side_data_list", [])
                    for sd in side_data:
                        if sd.get("side_data_type") == "Display Matrix":
                            rot_val = sd.get("rotation", 0)
                            r = (-int(rot_val)) % 360
                            if r in (90, 180, 270):
                                logger.info(f"Rotation from Display Matrix: {r}")
                                return r

                # Check format-level tags (some containers store it here)
                fmt_tags = info.get("format", {}).get("tags", {})
                rot = fmt_tags.get("rotate", "0")
                r = int(rot) % 360
                if r in (90, 180, 270):
                    logger.info(f"Rotation from ffprobe format tags: {r}")
                    return r
        except FileNotFoundError:
            break
        except Exception as e:
            logger.warning(f"ffprobe error: {e}")

    logger.warning(f"No rotation metadata found for {video_path}")
    return 0


def apply_rotation(frame, rotation: int):
    """Apply rotation to correct video orientation for display.

    The rotation value from metadata means 'rotate CW by this many degrees
    to display correctly.'
    """
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def collect_frames_at_indices_sequential(
    video_path: str,
    frame_indices: list[int],
    *,
    rotation: Optional[int] = None,
):
    """Decode from frame 0 forward and grab exact indices (avoids H.264 seek errors).

    ``cap.set(CAP_PROP_POS_FRAMES)`` + ``read()`` often returns the wrong frame
    on long-GOP MP4 (repeated address pose across different timestamps).
    Pose extraction already walked the file; keyframe JPEGs must match the same
    ``frame_index`` values — sequential read is slower but accurate.

    **Important:** ``VideoCapture.read()`` reuses the same buffer across calls.
    We must ``.copy()`` each stored frame; otherwise every slot aliases one array
    and thumbnails all look like the last decoded frame (or one frozen pose).
    """
    targets = sorted({int(x) for x in frame_indices if int(x) >= 0})
    if not targets:
        return {}
    if rotation is None:
        rotation = get_video_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    out: dict[int, Any] = {}
    ti = 0
    current = 0
    max_need = targets[-1]
    try:
        while ti < len(targets) and current <= max_need:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current == targets[ti]:
                fr = apply_rotation(frame, rotation)
                if fr is not None:
                    out[current] = fr.copy()
                ti += 1
            current += 1
    finally:
        cap.release()
    return out
