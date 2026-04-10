import tempfile
import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.video_upload_suffix import temp_suffix_from_url

router = APIRouter()


class PoseFromVideoRequest(BaseModel):
    video_url: str
    max_frames: int = 30
    mode: str = "lite"


class PoseFromFrameRequest(BaseModel):
    image_base64: str
    mode: str = "lite"


@router.post("/from-video")
async def extract_pose_video(req: PoseFromVideoRequest):
    try:
        from services.pose_service import extract_poses_from_video, pose_for_skeleton_render
        from services.hud_service import generate_hud_data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pose services unavailable: {e}")

    tmp_path = None
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(req.video_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download video")

        suffix = temp_suffix_from_url(req.video_url)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        poses, pose_quality_bundle = extract_poses_from_video(tmp_path, max_frames=req.max_frames)

        if not poses:
            raise HTTPException(status_code=422, detail="No poses detected in video")

        hud_frames = []
        for pose in poses:
            hud = generate_hud_data(pose_for_skeleton_render(pose), mode=req.mode)
            hud["frame_index"] = pose["frame_index"]
            hud["timestamp"] = pose["timestamp"]
            hud_frames.append(hud)

        return {
            "total_frames": len(poses),
            "poses": poses,
            "hud_frames": hud_frames,
            "pose_quality_bundle": pose_quality_bundle,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose extraction failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/from-frame")
async def extract_pose_frame(req: PoseFromFrameRequest):
    import base64
    import numpy as np
    import cv2

    try:
        from services.pose_service import extract_pose_from_frame
        from services.hud_service import generate_hud_data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pose services unavailable: {e}")

    try:
        img_bytes = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        pose = extract_pose_from_frame(frame)
        if pose is None:
            raise HTTPException(status_code=422, detail="No pose detected in image")

        hud = generate_hud_data(pose, mode=req.mode)

        return {"pose": pose, "hud": hud}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose extraction failed: {str(e)}")
