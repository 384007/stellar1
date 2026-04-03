from __future__ import annotations

import os
import tempfile

from services.provider_registry import role_log
from services.provider_schema import provider_result


def _clip_topk_from_pred(pred, model, k: int = 5) -> list[dict]:
    out: list[dict] = []
    try:
        scores = pred.pred_score.tolist() if hasattr(pred.pred_score, "tolist") else list(pred.pred_score)
        labels = getattr(model, "dataset_meta", {}).get("classes", []) or []
        pairs = sorted(enumerate(scores), key=lambda x: -float(x[1]))[:k]
        for idx, sc in pairs:
            lab = str(labels[idx]) if idx < len(labels) else str(idx)
            out.append({"label": lab, "score": round(float(sc), 6)})
    except Exception:
        pass
    return out


def _sliding_window_inference(
    model,
    video_path: str,
    *,
    total_video_frames: int,
    pose_frame_count: int,
) -> list[dict]:
    """Optional short sub-clips along the timeline for temporal action signal."""
    import cv2

    raw = os.getenv("STELLAR_MMACTION2_SLIDING_WINDOWS")
    if raw is None or raw.strip() == "":
        nw = 3 if total_video_frames >= 20 else 0
    else:
        nw = max(0, int(raw.strip()))
    if nw < 2 or total_video_frames < 16:
        return []

    min_win = max(8, int(os.getenv("STELLAR_MMACTION2_MIN_WINDOW_FRAMES") or "8"))
    wlen = max(min_win, total_video_frames // (nw + 1))
    wlen = min(wlen, total_video_frames - 1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0:
        fps = 30.0

    preds: list[dict] = []
    tmpdir = tempfile.mkdtemp(prefix="stellar_mmaction2_win_")
    try:
        for i in range(nw):
            if nw <= 1:
                s = 0
            else:
                s = int(i * (total_video_frames - wlen) / max(nw - 1, 1))
            s = max(0, min(s, total_video_frames - wlen))
            e = min(total_video_frames - 1, s + wlen - 1)
            out_path = os.path.join(tmpdir, f"w{i}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer: cv2.VideoWriter | None = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(s))
            wrote = 0
            for _ in range(e - s + 1):
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                if writer is None:
                    h, w = fr.shape[:2]
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                writer.write(fr)
                wrote += 1
            if writer is not None:
                writer.release()
                writer = None
            if wrote < min_win:
                continue
            try:
                from mmaction.apis import inference_recognizer

                wp = inference_recognizer(model, out_path)
                label = "unknown"
                confidence = 0.0
                if hasattr(wp, "pred_score") and hasattr(wp, "pred_label"):
                    scores = wp.pred_score.tolist() if hasattr(wp.pred_score, "tolist") else list(wp.pred_score)
                    idx = int(wp.pred_label.item() if hasattr(wp.pred_label, "item") else int(wp.pred_label))
                    confidence = float(scores[idx]) if idx < len(scores) else 0.0
                    labels = getattr(model, "dataset_meta", {}).get("classes", []) or []
                    label = str(labels[idx]) if idx < len(labels) else str(idx)
                p0 = int(round(s / max(total_video_frames, 1) * max(pose_frame_count, 1)))
                p1 = int(round((e + 1) / max(total_video_frames, 1) * max(pose_frame_count, 1)))
                p0 = max(0, min(p0, max(pose_frame_count - 1, 0)))
                p1 = max(p0, min(p1, max(pose_frame_count - 1, 0)))
                preds.append(
                    {
                        "window_index": i,
                        "video_start_frame": s,
                        "video_end_frame": e,
                        "pose_start_idx": p0,
                        "pose_end_idx": p1,
                        "label": label,
                        "confidence": round(confidence, 6),
                    },
                )
            except Exception:
                continue
    finally:
        cap.release()
        try:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
    return preds


def run(video_path: str, frame_count: int) -> dict:
    try:
        from mmaction.apis import inference_recognizer, init_recognizer
    except Exception:
        role_log(f"[ROLE=MMACTION2] status=dependency_missing clip_len={frame_count}")
        return provider_result(
            role="action",
            provider_name="mmaction2",
            status="dependency_missing",
            frame_count=frame_count,
            payload={},
            error_reason="dependency_missing",
        )

    config = (os.getenv("STELLAR_MMACTION2_CONFIG") or "").strip()
    checkpoint = (os.getenv("STELLAR_MMACTION2_CHECKPOINT") or "").strip()
    label = "unknown"
    confidence = 0.0
    abnormal = False
    if not config or not checkpoint:
        role_log(f"[ROLE=MMACTION2] status=model_config_missing clip_len={frame_count}")
        return provider_result(
            role="action",
            provider_name="mmaction2",
            status="model_config_missing",
            frame_count=frame_count,
            payload={},
            error_reason="model_config_missing",
        )
    topk: list[dict] = []
    window_predictions: list[dict] = []
    try:
        model = init_recognizer(config, checkpoint, device=(os.getenv("STELLAR_MMACTION2_DEVICE") or "cpu"))
        pred = inference_recognizer(model, video_path)
        if hasattr(pred, "pred_score") and hasattr(pred, "pred_label"):
            scores = pred.pred_score.tolist() if hasattr(pred.pred_score, "tolist") else list(pred.pred_score)
            idx = int(pred.pred_label.item() if hasattr(pred.pred_label, "item") else int(pred.pred_label))
            confidence = float(scores[idx]) if idx < len(scores) else 0.0
            labels = getattr(model, "dataset_meta", {}).get("classes", []) or []
            label = str(labels[idx]) if idx < len(labels) else str(idx)
            topk = _clip_topk_from_pred(pred, model, k=5)
        abnormal = confidence < 0.2

        total_vf = frame_count
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                tc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if tc > 0:
                    total_vf = tc
            cap.release()
        except Exception:
            pass

        try:
            window_predictions = _sliding_window_inference(
                model,
                video_path,
                total_video_frames=total_vf,
                pose_frame_count=frame_count,
            )
        except Exception:
            window_predictions = []
    except Exception as exc:
        role_log(f"[ROLE=MMACTION2] status=inference_failed clip_len={frame_count} err={exc}")
        return provider_result(
            role="action",
            provider_name="mmaction2",
            status="inference_failed",
            frame_count=frame_count,
            payload={},
            error_reason=str(exc),
        )
    role_log(
        f"[ROLE=MMACTION2] status=ok clip_len={frame_count} action={label} confidence={confidence:.3f} "
        f"windows={len(window_predictions)}",
    )
    return provider_result(
        role="action",
        provider_name="mmaction2",
        status="ok",
        frame_count=frame_count,
        payload={
            "action_label": label,
            "action_confidence": confidence,
            "abnormal_flag": abnormal,
            "pred_scores_topk": topk,
            "window_predictions": window_predictions,
        },
    )
