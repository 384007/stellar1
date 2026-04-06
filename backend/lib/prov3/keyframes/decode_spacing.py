"""Enforce minimum spacing between keyframe **decode indices** (analysis MP4 frame numbers).

When SwingNet confidence is poor, per-class argmax rows often sit at the start of the batch; the
code only enforces **row** monotonicity (+1), which can map to **~10 consecutive decode frames** at
240fps — thumbnails look identical even though ffmpeg extracted 8 different indices.
"""

from __future__ import annotations

import copy
from typing import Any

from lib.prov3.keyframes.constants import EVENT_SEQUENCE


def min_decode_gap_for_total(total_frames: int) -> int:
    """Minimum gap between consecutive events in decode space (~3% of clip, capped to fit 8 events)."""
    if total_frames <= 1:
        return 1
    hi = total_frames - 1
    ideal = max(12, int(round(total_frames * 0.03)))
    return max(1, min(ideal, max(1, hi // 9)))


def spread_keyframes_min_decode_gap(
    keyframes: list[dict[str, Any]],
    total_frames: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Sort by ``EVENT_SEQUENCE``, enforce monotonic ``frame_index`` with at least ``min_gap`` frames.

    If the model output is already spread enough, only nudges indices forward where needed.
    If outputs are clustered (common under low trust), spreads evenly across a window inside ``[0, hi]``.
    """
    if len(keyframes) != 8 or total_frames <= 0:
        return keyframes, False

    hi = max(0, int(total_frames) - 1)
    mg = min_decode_gap_for_total(int(total_frames))
    while mg > 1 and 7 * mg > hi:
        mg -= 1
    mg = max(1, mg)

    order = {name: i for i, name in enumerate(EVENT_SEQUENCE)}
    sorted_kfs = sorted(keyframes, key=lambda k: order.get(str(k.get("event_name")), 99))
    old = [max(0, min(int(k.get("frame_index", 0)), hi)) for k in sorted_kfs]

    span = old[-1] - old[0]
    clustered = span < 7 * mg

    if clustered:
        anchor = max(0, min(old[0], hi - 7 * mg))
        new_fis = [anchor + i * mg for i in range(8)]
    else:
        new_fis = list(old)
        for i in range(1, 8):
            new_fis[i] = max(new_fis[i], new_fis[i - 1] + mg)
        if new_fis[-1] > hi:
            new_fis[-1] = hi
            for i in range(6, -1, -1):
                new_fis[i] = min(new_fis[i], new_fis[i + 1] - mg)
            for i in range(6, -1, -1):
                new_fis[i] = max(0, new_fis[i])
            for i in range(1, 8):
                new_fis[i] = max(new_fis[i], new_fis[i - 1] + mg)
            if new_fis[-1] > hi:
                new_fis[-1] = hi
                for i in range(6, -1, -1):
                    new_fis[i] = min(new_fis[i], new_fis[i + 1] - mg)

    new_fis = [max(0, min(int(x), hi)) for x in new_fis]
    for i in range(1, 8):
        if new_fis[i] <= new_fis[i - 1]:
            new_fis[i] = min(new_fis[i - 1] + 1, hi)

    changed = new_fis != old
    for k, fi in zip(sorted_kfs, new_fis):
        k["frame_index"] = int(fi)

    return keyframes, changed


def clone_keyframes_shallow(keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy row dicts so preview-only transforms never mutate official SwingNet outputs."""
    return [copy.copy(x) for x in keyframes]


def spread_keyframes_for_preview_strip(
    keyframes: list[dict[str, Any]],
    total_frames: int,
) -> list[dict[str, Any]]:
    """Return a **new** keyframe list with decode-gap spread for UI strip / thumbnails only.

    Official ``frame_index`` values from A/B must never be overwritten for product accuracy;
    call this only when building ``preview_keyframes`` / contact-sheet visuals.
    """
    rows = clone_keyframes_shallow(keyframes)
    spread_keyframes_min_decode_gap(rows, int(total_frames))
    return rows
