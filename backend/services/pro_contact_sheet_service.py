from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _decode_b64_image(payload: str) -> Image.Image | None:
    if not payload:
        return None
    raw = payload
    if ',' in raw and raw.startswith('data:image'):
        raw = raw.split(',', 1)[1]
    try:
        data = base64.b64decode(raw)
        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        return None


def build_pro_keyframe_contact_sheet(
    keyframes: list[dict[str, Any]],
    output_path: str | Path,
    *,
    columns: int = 4,
    background: tuple[int, int, int] = (16, 16, 16),
    label_height: int = 34,
    tile_width: int = 320,
) -> str:
    """Create a product-style keyframe overview board from backend keyframes.

    Intended for Pro first-screen overview: clean 2x4 / 3x3 boards with only
    phase label + timestamp, no technical overlays.
    """
    rows = []
    for kf in keyframes or []:
        img = _decode_b64_image(str(kf.get('image_base64') or ''))
        if img is None:
            continue
        rows.append(
            {
                'image': img,
                'label': str(kf.get('label_zh') or kf.get('label_en') or kf.get('phase') or '').strip(),
                'timestamp': float(kf.get('timestamp') or 0.0),
            }
        )
    if not rows:
        raise ValueError('No valid keyframe images provided')

    tile_height = int(rows[0]['image'].height * tile_width / rows[0]['image'].width)
    total_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        'RGB',
        (tile_width * columns, (tile_height + label_height) * total_rows),
        background,
    )
    draw = ImageDraw.Draw(sheet)

    for idx, row in enumerate(rows):
        r = idx // columns
        c = idx % columns
        x = c * tile_width
        y = r * (tile_height + label_height)
        thumb = row['image'].resize((tile_width, tile_height))
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y + tile_height, x + tile_width, y + tile_height + label_height), fill=(0, 0, 0))
        text = f"{row['label']}  {row['timestamp']:.2f}s"
        draw.text((x + 8, y + tile_height + 8), text, fill=(255, 255, 255))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return str(out)
