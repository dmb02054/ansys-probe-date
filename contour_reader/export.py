"""마커가 표시된 이미지를 라벨까지 포함해 흰 배경으로 합성."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

_FONTS: dict[int, ImageFont.ImageFont] = {}


def get_font(size: float) -> ImageFont.ImageFont:
    size = max(6, int(round(size)))
    if size not in _FONTS:
        font = None
        for name in ("malgun.ttf", "NanumGothic.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        _FONTS[size] = font or ImageFont.load_default(size)
    return _FONTS[size]


@dataclass
class ExportMarker:
    n: int
    x: int
    y: int
    label: str                             # 라벨에 표시할 값 (단위 제외)
    color: str | None                      # 매칭된 구간 색 (없으면 None)
    offset: tuple[float, float] | None     # 라벨 좌상단의 마커 기준 오프셋(이미지 px), None=자동


def auto_offset(bw: float, bh: float, width: float, height: float, x: float, y: float, gap: float):
    """라벨 박스(좌상단)의 마커 기준 오프셋: 기본 우상단, 경계를 넘으면 반대쪽."""
    dx, dy = gap, -gap - bh
    if x + dx + bw > width:
        dx = -gap - bw
    if y + dy < 0:
        dy = gap
    return dx, dy


def compose(img: Image.Image, markers: list[ExportMarker], unit: str = "",
            selected: int | None = None) -> Image.Image:
    """이미지 밖으로 옮긴 라벨도 잘리지 않도록 캔버스를 넓히고 배경은 흰색으로 통일."""
    s = 2 if max(img.size) < 600 else 1
    base = img.resize((img.width * s, img.height * s), Image.NEAREST) if s > 1 else img
    W, H = base.size
    font = get_font(15 * s)
    chip, pad = 14 * s, 7 * s
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    boxes = []
    for m in markers:
        text = f"#{m.n}  {m.label}" + (f" {unit}" if unit and m.color is not None else "")
        tb = measure.textbbox((0, 0), text, font=font)
        bw, bh = pad * 2 + chip + 7 * s + (tb[2] - tb[0]), max(chip, tb[3] - tb[1]) + pad * 2
        mx, my = (m.x + 0.5) * s, (m.y + 0.5) * s
        if m.offset is None:
            dx, dy = auto_offset(bw, bh, W, H, mx, my, 16 * s)
        else:
            dx, dy = m.offset[0] * s, m.offset[1] * s
        boxes.append((m, text, mx, my, mx + dx, my + dy, bw, bh))

    margin = 14 * s
    left = min([0] + [b[4] for b in boxes]) - margin
    top = min([0] + [b[5] for b in boxes]) - margin
    right = max([W] + [b[4] + b[6] for b in boxes]) + margin
    bottom = max([H] + [b[5] + b[7] for b in boxes]) + margin
    ox, oy = round(-left), round(-top)
    out = Image.new("RGB", (round(right - left), round(bottom - top)), "white")
    out.paste(base, (ox, oy))
    d = ImageDraw.Draw(out)
    d.rectangle((ox - 1, oy - 1, ox + W, oy + H), outline="#bbbbbb")

    for m, text, mx, my, bx, by, bw, bh in boxes:
        mx, my, bx, by = mx + ox, my + oy, bx + ox, by + oy
        cx, cy = min(max(mx, bx), bx + bw), min(max(my, by), by + bh)
        if (cx, cy) != (mx, my):
            d.line((mx, my, cx, cy), fill="white", width=4 * s)
            d.line((mx, my, cx, cy), fill="#333333", width=2 * s)
    for m, text, mx, my, *_ in boxes:
        mx, my, r = mx + ox, my + oy, 7 * s
        d.ellipse((mx - r, my - r, mx + r, my + r), outline="white", width=4 * s)
        d.ellipse((mx - r, my - r, mx + r, my + r), outline="#e00000", width=2 * s)
        d.line((mx - r - 4 * s, my, mx + r + 4 * s, my), fill="#e00000", width=s)
        d.line((mx, my - r - 4 * s, mx, my + r + 4 * s), fill="#e00000", width=s)
    for m, text, _, _, bx, by, bw, bh in boxes:
        bx, by = bx + ox, by + oy
        d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=6 * s, fill="#ffffff",
                            outline=m.color or "#e00000", width=(4 if m.n == selected else 3) * s)
        cy = by + bh / 2
        d.rectangle((bx + pad, cy - chip / 2, bx + pad + chip, cy + chip / 2),
                    fill=m.color or "#cccccc", outline="#555555")
        d.text((bx + pad + chip + 7 * s, cy), text, fill="#111111", font=font, anchor="lm")
    return out
