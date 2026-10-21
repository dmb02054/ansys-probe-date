"""마커가 표시된 이미지 + 판독 결과 패널 합성."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from .drawing import get_font
from .legend import Legend, fmt


@dataclass
class ExportMarker:
    n: int
    x: int
    y: int
    text: str                              # 전체 값 설명
    short: str                             # 라벨용 짧은 값
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


def _annotate(img: Image.Image, markers: list[ExportMarker], lg: Legend, selected: int | None) -> Image.Image:
    s = 2 if max(img.size) < 600 else 1
    base = img.resize((img.width * s, img.height * s), Image.NEAREST) if s > 1 else img.copy()
    W, H = base.size
    d = ImageDraw.Draw(base)
    font = get_font(15 * s)
    unit = f" {lg.unit}" if lg.unit else ""
    boxes = []
    for m in markers:
        label = f"#{m.n}  {m.short}" + ("" if m.color is None else unit)
        tb = d.textbbox((0, 0), label, font=font)
        chip, pad = 14 * s, 7 * s
        bw, bh = pad * 2 + chip + 7 * s + (tb[2] - tb[0]), max(chip, tb[3] - tb[1]) + pad * 2
        mx, my = (m.x + 0.5) * s, (m.y + 0.5) * s
        if m.offset is None:
            dx, dy = auto_offset(bw, bh, W, H, mx, my, 16 * s)
        else:
            dx, dy = m.offset[0] * s, m.offset[1] * s
        boxes.append((m, label, mx, my, mx + dx, my + dy, bw, bh, pad, chip))

    for m, _, mx, my, bx, by, bw, bh, *_ in boxes:
        cx, cy = min(max(mx, bx), bx + bw), min(max(my, by), by + bh)
        if (cx, cy) != (mx, my):
            d.line((mx, my, cx, cy), fill="white", width=4 * s)
            d.line((mx, my, cx, cy), fill="#333333", width=2 * s)
    for m, *_ in boxes:
        mx, my = (m.x + 0.5) * s, (m.y + 0.5) * s
        r = 7 * s
        d.ellipse((mx - r, my - r, mx + r, my + r), outline="white", width=4 * s)
        d.ellipse((mx - r, my - r, mx + r, my + r), outline="#e00000", width=2 * s)
        d.line((mx - r - 4 * s, my, mx + r + 4 * s, my), fill="#e00000", width=s)
        d.line((mx, my - r - 4 * s, mx, my + r + 4 * s), fill="#e00000", width=s)
    for m, label, _, _, bx, by, bw, bh, pad, chip in boxes:
        border = m.color or "#e00000"
        d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=6 * s, fill="#ffffff",
                            outline=border, width=(4 if m.n == selected else 3) * s)
        cy = by + bh / 2
        d.rectangle((bx + pad, cy - chip / 2, bx + pad + chip, cy + chip / 2),
                    fill=m.color or "#cccccc", outline="#555555")
        d.text((bx + pad + chip + 7 * s, cy), label, fill="#111111", font=font, anchor="lm")
    return base


def _panel(markers: list[ExportMarker], lg: Legend, chart_fn, caption: str, k: int) -> Image.Image:
    PW = 440 * k
    pad = 18 * k
    im = Image.new("RGB", (PW, 4000 * k), "#f6f7f9")
    d = ImageDraw.Draw(im)
    f_title, f_sub, f_txt, f_small = get_font(22 * k), get_font(13 * k), get_font(14 * k), get_font(12 * k)
    unit = f" {lg.unit}" if lg.unit else ""
    y = pad
    d.text((pad, y), "판독 결과", fill="#111", font=f_title, anchor="lt")
    y += 34 * k
    d.text((pad, y), f"범례 {fmt(lg.vmin)} ~ {fmt(lg.vmax)}{unit}  ·  {lg.n}구간", fill="#555", font=f_sub, anchor="lt")
    y += 30 * k

    d.text((pad, y), "범례 색상", fill="#111", font=f_txt, anchor="lt")
    y += 24 * k
    for i in range(lg.n - 1, -1, -1):
        d.rectangle((pad, y, pad + 30 * k, y + 16 * k), fill=lg.colors[i], outline="#666")
        d.text((pad + 40 * k, y + 8 * k), lg.short(i) + unit, fill="#222", font=f_small, anchor="lm")
        y += 20 * k
    y += 14 * k

    d.text((pad, y), f"마커 ({len(markers)})", fill="#111", font=f_txt, anchor="lt")
    y += 26 * k
    if not markers:
        d.text((pad, y), "마커 없음", fill="#888", font=f_small, anchor="lt")
        y += 24 * k
    for m in markers:
        row_h = 30 * k
        d.rounded_rectangle((pad, y, PW - pad, y + row_h), radius=5 * k, fill="white", outline="#dcdfe4")
        cy = y + row_h / 2
        d.rectangle((pad + 8 * k, cy - 8 * k, pad + 24 * k, cy + 8 * k), fill=m.color or "#cccccc", outline="#555")
        d.text((pad + 34 * k, cy), f"#{m.n}", fill="#111", font=f_txt, anchor="lm")
        d.text((pad + 70 * k, cy), m.short + ("" if m.color is None else unit), fill="#111", font=f_txt, anchor="lm")
        d.text((PW - pad - 8 * k, cy), f"({m.x}, {m.y})", fill="#888", font=f_small, anchor="rm")
        y += row_h + 5 * k

    if chart_fn is not None:
        y += 12 * k
        if caption:
            d.text((pad, y), caption, fill="#111", font=f_txt, anchor="lt")
            y += 26 * k
        cw = PW - 2 * pad
        ch = round(cw * 0.62)
        im.paste(chart_fn((cw, ch), k), (pad, y))
        d.rectangle((pad, y, pad + cw, y + ch), outline="#dcdfe4")
        y += ch
    return im.crop((0, 0, PW, y + pad))


def compose(img: Image.Image, markers: list[ExportMarker], lg: Legend, chart_fn=None,
            caption: str = "", selected: int | None = None) -> Image.Image:
    """chart_fn(size, scale) -> PIL 이미지 (프로파일 차트). None 이면 차트 생략."""
    annotated = _annotate(img, markers, lg, selected)
    k = 2 if max(img.size) < 600 else 1
    panel = _panel(markers, lg, chart_fn, caption, k)
    margin = 20 * k
    W = margin * 3 + annotated.width + panel.width
    H = margin * 2 + max(annotated.height, panel.height)
    out = Image.new("RGB", (W, H), "white")
    out.paste(annotated, (margin, margin))
    ImageDraw.Draw(out).rectangle(
        (margin - 1, margin - 1, margin + annotated.width, margin + annotated.height), outline="#bbbbbb")
    out.paste(panel, (margin * 2 + annotated.width, margin))
    return out
