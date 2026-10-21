"""PIL 기반 그리기: 폰트, 거리-값 프로파일 차트."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .legend import Legend, fmt

_FONTS: dict[int, ImageFont.ImageFont] = {}
SERIES_COLORS = ["#d32f2f", "#f57c00", "#388e3c", "#00838f", "#1976d2", "#7b1fa2", "#c2185b", "#5d4037"]


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


def _runs(ys: np.ndarray):
    """NaN이 아닌 연속 구간의 (시작, 끝) 인덱스."""
    ok = np.isfinite(ys)
    start = None
    for i in range(len(ys) + 1):
        if i < len(ys) and ok[i]:
            if start is None:
                start = i
        elif start is not None:
            yield start, i
            start = None


def _nice_step(span: float, target: int = 6) -> float:
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def render_chart(series: list[dict], lg: Legend, xlabel: str, size: tuple[int, int],
                 title: str = "", band: dict | None = None, scale: float = 1.0) -> Image.Image:
    """series: [{label,color,xs,ys,width}], band: {xs,lo,hi} (최소~최대 음영). scale: 글꼴/여백 배율."""
    W, H = size[0] * 2, size[1] * 2
    ss = u = int(round(2 * scale))
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f, ft = get_font(11 * u), get_font(12 * u)
    multi = len(series) > 1
    px0, py0, px1, py1 = 78 * u, 28 * u, W - (44 * u if multi else 16 * u), H - 34 * u

    xmax = 0.0
    for s in series:
        if np.isfinite(s["xs"]).any():
            xmax = max(xmax, float(np.nanmax(s["xs"])))
    xmax = xmax or 1.0
    span = lg.vmax - lg.vmin

    def X(x):
        return px0 + x / xmax * (px1 - px0)

    def Y(v):
        return py1 - (v - lg.vmin) / span * (py1 - py0)

    # 좌측 범례 색 막대 + 눈금
    skip = max(1, math.ceil(lg.n / max(1, (py1 - py0) / (ss * 15))))
    for i in range(lg.n):
        lo, hi = lg.band_range(i)
        d.rectangle((px0 - 12 * ss, Y(hi), px0 - 3 * ss, Y(lo)), fill=lg.colors[i])
    for i in range(0, lg.n + 1, skip):
        v = lg.vmin + span * i / lg.n
        d.line((px0, Y(v), px1, Y(v)), fill="#e3e3e3", width=ss)
        d.text((px0 - 16 * ss, Y(v)), fmt(v), fill="#333", font=f, anchor="rm")

    nice = _nice_step(xmax)
    t = 0.0
    while t <= xmax + 1e-9:
        d.line((X(t), py0, X(t), py1), fill="#efefef", width=ss)
        d.line((X(t), py1, X(t), py1 + 4 * ss), fill="#555", width=ss)
        d.text((X(t), py1 + 6 * ss), fmt(t), fill="#333", font=f, anchor="mt")
        t += nice
    d.rectangle((px0, py0, px1, py1), outline="#888", width=ss)
    d.text(((px0 + px1) / 2, H - 4 * ss), xlabel, fill="#222", font=f, anchor="ms")
    if title:
        d.text((px0, 6 * ss), title, fill="#111", font=ft, anchor="lt")

    if band is not None:
        bx, lo, hi = band["xs"], band["lo"], band["hi"]
        for a, b in _runs(lo):
            if b - a < 2:
                continue
            top = [(X(x), Y(v)) for x, v in zip(bx[a:b], hi[a:b])]
            bottom = [(X(x), Y(v)) for x, v in zip(bx[a:b][::-1], lo[a:b][::-1])]
            d.polygon(top + bottom, fill="#cfe0f7")

    for s in series:
        w = int(s.get("width", 2) * ss)
        for a, b in _runs(s["ys"]):
            pts = [(X(x), Y(v)) for x, v in zip(s["xs"][a:b], s["ys"][a:b])]
            if len(pts) == 1:
                x, y = pts[0]
                d.ellipse((x - w, y - w, x + w, y + w), fill=s["color"])
            else:
                d.line(pts, fill=s["color"], width=w, joint="curve")

    if multi:
        for k, s in enumerate(series):
            y = py0 + 4 * ss + k * 16 * ss
            d.text((px1 + 8 * ss, y), s["label"], fill=s["color"], font=ft, anchor="lt")
    return im.resize(size, Image.LANCZOS)
