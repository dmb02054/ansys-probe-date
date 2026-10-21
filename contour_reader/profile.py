"""마커 주변 색상 -> 거리별 값 프로파일 계산."""
from __future__ import annotations

import math

import numpy as np

from .legend import Legend

DIRECTIONS = [("→", 0), ("↗", 45), ("↑", 90), ("↖", 135), ("←", 180), ("↙", 225), ("↓", 270), ("↘", 315)]


def radial_profile(arr: np.ndarray, cx: int, cy: int, radius: int, lg: Legend) -> dict:
    """중심에서 거리 r(px)인 고리 위 픽셀들의 평균/최소/최대 값."""
    h, w = arr.shape[:2]
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    val = lg.value_map(arr[y0:y1, x0:x1])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.rint(np.hypot(xx - cx, yy - cy)).astype(int)
    ok = (r <= radius) & ~np.isnan(val)
    r, v = r[ok], val[ok]
    n = radius + 1
    counts = np.bincount(r, minlength=n)
    sums = np.bincount(r, weights=v, minlength=n)
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    np.minimum.at(lo, r, v)
    np.maximum.at(hi, r, v)
    empty = counts == 0
    mean = np.where(empty, np.nan, sums / np.maximum(counts, 1))
    lo[empty] = np.nan
    hi[empty] = np.nan
    return {"xs": np.arange(n, dtype=float), "mean": mean, "min": lo, "max": hi}


def _sample_line(arr: np.ndarray, x0: float, y0: float, x1: float, y1: float, lg: Legend):
    length = math.hypot(x1 - x0, y1 - y0)
    n = int(math.ceil(length)) + 1
    t = np.linspace(0, 1, n)
    px = np.rint(x0 + (x1 - x0) * t).astype(int)
    py = np.rint(y0 + (y1 - y0) * t).astype(int)
    h, w = arr.shape[:2]
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    vals = np.full(n, np.nan)
    if inside.any():
        vals[inside] = lg.value_map(arr[py[inside], px[inside]])
    return t * length, vals


def direction_profiles(arr: np.ndarray, cx: int, cy: int, radius: int,
                       lg: Legend) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for name, deg in DIRECTIONS:
        rad = math.radians(deg)
        xs, ys = _sample_line(arr, cx, cy, cx + radius * math.cos(rad), cy - radius * math.sin(rad), lg)
        out.append((name, xs, ys))
    return out


def line_profile(arr: np.ndarray, a: tuple[int, int], b: tuple[int, int], lg: Legend):
    return _sample_line(arr, a[0], a[1], b[0], b[1], lg)
