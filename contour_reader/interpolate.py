"""구간 색상 이미지에서 클릭 위치의 단일 값을 선형 보간으로 추정."""
from __future__ import annotations

import numpy as np

from .legend import Legend, Probe


def _dilate(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[1:] |= a[:-1]
    out[:-1] |= a[1:]
    out[:, 1:] |= a[:, :-1]
    out[:, :-1] |= a[:, 1:]
    out[1:, 1:] |= a[:-1, :-1]
    out[1:, :-1] |= a[:-1, 1:]
    out[:-1, 1:] |= a[1:, :-1]
    out[:-1, :-1] |= a[1:, 1:]
    return out


def _nearest(mask: np.ndarray, cx: int, cy: int) -> float | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    # 경계는 두 픽셀 사이에 있으므로 반 픽셀 보정
    return max(float(np.hypot(xs - cx, ys - cy).min()) - 0.5, 0.5)


def refine_probe(p: Probe, arr: np.ndarray, x: int, y: int, lg: Legend, radius: int,
                 band_map: np.ndarray | None = None) -> None:
    """같은 색 영역(검은 선 등은 통과)과 맞닿은 아래/위 구간까지의 거리로 값을 보간.

    value = lo + (hi - lo) * d_lo / (d_lo + d_hi)   (lo/hi: 현재 구간의 하/상한)
    band_map: 이미지 전체의 구간 번호 캐시(있으면 재계산 생략).
    """
    i = p.idx
    if i is None:
        return
    unit = f" {lg.unit}" if lg.unit else ""
    if lg.open_ends and i in (0, lg.n - 1):
        p.kind = "open"   # 범례 최대/최소 밖의 값이 같은 색으로 표시되므로 보간 불가
        return

    h, w = arr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    bmap = band_map[y0:y1, x0:x1] if band_map is not None else lg.band_map(arr[y0:y1, x0:x1])
    cx, cy = x - x0, y - y0
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
    passable = ((bmap == i) | (bmap < 0)) & (np.hypot(xx - cx, yy - cy) <= radius)

    comp = np.zeros(bmap.shape, dtype=bool)
    comp[cy, cx] = True
    lo_mask = bmap == i - 1 if i > 0 else np.zeros_like(comp)
    hi_mask = bmap == i + 1 if i < lg.n - 1 else np.zeros_like(comp)
    stop_at = None
    for k in range(radius):
        adj = _dilate(comp)
        if stop_at is None and (adj & lo_mask).any() and (adj & hi_mask).any():
            stop_at = int(k * 1.5) + 2   # 대각선 방향 최단거리를 놓치지 않도록 조금 더 확장
        grown = (adj & passable) | comp
        if (grown == comp).all() or (stop_at is not None and k >= stop_at):
            break
        comp = grown
    adj = _dilate(comp)

    d_lo = _nearest(adj & lo_mask, cx, cy) if i > 0 else None
    d_hi = _nearest(adj & hi_mask, cx, cy) if i < lg.n - 1 else None
    p.d_lo, p.d_hi = d_lo, d_hi
    lo, hi = lg.band_range(i)
    if d_lo is not None and d_hi is not None:
        p.value = lo + (hi - lo) * d_lo / (d_lo + d_hi)
        p.kind = "interp"
        p.label = f"{p.value:.2f}"
        p.text = f"{p.value:.2f}{unit}"
    else:
        p.value = (lo + hi) / 2
        p.kind = "center"
        p.label = f"~{p.value:.2f}"
        p.text = f"≈ {p.value:.2f}{unit}  (경계 미탐지: 구간 중앙값)"
