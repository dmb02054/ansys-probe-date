"""구간 색상 이미지에서 클릭 위치의 단일 값을 선형 보간으로 추정한다.

컨투어 이미지는 값이 '계단식 색상 구간'으로 표시되므로 색만으로는 구간(예: 10~12)까지만 알 수 있다.
클릭 지점에서 아래/위 구간과 맞닿은 경계까지의 거리를 재서 구간 안에서의 상대 위치를 구한다.
"""
from __future__ import annotations

import numpy as np

from .legend import Legend, Probe


def _dilate(a: np.ndarray) -> np.ndarray:
    """불리언 마스크를 8방향으로 1픽셀 확장한다 (영역 성장용, scipy 없이 구현)."""
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
    """마스크의 True 픽셀 중 (cx, cy) 에 가장 가까운 거리(px). 없으면 None."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    # 경계는 두 픽셀 사이에 있으므로 반 픽셀 보정
    return max(float(np.hypot(xs - cx, ys - cy).min()) - 0.5, 0.5)


def refine_probe(p: Probe, arr: np.ndarray, x: int, y: int, lg: Legend, radius: int,
                 band_map: np.ndarray | None = None) -> None:
    """같은 색 영역(검은 선 등은 통과)과 맞닿은 아래/위 구간까지의 거리로 값을 보간한다. p 를 직접 갱신한다.

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

    # 클릭 지점 주변 radius 범위만 잘라서 처리 (전체 이미지를 훑지 않아 빠르다)
    h, w = arr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    bmap = band_map[y0:y1, x0:x1] if band_map is not None else lg.band_map(arr[y0:y1, x0:x1])
    cx, cy = x - x0, y - y0
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
    # 같은 구간이거나 판단 불가(-1: 검은 선/글자)인 픽셀은 통과 가능, 반경 밖은 제외
    passable = ((bmap == i) | (bmap < 0)) & (np.hypot(xx - cx, yy - cy) <= radius)

    comp = np.zeros(bmap.shape, dtype=bool)   # 클릭 지점에서 퍼져나간 같은 색 영역
    comp[cy, cx] = True
    lo_mask = bmap == i - 1 if i > 0 else np.zeros_like(comp)            # 아래 구간 픽셀
    hi_mask = bmap == i + 1 if i < lg.n - 1 else np.zeros_like(comp)     # 위 구간 픽셀
    stop_at = None
    for k in range(radius):
        adj = _dilate(comp)
        # 아래/위 경계를 모두 만나면 조금만 더 확장하고 멈춘다
        if stop_at is None and (adj & lo_mask).any() and (adj & hi_mask).any():
            stop_at = int(k * 1.5) + 2   # 대각선 방향 최단거리를 놓치지 않도록 조금 더 확장
        grown = (adj & passable) | comp
        if (grown == comp).all() or (stop_at is not None and k >= stop_at):
            break
        comp = grown
    adj = _dilate(comp)

    # 같은 색 영역에 맞닿은 아래/위 구간 픽셀까지의 최단 거리
    d_lo = _nearest(adj & lo_mask, cx, cy) if i > 0 else None
    d_hi = _nearest(adj & hi_mask, cx, cy) if i < lg.n - 1 else None
    p.d_lo, p.d_hi = d_lo, d_hi
    lo, hi = lg.band_range(i)
    if d_lo is not None and d_hi is not None:
        # 양쪽 경계를 모두 찾음: 거리 비율로 구간 안의 값을 선형 보간
        p.value = lo + (hi - lo) * d_lo / (d_lo + d_hi)
        p.kind = "interp"
        p.label = f"{p.value:.2f}"
        p.text = f"{p.value:.2f}{unit}"
    else:
        # 한쪽 경계를 못 찾음(탐색 반경 부족 등): 구간 중앙값으로 근사
        p.value = (lo + hi) / 2
        p.kind = "center"
        p.label = f"~{p.value:.2f}"
        p.text = f"≈ {p.value:.2f}{unit}  (근사)"
