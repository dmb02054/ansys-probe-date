"""범례(색상 구간) 모델: 색상 -> 값 구간 변환."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 낮은 값 -> 높은 값 순서 (첨부 이미지의 Temperature 범례 기본 색상)
DEFAULT_COLORS = [
    "#3b3181", "#4675eb", "#2cb7ed", "#20e7b0", "#71fd60",
    "#c0f235", "#f3c63a", "#fb8524", "#e14209", "#a51401",
]


def fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb)


def resample_colors(colors: list[str], n: int) -> list[str]:
    """색상 리스트를 n개로 선형 보간."""
    if n == len(colors):
        return list(colors)
    src = np.array([hex_to_rgb(c) for c in colors], dtype=float)
    xs = np.linspace(0, 1, len(src))
    xn = np.linspace(0, 1, n)
    out = np.stack([np.interp(xn, xs, src[:, k]) for k in range(3)], axis=1)
    return [rgb_to_hex(c) for c in out]


@dataclass
class Probe:
    idx: int | None          # 매칭된 구간 번호(0=최소 구간), 없으면 None
    rgb: tuple[int, int, int]
    distance: float
    confidence: float        # 샘플 영역 중 해당 구간 비율
    text: str                # 화면 표시용 값 문자열
    center: float | None = None
    label: str = ""          # 마커 라벨용 짧은 값
    value: float | None = None   # 보간된 단일 값
    kind: str = "band"       # band(구간) | interp(보간) | center(경계 미탐지) | open(최대/최소 밖)
    d_lo: float | None = None    # 아래/위 구간 경계까지의 거리(px)
    d_hi: float | None = None


class Legend:
    def __init__(self, vmin: float, vmax: float, colors: list[str],
                 tolerance: float = 40.0, open_ends: bool = True, unit: str = "",
                 ignore_black: bool = True, black_threshold: float = 70.0,
                 edges: dict[int, float] | None = None, shade: float = 0.0):
        """edges: {경계 번호: 값}. 경계 i는 구간 i-1과 i 사이(0=최소값, n=최대값).
        지정하지 않은 경계는 지정된 값들 사이를 선형 보간한다.
        shade: 0~100, 밝기(그림자/조명) 차이를 무시하는 정도. 0이면 순수 RGB 거리."""
        if len(colors) < 2:
            raise ValueError("색상은 2개 이상 필요합니다.")
        n = len(colors)
        known = {0: float(vmin), n: float(vmax)}
        for k, v in (edges or {}).items():
            if 0 < k < n:
                known[int(k)] = float(v)
        ks = sorted(known)
        vals = [known[k] for k in ks]
        if any(b <= a for a, b in zip(vals, vals[1:])):
            raise ValueError("범례 값은 아래에서 위로 갈수록 커야 합니다 (최소 < 중간값 < 최대).")
        self.vmin, self.vmax = vmin, vmax
        self.edges = np.interp(np.arange(n + 1), ks, vals)
        self.colors = list(colors)
        self.tolerance = tolerance
        self.open_ends = open_ends
        self.unit = unit
        self.ignore_black = ignore_black
        self.black_threshold = black_threshold
        self.shade = min(max(float(shade), 0.0), 100.0)
        self._palette = np.array([hex_to_rgb(c) for c in colors], dtype=float)

    @property
    def n(self) -> int:
        return len(self.colors)

    def band_range(self, i: int) -> tuple[float, float]:
        return float(self.edges[i]), float(self.edges[i + 1])

    def describe(self, i: int) -> str:
        lo, hi = self.band_range(i)
        u = f" {self.unit}" if self.unit else ""
        if self.open_ends and i == self.n - 1:
            return f"≥ {fmt(lo)}{u}  (최대 {fmt(self.vmax)} 이상)"
        if self.open_ends and i == 0:
            return f"≤ {fmt(hi)}{u}  (최소 {fmt(self.vmin)} 이하)"
        return f"{fmt(lo)} ~ {fmt(hi)}{u}"

    def short(self, i: int) -> str:
        lo, hi = self.band_range(i)
        if self.open_ends and i == self.n - 1:
            return f"≥ {fmt(lo)}"
        if self.open_ends and i == 0:
            return f"≤ {fmt(hi)}"
        return f"{fmt(lo)} ~ {fmt(hi)}"

    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2

    @staticmethod
    def _hsv_parts(rgb: np.ndarray):
        """(밝기 V, 순색 벡터(0~255, 흰색 성분 제거·정규화), 채도 S)."""
        v = rgb.max(axis=1)
        pure = rgb - rgb.min(axis=1, keepdims=True)
        hue = pure / np.maximum(pure.max(axis=1), 1.0)[:, None] * 255.0
        return v, hue, (v - rgb.min(axis=1)) / np.maximum(v, 1.0)

    def black_mask(self, pixels: np.ndarray) -> np.ndarray:
        """검은색(선/테두리) 픽셀 여부. 범례에 어두운 색이 있으면 그보다 낮게 제한."""
        if not self.ignore_black:
            return np.zeros(pixels.shape[:-1], dtype=bool)
        limit = min(self.black_threshold, float(self._palette.max(axis=1).min()) - 1)
        return pixels.max(axis=-1) <= limit

    def nearest(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(...,3) RGB -> (가장 가까운 구간 번호, 색상 거리). 검은색은 거리 무한대(판단 제외)."""
        px = pixels.reshape(-1, 3).astype(float)
        d = np.empty((len(px), self.n))
        if self.shade > 0:
            # 그림자는 색을 밝기 배수로, 조명(반사광)은 흰색을 섞는 방향으로 바꾼다.
            # 그래서 흰색 성분을 뺀 뒤 밝기를 정규화한 '색상'으로 비교하고,
            # 밝기·채도 차이는 (1 - shade)만큼만 반영한다.
            w = 1.0 - self.shade / 100.0
            vpx, hpx, spx = self._hsv_parts(px)
            vpal, hpal, spal = self._hsv_parts(self._palette)
            for k in range(self.n):
                d[:, k] = np.sqrt(((hpx - hpal[k]) ** 2).sum(axis=1)
                                  + (w * (vpx - vpal[k])) ** 2 + (w * 255.0 * (spx - spal[k])) ** 2)
            d[spx < 0.12] = np.inf   # 무채색(배경/문자)은 색상 비교가 불안정하므로 제외
        else:
            for k in range(self.n):
                d[:, k] = np.sqrt(((px - self._palette[k]) ** 2).sum(axis=1))
        shape = pixels.shape[:-1]
        dmin = d.min(axis=1)
        dmin[self.black_mask(px)] = np.inf
        return d.argmin(axis=1).reshape(shape), dmin.reshape(shape)

    def band_map(self, pixels: np.ndarray) -> np.ndarray:
        """구간 번호 배열. 허용오차를 넘는 픽셀은 -1."""
        idx, dmin = self.nearest(pixels)
        return np.where(dmin <= self.tolerance, idx, -1)

    def value_map(self, pixels: np.ndarray) -> np.ndarray:
        """구간 중앙값 배열. 범례 색상이 아닌 픽셀은 NaN."""
        idx = self.band_map(pixels)
        return np.where(idx >= 0, self.centers()[np.maximum(idx, 0)], np.nan)

    def classify(self, pixels: np.ndarray) -> Probe:
        """pixels: (N,3) RGB. 허용오차 내 픽셀들의 다수결로 구간 결정."""
        px = pixels.reshape(-1, 3).astype(float)
        idx, dmin = self.nearest(px)
        ok = dmin <= self.tolerance
        med = tuple(int(v) for v in np.median(px, axis=0))
        if not ok.any():
            msg = "검은색(선)만 있음 — 판단 제외" if self.black_mask(px).all() else "범례 색상 아님 (배경/문자)"
            return Probe(None, med, float(dmin.min()), 0.0, msg, label="범례 외")
        counts = np.bincount(idx[ok], minlength=self.n)
        best = int(counts.argmax())
        sel = ok & (idx == best)
        lo, hi = self.band_range(best)
        return Probe(best, med, float(dmin[sel].mean()), float(sel.sum() / len(px)),
                     self.describe(best), (lo + hi) / 2, label=self.short(best))


def sample_colors(arr: np.ndarray, box: tuple[int, int, int, int], n: int) -> list[str]:
    """드래그한 범례 막대에서 n개 색상 추출. 세로면 위=높은 값, 가로면 오른쪽=높은 값."""
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, x0), min(arr.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(arr.shape[0], y1)))
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        raise ValueError("영역이 너무 작습니다.")
    vertical = h >= w
    length = h if vertical else w
    cross0, cross1 = ((x0, x1) if vertical else (y0, y1))
    c_mid = (cross0 + cross1) // 2
    c_half = max(1, (cross1 - cross0) // 6)
    out = []
    for k in range(n):
        pos = int((k + 0.5) * length / n)
        half = max(1, length // n // 4)
        a, b = max(0, pos - half), pos + half + 1
        if vertical:
            patch = arr[y0 + a:y0 + b, c_mid - c_half:c_mid + c_half + 1]
        else:
            patch = arr[c_mid - c_half:c_mid + c_half + 1, x0 + a:x0 + b]
        out.append(rgb_to_hex(np.median(patch.reshape(-1, 3), axis=0)))
    return out[::-1] if vertical else out
