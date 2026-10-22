"""범례(색상 구간) 모델.

컨투어 이미지의 범례는 '색상 N개 = 값 구간 N개' 로 이루어져 있다.
이 모듈은 다음을 담당한다.
  - 이미지 픽셀 색상 -> 가장 가까운 범례 구간 번호 찾기 (그림자/조명 보정 포함)
  - 구간 번호 -> 값 범위(경계값) 변환. 경계값은 사용자가 일부만 입력해도 나머지를 선형 보간
  - 이미지 위의 범례 막대에서 색상 추출
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 낮은 값 -> 높은 값 순서의 기본 범례 색상 (Temperature 범례 10단계)
DEFAULT_COLORS = [
    "#3b3181", "#4675eb", "#2cb7ed", "#20e7b0", "#71fd60",
    "#c0f235", "#f3c63a", "#fb8524", "#e14209", "#a51401",
]


def fmt(v: float) -> str:
    """숫자를 소수 3자리까지 표시하되 불필요한 0과 '-0' 은 제거한다."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    """'#rrggbb' 문자열을 (R, G, B) 정수 튜플로 변환."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    """(R, G, B) 값을 '#rrggbb' 문자열로 변환 (소수는 반올림)."""
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb)


def resample_colors(colors: list[str], n: int) -> list[str]:
    """색상 리스트를 n개로 선형 보간해 구간 수를 바꿀 때 사용한다."""
    if n == len(colors):
        return list(colors)
    src = np.array([hex_to_rgb(c) for c in colors], dtype=float)
    xs = np.linspace(0, 1, len(src))   # 기존 색상 위치 (0~1)
    xn = np.linspace(0, 1, n)          # 새 색상 위치 (0~1)
    out = np.stack([np.interp(xn, xs, src[:, k]) for k in range(3)], axis=1)
    return [rgb_to_hex(c) for c in out]


@dataclass
class Probe:
    """이미지의 한 지점을 판독한 결과."""
    idx: int | None          # 매칭된 구간 번호(0=최소 구간), 없으면 None
    rgb: tuple[int, int, int]    # 샘플 영역의 대표(중앙값) 색상
    distance: float          # 범례 색상과의 평균 색상 거리
    confidence: float        # 샘플 영역 중 해당 구간으로 판정된 픽셀 비율
    text: str                # 화면 표시용 값 문자열 (단위 포함)
    label: str = ""          # 마커 라벨용 짧은 값
    value: float | None = None   # 보간된 단일 값
    kind: str = "band"       # band(구간) | interp(보간) | center(경계 미탐지) | open(최대/최소 밖)
    d_lo: float | None = None    # 아래 구간 경계까지의 거리(px)
    d_hi: float | None = None    # 위 구간 경계까지의 거리(px)


class Legend:
    """색상 리스트 + 경계값으로 이루어진 범례."""

    def __init__(self, vmin: float, vmax: float, colors: list[str],
                 tolerance: float = 40.0, open_ends: bool = True, unit: str = "",
                 ignore_black: bool = True, black_threshold: float = 70.0,
                 edges: dict[int, float] | None = None, shade: float = 0.0):
        """
        vmin, vmax    : 범례의 최소/최대값 (경계 0 과 경계 n)
        colors        : 낮은 값 -> 높은 값 순서의 '#rrggbb' 리스트 (구간 수 = 길이)
        tolerance     : 이 색상 거리 이하일 때만 범례 색으로 인정
        open_ends     : 양 끝 구간을 '이상/이하' 로 표시할지 여부
        ignore_black  : 검은 선(테두리)을 판단에서 제외할지 여부
        black_threshold: 픽셀의 최대 채널값이 이 값 이하이면 검은색으로 간주
        edges         : {경계 번호: 값}. 경계 i 는 구간 i-1 과 i 사이(0=최소, n=최대).
                        지정하지 않은 경계는 지정된 값들 사이를 선형 보간한다.
        shade         : 0~100. 그림자/조명으로 인한 밝기 차이를 무시하는 정도. 0 이면 순수 RGB 거리.
        """
        if len(colors) < 2:
            raise ValueError("색상은 2개 이상 필요합니다.")
        n = len(colors)
        # 사용자가 지정한 경계값만 모은다 (최소/최대는 항상 포함)
        known = {0: float(vmin), n: float(vmax)}
        for k, v in (edges or {}).items():
            if 0 < k < n:
                known[int(k)] = float(v)
        ks = sorted(known)
        vals = [known[k] for k in ks]
        if any(b <= a for a, b in zip(vals, vals[1:])):
            raise ValueError("범례 값은 아래에서 위로 갈수록 커야 합니다 (최소 < 중간값 < 최대).")
        self.vmin, self.vmax = vmin, vmax
        # 비어 있는 경계는 지정된 값 사이를 선형 보간해 길이 n+1 의 경계 배열을 만든다
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
        """구간(색상) 수."""
        return len(self.colors)

    def band_range(self, i: int) -> tuple[float, float]:
        """구간 i 의 (하한, 상한) 값."""
        return float(self.edges[i]), float(self.edges[i + 1])

    def describe(self, i: int) -> str:
        """구간 i 의 설명 문자열 (단위 포함). 양 끝은 '이상/이하' 로 표시할 수 있다."""
        lo, hi = self.band_range(i)
        u = f" {self.unit}" if self.unit else ""
        if self.open_ends and i == self.n - 1:
            return f"≥ {fmt(lo)}{u}  (최대 {fmt(self.vmax)} 이상)"
        if self.open_ends and i == 0:
            return f"≤ {fmt(hi)}{u}  (최소 {fmt(self.vmin)} 이하)"
        return f"{fmt(lo)} ~ {fmt(hi)}{u}"

    def short(self, i: int) -> str:
        """구간 i 의 짧은 표기 (목록/마커 라벨용, 단위 제외)."""
        lo, hi = self.band_range(i)
        if self.open_ends and i == self.n - 1:
            return f"≥ {fmt(lo)}"
        if self.open_ends and i == 0:
            return f"≤ {fmt(hi)}"
        return f"{fmt(lo)} ~ {fmt(hi)}"

    @staticmethod
    def _hsv_parts(rgb: np.ndarray):
        """그림자/조명 보정용 분해. (밝기 V, 순색 벡터, 채도 S) 를 돌려준다.

        순색 벡터는 흰색 성분(최소 채널)을 빼고 최대 채널을 255 로 정규화한 값으로,
        밝기가 배수로 달라지거나 흰빛이 섞여도 거의 변하지 않는다.
        """
        v = rgb.max(axis=1)
        pure = rgb - rgb.min(axis=1, keepdims=True)
        hue = pure / np.maximum(pure.max(axis=1), 1.0)[:, None] * 255.0
        return v, hue, (v - rgb.min(axis=1)) / np.maximum(v, 1.0)

    def black_mask(self, pixels: np.ndarray) -> np.ndarray:
        """검은색(선/테두리) 픽셀 여부. 범례에 어두운 색이 있으면 임계값을 그보다 낮게 제한한다."""
        if not self.ignore_black:
            return np.zeros(pixels.shape[:-1], dtype=bool)
        limit = min(self.black_threshold, float(self._palette.max(axis=1).min()) - 1)
        return pixels.max(axis=-1) <= limit

    def nearest(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(...,3) RGB -> (가장 가까운 구간 번호, 그 색상 거리). 검은색은 거리 무한대(판단 제외)."""
        px = pixels.reshape(-1, 3).astype(float)
        d = np.empty((len(px), self.n))   # 각 픽셀 x 각 범례 색상 사이의 거리
        if self.shade > 0:
            # 그림자는 색을 밝기 배수로, 조명(반사광)은 흰색을 섞는 방향으로 바꾼다.
            # 그래서 흰색 성분을 뺀 뒤 밝기를 정규화한 '색상'으로 비교하고,
            # 밝기·채도 차이는 (1 - shade) 비율만큼만 반영한다.
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
        """이미지 전체의 구간 번호 배열. 허용오차를 넘는 픽셀은 -1."""
        idx, dmin = self.nearest(pixels)
        return np.where(dmin <= self.tolerance, idx, -1)

    def classify(self, pixels: np.ndarray) -> Probe:
        """pixels: (N,3) RGB 샘플 영역. 허용오차 안의 픽셀들을 다수결로 투표해 구간을 정한다."""
        px = pixels.reshape(-1, 3).astype(float)
        idx, dmin = self.nearest(px)
        ok = dmin <= self.tolerance
        med = tuple(int(v) for v in np.median(px, axis=0))
        if not ok.any():
            msg = "검은색(선)만 있음 — 판단 제외" if self.black_mask(px).all() else "범례 색상 아님 (배경/문자)"
            return Probe(None, med, float(dmin.min()), 0.0, msg, label="범례 외")
        counts = np.bincount(idx[ok], minlength=self.n)
        best = int(counts.argmax())
        sel = ok & (idx == best)   # 가장 많이 득표한 구간으로 판정된 픽셀들
        return Probe(best, med, float(dmin[sel].mean()), float(sel.sum() / len(px)),
                     self.describe(best), label=self.short(best))


def sample_colors(arr: np.ndarray, box: tuple[int, int, int, int], n: int) -> list[str]:
    """드래그한 범례 막대에서 n개 색상 추출. 세로면 위=높은 값, 가로면 오른쪽=높은 값."""
    x0, y0, x1, y1 = box
    # 이미지 범위를 벗어난 좌표는 잘라내고, 드래그 방향과 무관하게 좌상단 -> 우하단으로 정렬
    x0, x1 = sorted((max(0, x0), min(arr.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(arr.shape[0], y1)))
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        raise ValueError("영역이 너무 작습니다.")
    vertical = h >= w                      # 세로로 긴 막대면 세로 범례
    length = h if vertical else w          # 막대의 길이 방향
    cross0, cross1 = ((x0, x1) if vertical else (y0, y1))
    c_mid = (cross0 + cross1) // 2         # 막대 폭의 중앙선 (글자/테두리를 피하려고 중앙만 사용)
    c_half = max(1, (cross1 - cross0) // 6)
    out = []
    for k in range(n):
        # k 번째 색 칸의 중앙 위치 주변 작은 패치의 중앙값을 그 칸의 색으로 사용
        pos = int((k + 0.5) * length / n)
        half = max(1, length // n // 4)
        a, b = max(0, pos - half), pos + half + 1
        if vertical:
            patch = arr[y0 + a:y0 + b, c_mid - c_half:c_mid + c_half + 1]
        else:
            patch = arr[c_mid - c_half:c_mid + c_half + 1, x0 + a:x0 + b]
        out.append(rgb_to_hex(np.median(patch.reshape(-1, 3), axis=0)))
    # 세로 막대는 위쪽이 최대값이므로 뒤집어서 '낮은 값 -> 높은 값' 순서로 맞춘다
    return out[::-1] if vertical else out
