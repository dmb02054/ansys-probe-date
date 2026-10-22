"""설정 저장/불러오기 (config/settings.json).

- 소스 실행: 프로젝트 폴더의 config/settings.json
- exe 실행(PyInstaller): exe 와 같은 폴더의 config/settings.json
  (onefile exe 는 실행할 때마다 임시 폴더에 풀리므로 그곳에 저장하면 설정이 사라진다)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .legend import DEFAULT_COLORS

if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys.executable).resolve().parent
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = _BASE_DIR / "config" / "settings.json"

# 설정 파일이 없거나 일부 항목이 빠졌을 때 사용하는 기본값
DEFAULTS = {
    "vmin": "350",            # 범례 최소값
    "vmax": "375",            # 범례 최대값
    "unit": "°C",             # 단위
    "colors": DEFAULT_COLORS, # 범례 색상 (낮은 값 -> 높은 값)
    "tolerance": 40,          # 색상 허용오차
    "sample": 5,              # 클릭 지점 샘플 크기(px)
    "open_ends": True,        # 양 끝 구간을 이상/이하로 표시
    "delay": 0,               # 화면 캡처 지연(초)
    "hover": True,            # 마우스 이동 시 실시간 값 표시
    "interp": True,           # 선형 보간으로 소수 값 추정
    "search": 60,             # 경계 탐색 반경(px)
    "ignore_black": True,     # 검은 선 무시
    "black_thr": 70,          # 검은색으로 간주할 최대 밝기
    "shade": 80,              # 그림자/조명 보정 정도(0~100)
    "edges": {},              # 사용자가 입력한 중간 경계값 {경계 번호: 값}
}


def load() -> dict:
    """저장된 설정을 읽어 기본값 위에 덮어쓴다. 파일이 없거나 손상되면 기본값을 사용."""
    data = dict(DEFAULTS)
    try:
        data.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return data


def save(data: dict) -> None:
    """설정을 JSON 으로 저장한다 (폴더가 없으면 생성)."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
