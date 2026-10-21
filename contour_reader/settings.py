"""설정 저장/불러오기 (config/settings.json)."""
from __future__ import annotations

import json
from pathlib import Path

from .legend import DEFAULT_COLORS

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.json"

DEFAULTS = {
    "vmin": "350",
    "vmax": "375",
    "unit": "°C",
    "colors": DEFAULT_COLORS,
    "tolerance": 40,
    "sample": 5,
    "open_ends": True,
    "delay": 0,
    "hover": True,
    "pmode": "방사 평균 (전방향)",
    "radius": 60,
    "scale": "",
    "scale_unit": "mm",
    "save_chart": True,
}


def load() -> dict:
    data = dict(DEFAULTS)
    try:
        data.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return data


def save(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
