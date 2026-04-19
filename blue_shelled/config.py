from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

# Normalized HUD regions: (y0, y1, x0, x1) in 0..1 relative to frame height/width.
_DEFAULT_PLACE_ROI = (0.70, 0.88, 0.02, 0.22)
_DEFAULT_LAP_ROI = (0.77, 0.94, 0.77, 0.98)
_DEFAULT_ITEM_ROI = (0.03, 0.20, 0.73, 0.96)


def _parse_normalized_roi(value: Any, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 4:
        y0, y1, x0, x1 = (float(value[i]) for i in range(4))
    elif isinstance(value, dict):
        y0 = float(value.get("y0", default[0]))
        y1 = float(value.get("y1", default[1]))
        x0 = float(value.get("x0", default[2]))
        x1 = float(value.get("x1", default[3]))
    else:
        return default

    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    y0, y1, x0, x1 = _clamp01(y0), _clamp01(y1), _clamp01(x0), _clamp01(x1)
    if y0 > y1:
        y0, y1 = y1, y0
    if x0 > x1:
        x0, x1 = x1, x0
    if y1 - y0 < 0.01 or x1 - x0 < 0.01:
        return default
    return (y0, y1, x0, x1)


@dataclass(slots=True)
class AppConfig:
    capture_fps: int = 2
    roast_cooldown_sec: float = 8.0
    voice_id: str = "default"
    roast_intensity: int = 3
    mute_on_menu: bool = True
    dolphin_window_title: str = "Dolphin"
    frame_scale: float = 1.0
    frame_jpeg_quality: int = 80
    gemini_model: str = "gemini-2.0-flash"
    gemini_tts_model: str = "gemini-2.5-flash-preview-tts"
    max_roast_tokens: int = 100
    gemini_temperature: float = 0.9
    event_history_size: int = 8
    roast_history_size: int = 6
    speaking_mode: str = "interrupt"
    analyze_every_n_frames: int = 2
    ocr_enabled: bool = True
    ocr_sample_stride: int = 2
    hud_roi_place: tuple[float, float, float, float] = _DEFAULT_PLACE_ROI
    hud_roi_lap: tuple[float, float, float, float] = _DEFAULT_LAP_ROI
    hud_roi_item: tuple[float, float, float, float] = _DEFAULT_ITEM_ROI


def _clamp_intensity(value: int) -> int:
    return max(1, min(5, value))


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")

    merged: dict[str, Any] = asdict(AppConfig())
    merged.update(raw)
    merged["roast_intensity"] = _clamp_intensity(int(merged["roast_intensity"]))

    merged["hud_roi_place"] = _parse_normalized_roi(
        raw.get("hud_roi_place"), _DEFAULT_PLACE_ROI
    )
    merged["hud_roi_lap"] = _parse_normalized_roi(raw.get("hud_roi_lap"), _DEFAULT_LAP_ROI)
    merged["hud_roi_item"] = _parse_normalized_roi(
        raw.get("hud_roi_item"), _DEFAULT_ITEM_ROI
    )

    return AppConfig(**merged)
