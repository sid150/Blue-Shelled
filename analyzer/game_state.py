from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import re

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

from analyzer.mkw_memory import MKWDolphinMemoryReader


@dataclass(slots=True)
class GameState:
    place: int | None = None
    lap: int | None = None
    item_held: str | None = None
    item_used: str | None = None
    in_race: bool = True
    fell_off_track: bool = False
    hit_by_blue_shell: bool = False
    race_finished: bool = False
    recent_events: list[str] = field(default_factory=list)

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "place": self.place,
            "lap": self.lap,
            "item_held": self.item_held,
            "item_used": self.item_used,
            "in_race": self.in_race,
            "fell_off_track": self.fell_off_track,
            "hit_by_blue_shell": self.hit_by_blue_shell,
            "race_finished": self.race_finished,
            "recent_events": self.recent_events,
        }


class GameStateAnalyzer:
    """
    Lightweight vision heuristics + temporal change detector.
    It keeps the pipeline responsive even without Dolphin memory reads.
    """

    def __init__(
        self,
        history_size: int = 8,
        ocr_enabled: bool = True,
        ocr_sample_stride: int = 2,
        hud_roi_place: tuple[float, float, float, float] = (0.70, 0.88, 0.02, 0.22),
        hud_roi_lap: tuple[float, float, float, float] = (0.77, 0.94, 0.77, 0.98),
        hud_roi_item: tuple[float, float, float, float] = (0.03, 0.20, 0.73, 0.96),
        memory_reader: MKWDolphinMemoryReader | None = None,
    ) -> None:
        self.history_size = max(2, history_size)
        self.ocr_enabled = ocr_enabled
        self.ocr_sample_stride = max(1, ocr_sample_stride)
        self._roi_place = hud_roi_place
        self._roi_lap = hud_roi_lap
        self._roi_item = hud_roi_item
        self._prev_frame_gray: np.ndarray | None = None
        self._prev_state: GameState | None = None
        self._event_history: list[str] = []
        self._frame_index = 0
        self._mem_reader = memory_reader

    def analyze(self, frame_bgr: np.ndarray) -> GameState:
        self._frame_index += 1
        state = GameState()
        state.in_race = self._detect_in_race(frame_bgr)
        state.place = self._extract_place(frame_bgr)
        state.lap = self._extract_lap(frame_bgr)
        state.item_held = self._extract_item(frame_bgr)
        state.fell_off_track = self._detect_fall(frame_bgr)
        state.hit_by_blue_shell = self._detect_blue_shell(frame_bgr)
        state.race_finished = bool(state.lap == 3 and state.place is not None and state.place <= 3)

        self._overlay_dolphin_memory(state)

        if state.fell_off_track:
            self._push_event("fell_off_track")
        if state.hit_by_blue_shell:
            self._push_event("blue_shell_hit")
        if not state.in_race:
            self._push_event("menu_screen")
        self._push_delta_events(state)

        state.recent_events = self._event_history.copy()
        self._prev_state = state
        return state

    def _overlay_dolphin_memory(self, state: GameState) -> None:
        if self._mem_reader is None:
            return
        overlay = self._mem_reader.read_overlay()
        if overlay is None:
            return
        if overlay.place is not None:
            state.place = overlay.place
        if overlay.lap is not None:
            state.lap = overlay.lap
        if overlay.in_race is not None:
            state.in_race = overlay.in_race
        if overlay.race_finished is True:
            state.race_finished = True

    def _detect_in_race(self, frame: np.ndarray) -> bool:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[..., 1].mean())
        return saturation > 35

    def _detect_fall(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0
        if self._prev_frame_gray is not None:
            diff = cv2.absdiff(gray, self._prev_frame_gray)
            motion_score = float(diff.mean())
        self._prev_frame_gray = gray
        dark_ratio = float((gray < 25).mean())
        return dark_ratio > 0.6 and motion_score > 25

    def _detect_blue_shell(self, frame: np.ndarray) -> bool:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, (95, 110, 80), (130, 255, 255))
        return float((blue_mask > 0).mean()) > 0.18

    def _slice_normalized(self, frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        y0, y1, x0, x1 = roi
        r0 = int(y0 * h)
        r1 = max(r0 + 1, int(y1 * h))
        c0 = int(x0 * w)
        c1 = max(c0 + 1, int(x1 * w))
        return frame[r0:r1, c0:c1]

    def _extract_place(self, frame: np.ndarray) -> int | None:
        if not self._can_ocr():
            return self._prev_state.place if self._prev_state else None
        roi = self._slice_normalized(frame, self._roi_place)
        text = self._ocr_digits(roi)
        place = self._extract_int(text)
        if place is None:
            return self._prev_state.place if self._prev_state else None
        if 1 <= place <= 12:
            return place
        return self._prev_state.place if self._prev_state else None

    def _extract_lap(self, frame: np.ndarray) -> int | None:
        if not self._can_ocr():
            return self._prev_state.lap if self._prev_state else None
        roi = self._slice_normalized(frame, self._roi_lap)
        text = self._ocr_digits(roi)
        lap = self._extract_int(text)
        if lap is None:
            return self._prev_state.lap if self._prev_state else None
        if 1 <= lap <= 3:
            return lap
        return self._prev_state.lap if self._prev_state else None

    def _extract_item(self, frame: np.ndarray) -> str | None:
        roi = self._slice_normalized(frame, self._roi_item)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_ratio = float(cv2.inRange(hsv, (95, 80, 80), (130, 255, 255)).mean()) / 255.0
        red_ratio = float(cv2.inRange(hsv, (0, 90, 70), (10, 255, 255)).mean()) / 255.0
        yellow_ratio = float(cv2.inRange(hsv, (18, 90, 90), (40, 255, 255)).mean()) / 255.0
        green_ratio = float(cv2.inRange(hsv, (40, 70, 70), (85, 255, 255)).mean()) / 255.0

        if blue_ratio > 0.12:
            return "blue_shell_or_spiny_item"
        if red_ratio > 0.14:
            return "red_shell_or_fire_item"
        if yellow_ratio > 0.16:
            return "banana_or_star"
        if green_ratio > 0.16:
            return "green_shell_or_plant_item"
        return "unknown_item" if roi.mean() > 40 else None

    def _can_ocr(self) -> bool:
        return (
            self.ocr_enabled
            and pytesseract is not None
            and self._frame_index % self.ocr_sample_stride == 0
        )

    def _ocr_digits(self, roi: np.ndarray) -> str:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(thresh)
        upscaled = cv2.resize(inv, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return pytesseract.image_to_string(
            upscaled,
            config="--psm 7 -c tessedit_char_whitelist=0123456789/",
        )

    def _extract_int(self, text: str) -> int | None:
        matches = re.findall(r"\d+", text or "")
        if not matches:
            return None
        try:
            return int(matches[0])
        except ValueError:
            return None

    def _push_delta_events(self, state: GameState) -> None:
        prev = self._prev_state
        if prev is None:
            return
        if prev.place is not None and state.place is not None:
            if state.place > prev.place:
                self._push_event(f"position_drop_{prev.place}_to_{state.place}")
            elif state.place < prev.place:
                self._push_event(f"position_gain_{prev.place}_to_{state.place}")
        if prev.lap is not None and state.lap is not None and state.lap > prev.lap:
            self._push_event(f"lap_advance_{prev.lap}_to_{state.lap}")
        if prev.item_held != state.item_held and state.item_held:
            self._push_event(f"item_changed_{state.item_held}")

    def _push_event(self, event: str) -> None:
        self._event_history.append(event)
        if len(self._event_history) > self.history_size:
            self._event_history = self._event_history[-self.history_size :]
