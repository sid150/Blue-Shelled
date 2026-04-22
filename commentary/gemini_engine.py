from __future__ import annotations

from collections import deque
import base64
from dataclasses import dataclass
from typing import Any

import cv2

from analyzer.game_state import GameState

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None


SYSTEM_PROMPT = """
You are Blue-Shelled, a Mario Kart commentator that roasts the player in real time.
Style requirements:
- Funny and game-aware, never hateful or abusive.
- Keep each roast to max 2 short sentences.
- Mention race context (position/events/item chaos) whenever present.
- Avoid repeating the exact same joke templates.
"""


@dataclass(slots=True)
class RoastResult:
    text: str
    triggered: bool


class GeminiCommentaryEngine:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        roast_intensity: int,
        frame_jpeg_quality: int = 80,
        roast_history_size: int = 6,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.roast_intensity = roast_intensity
        self.frame_jpeg_quality = max(30, min(100, frame_jpeg_quality))
        self._history: deque[str] = deque(maxlen=max(1, roast_history_size))
        self._client = genai.Client() if genai else None

    def should_trigger(self, state: GameState) -> bool:
        if state.fell_off_track or state.hit_by_blue_shell:
            return True
        if state.recent_events:
            latest = state.recent_events[-1]
            return latest.startswith(
                (
                    "position_drop_",
                    "position_gain_",
                    "lap_advance_",
                    "item_changed_",
                    "fell_off_track",
                    "blue_shell_hit",
                )
            )
        return False

    def generate_roast(self, frame_bgr: Any, state: GameState) -> RoastResult:
        if not self.should_trigger(state):
            return RoastResult(text="", triggered=False)

        if self._client is None:
            return RoastResult(
                text=self._fallback_roast(state),
                triggered=True,
            )

        prompt = self._build_prompt(state)
        payload = self._build_image_payload(frame_bgr)
        parts = [types.Part(text=prompt)]
        if payload:
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=base64.b64decode(payload),
                    )
                )
            )

        response = self._client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT.strip(),
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            contents=[
                types.Content(
                    role="user",
                    parts=parts,
                )
            ],
        )
        text = (response.text or "").strip()
        if not text:
            text = self._fallback_roast(state)
        self._history.append(text)
        return RoastResult(text=text, triggered=True)

    def _build_prompt(self, state: GameState) -> str:
        context = state.as_prompt_dict()
        return (
            f"Roast intensity (1-5): {self.roast_intensity}\n"
            f"Race context: {context}\n"
            f"Recent roast history to avoid repeating: {list(self._history)}\n"
            "Give one short roast now."
        )

    def _build_image_payload(self, frame_bgr: Any) -> str:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_jpeg_quality],
        )
        if not ok:
            return ""
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _fallback_roast(self, state: GameState) -> str:
        if state.recent_events:
            latest = state.recent_events[-1]
            if latest.startswith("position_drop_"):
                return "That position drop was so fast even the minimap looked worried."
            if latest.startswith("position_gain_"):
                return "Nice climb. Keep driving like someone finally took your brakes off."
            if latest.startswith("lap_advance_"):
                return "New lap, same chaos. Let's see if this one is less dramatic."
        if state.hit_by_blue_shell:
            return "Blue shell tax collected. First place was a rental anyway."
        if state.fell_off_track:
            return "Track boundaries are suggestions, not obligations, right?"
        return "You've got confidence. The lap time does not."
