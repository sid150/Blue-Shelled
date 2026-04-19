from __future__ import annotations

import argparse
import time

from analyzer.game_state import GameStateAnalyzer
from blue_shelled.config import load_config
from capture.dolphin_capture import DolphinCapture, FrameBuffer
from commentary.gemini_engine import GeminiCommentaryEngine
from voice.output import VoiceOutput


class BlueShelledApp:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = load_config(config_path)
        self.frame_buffer = FrameBuffer(max_frames=20)
        self.capture = DolphinCapture(
            frame_buffer=self.frame_buffer,
            fps=self.config.capture_fps,
            window_title=self.config.dolphin_window_title,
            frame_scale=self.config.frame_scale,
        )
        self.analyzer = GameStateAnalyzer(
            history_size=self.config.event_history_size,
            ocr_enabled=self.config.ocr_enabled,
            ocr_sample_stride=self.config.ocr_sample_stride,
            hud_roi_place=self.config.hud_roi_place,
            hud_roi_lap=self.config.hud_roi_lap,
            hud_roi_item=self.config.hud_roi_item,
        )
        self.commentary = GeminiCommentaryEngine(
            model=self.config.gemini_model,
            temperature=self.config.gemini_temperature,
            max_tokens=self.config.max_roast_tokens,
            roast_intensity=self.config.roast_intensity,
            frame_jpeg_quality=self.config.frame_jpeg_quality,
            roast_history_size=self.config.roast_history_size,
        )
        self.voice = VoiceOutput(speaking_mode=self.config.speaking_mode)
        self.last_roast_at = 0.0

    def run(self) -> None:
        self.capture.start()
        self.voice.start()
        frame_counter = 0

        try:
            while True:
                latest = self.frame_buffer.latest()
                if latest is None:
                    time.sleep(0.02)
                    continue

                frame_counter += 1
                if frame_counter % max(1, self.config.analyze_every_n_frames) != 0:
                    time.sleep(0.01)
                    continue

                state = self.analyzer.analyze(latest.image)
                if self.config.mute_on_menu and not state.in_race:
                    continue
                if time.time() - self.last_roast_at < self.config.roast_cooldown_sec:
                    continue

                roast = self.commentary.generate_roast(latest.image, state)
                if roast.triggered and roast.text:
                    self.voice.enqueue(roast.text)
                    self.last_roast_at = time.time()
                    print(f"[Blue-Shelled] {roast.text}")
                    print(
                        "[Blue-Shelled State]",
                        {
                            "place": state.place,
                            "lap": state.lap,
                            "item": state.item_held,
                            "events": state.recent_events[-3:],
                        },
                    )
        except KeyboardInterrupt:
            pass
        finally:
            self.capture.stop()
            self.voice.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Blue-Shelled runtime")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    app = BlueShelledApp(config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
