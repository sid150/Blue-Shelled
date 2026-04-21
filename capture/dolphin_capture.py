from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import platform
import subprocess
import threading
import time
from typing import Any

import cv2
import mss
import numpy as np


@dataclass(slots=True)
class CapturedFrame:
    image: np.ndarray
    captured_at: float


class FrameBuffer:
    def __init__(self, max_frames: int = 10) -> None:
        self._buffer: deque[CapturedFrame] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def add(self, frame: CapturedFrame) -> None:
        with self._lock:
            self._buffer.append(frame)

    def latest(self) -> CapturedFrame | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None


class DolphinCapture:
    def __init__(
        self,
        frame_buffer: FrameBuffer,
        fps: int = 2,
        window_title: str = "Dolphin",
        frame_scale: float = 1.0,
    ) -> None:
        self.frame_buffer = frame_buffer
        self.fps = fps
        self.window_title = window_title
        self.frame_scale = frame_scale
        self.running = False
        self._capture_thread: threading.Thread | None = None
        self._bbox: dict[str, int] | None = None

    def start(self) -> None:
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self) -> None:
        self.running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.5)

    def _capture_loop(self) -> None:
        interval = max(0.05, 1 / max(1, self.fps))
        with mss.mss() as sct:
            while self.running:
                bbox = self._resolve_bbox(sct)
                img = sct.grab(bbox)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
                if self.frame_scale != 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
                self.frame_buffer.add(CapturedFrame(image=frame, captured_at=time.time()))
                time.sleep(interval)

    def _resolve_bbox(self, sct: mss.base.MSSBase) -> dict[str, Any]:
        system = platform.system().lower()
        if system == "darwin":
            self._bbox = self._bbox or self._macos_bbox()
            if self._bbox:
                return self._bbox
        return sct.monitors[1]

    def _macos_bbox(self) -> dict[str, int] | None:
        script = f"""
        tell application "System Events"
            tell process "{self.window_title}"
                set w to window 1
                set p to position of w
                set s to size of w
                return ((item 1 of p) as string) & "," & ((item 2 of p) as string) & "," & ((item 1 of s) as string) & "," & ((item 2 of s) as string)
            end tell
        end tell
        """
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        left, top, width, height = map(int, result.stdout.strip().split(","))
        return {"top": top, "left": left, "width": width, "height": height}