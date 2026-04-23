from __future__ import annotations

from collections import deque
from io import BytesIO
import tempfile
import threading
import time

from gtts import gTTS
import pygame


class VoiceOutput:
    def __init__(self, speaking_mode: str = "interrupt") -> None:
        self.speaking_mode = speaking_mode
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        pygame.mixer.init()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        pygame.mixer.stop()
        pygame.mixer.quit()

    def enqueue(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            if self.speaking_mode == "interrupt":
                self._queue.clear()
                pygame.mixer.stop()
            self._queue.append(text)

    def _worker_loop(self) -> None:
        while self._running:
            text = None
            with self._lock:
                if self._queue:
                    text = self._queue.popleft()
            if text is None:
                time.sleep(0.05)
                continue
            self._speak_gtts(text)

    def _speak_gtts(self, text: str) -> None:
        # gTTS writes MP3, so we render into a temp file for pygame playback.
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
            tts = gTTS(text=text, lang="en")
            audio = BytesIO()
            tts.write_to_fp(audio)
            tmp.write(audio.getvalue())
            tmp.flush()

            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self._running:
                time.sleep(0.05)
