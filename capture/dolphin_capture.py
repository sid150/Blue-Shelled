from collections import deque
import threading
import time
import cv2
import mss
import numpy as np  
import pywinctl as wctl #for linux/mac
import subprocess
#TODO: add support for windows with pygetwindow

class FrameBuffer:

    def __init__(self,maxFrames = 10):
        #double ended queue stores 10 most recent frames for event
        self.buffer = deque(maxlen = maxFrames)
        self.lock = threading.Lock() 

    def add_to_buffer(self,frame):
        with self.lock:
            self.buffer.append(frame)

    def get_latest_from_buffer(self):
        with self.lock:
            return self.buffer[-1] if self.buffer else None

class DolphinCapture:

    def __init__(self,frame_buffer: FrameBuffer, monitor: int = 1, fps: int = 2):
        self.monitor = monitor
        self.frame_buffer = frame_buffer
        self.fps = fps
        self.running = False
        self.bbox = None
    
    def set_bbox(self):
        script = """
        tell application "System Events"
            tell process "Dolphin"
                set w to window 1
                set p to position of w
                set s to size of w
                return ((item 1 of p) as string) & "," & ((item 2 of p) as string) & "," & ((item 1 of s) as string) & "," & ((item 2 of s) as string)
            end tell
        end tell
        """
        print("Waiting for Dolphin window...")
        while True:
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                break
            time.sleep(1)  # retry every second until window is ready

        left, top, width, height = map(int, result.stdout.strip().split(","))
        self.bbox = {"top": top, "left": left, "width": width, "height": height}
        print(f"Dolphin bbox: {self.bbox}")
    
    def start(self):
        self.running = True
        self.set_bbox()  # add this line
        threading.Thread(target=self._capture_frames).start()
    
    def _capture_frames(self):
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor]
            while self.running:
                img = sct.grab(monitor)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
                self.frame_buffer.add_to_buffer(frame)
                time.sleep(1 / self.fps)
    
    def stop(self):
        self.running = False


