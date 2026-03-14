from collections import deque
import threading
import time
import cv2
import mss
import numpy as np  

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

    def start(self):
        self.running = True
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


