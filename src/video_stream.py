from abc import ABC, abstractmethod
import cv2
from mss import mss
import numpy as np

class VideoStream(ABC):
    @abstractmethod
    def get_cv_frame(self):
        """Returns an OpenCV frame"""
        pass

    @abstractmethod
    def close(self):
        pass

    def __del__(self):
        self.close()

    
class WebcamStream(VideoStream):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise IOError("Cannot open webcam")

    def get_cv_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return None
        else:
            return frame

    def close(self):
        self.capture.release()

class ScreenStream(VideoStream):
    def __init__(self, bounding_box={"top": 40, "left": 0, "width": 800, "height": 640}):
        self.bounds = bounding_box
        self.sct = mss()
        
    def get_cv_frame(self):
        img = self.sct.grab(self.bounds)
        return np.array(img)

    def close():
        pass
