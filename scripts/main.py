from cvtools.video_stream import WebcamStream, ScreenStream
from cvtools.annotation import AnnotatedVideoStream
import cv2

monitor = {"left": 0, "top": 0, "height": 1000, "width": 1000}

stream = ScreenStream(monitor)
annotated_stream = AnnotatedVideoStream(stream, model_path="trained_s.pt")

annotated_stream.show_in_window()

