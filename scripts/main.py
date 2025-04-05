from cvtools.video_stream import WebcamStream, ScreenStream
from cvtools.annotation import AnnotatedVideoStream
import cv2

monitor = {"left": 0, "top": 0, "height": 1000, "width": 1000}

stream = ScreenStream(monitor)
annotated_stream = AnnotatedVideoStream(stream, model_path="trained_s.pt")

while True:
    frame = annotated_stream.get_frame()

    cv2.imshow("OpenCV", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

