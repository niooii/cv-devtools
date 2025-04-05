from cvtools.video_stream import WebcamStream, ScreenStream
from cvtools.annotation import Annotator
import cv2

monitor = {"left": 0, "top": 0, "height": 1000, "width": 1000}
stream = ScreenStream(monitor)
annotator = Annotator("trained.pt")

while True:
    frame = stream.get_cv_frame()
    frame = annotator.annotate(frame)

    cv2.imshow('Image', frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

