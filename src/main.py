from video_stream import WebcamStream, ScreenStream
import cv2

stream = ScreenStream()

while True:
    frame = stream.get_cv_frame()

    cv2.imshow('Image', frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

