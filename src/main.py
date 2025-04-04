from video_stream import WebcamStream, ScreenStream
import cv2

stream = ScreenStream()
frame = stream.get_cv_frame()

cv2.imshow('Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
