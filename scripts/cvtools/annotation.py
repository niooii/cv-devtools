import cv2
from ultralytics import YOLO
from cvtools.video_stream import VideoStream

class Annotator():
    def __init__(self, model_path: str): 
        self.model = YOLO(model_path, verbose=False)

    def annotate_image(self, img: cv2.Mat, min_confidence = 0.80):
        results = self.model.predict(img, verbose=False, conf=min_confidence)
        for result in results:
            if not result.boxes: 
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                json = result.summary()
                obj = json[0]
                class_name = obj["name"]
                confidence = obj["confidence"]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{class_name} {confidence:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return img
    
    def annotate_video(video_path: str, output_path: str):
        """TODO"""
        pass
    
class AnnotatedVideoStream():
    def __init__(self, video_stream: VideoStream, model_path: str):
        self.stream = video_stream
        self.annotator = Annotator(model_path=model_path)

    def get_frame(self): 
        frame = self.stream.get_cv_frame()
        return self.annotator.annotate_image(frame)

    def show_in_window(self):
        while True:
            frame = self.get_frame()

            cv2.imshow("OpenCV", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
