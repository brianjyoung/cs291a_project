import cv2
import dlib
from imutils import face_utils
import numpy as np

FACE_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
MOUTH_CASCADE_FILE = 'haarcascade_smile.xml'


def largest_box(arr: np.ndarray) -> np.ndarray:
    index = np.argmax(arr, axis=0)[2]
    return arr[index]


class FeatureExtractor(object):
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)
    mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE_FILE)
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def __init__(self, video_file: str):
        self.video = cv2.VideoCapture(video_file)
        self.frames = []
        self.frames_gray = []
        self.faces = []
        self.mouths = []
        self.features = []

    def read_video(self):
        self.frames = []
        while self.video.isOpened():
            _, frame = self.video.read()
            if frame is None:
                break
            self.frames.append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames_gray.append(gray)

    def face_detect(self, draw: bool=False):
        self.faces = []
        for frame_gray in self.frames_gray:
            faces = FeatureExtractor.face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            if len(faces) > 0:
                self.faces.append(largest_box(faces))
                if draw:
                    (x, y, w, h) = largest_box(faces)
                    cv2.rectangle(frame_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                self.faces.append(None)

    def mouth_detect(self, draw: bool=False):
        self.mouths = []
        for i in range(len(self.frames)):
            if self.faces[i] is not None:
                (x, y, w, h) = self.faces[i]
                roi = self.frames_gray[i][y:y+h, x:x+w]
                mouths = FeatureExtractor.mouth_cascade.detectMultiScale(roi, 1.3, 5)
                if len(mouths) > 0:
                    self.mouths.append(largest_box(mouths))
                    if draw:
                        (x, y, w, h) = largest_box(mouths)
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    self.mouths.append(None)
            else:
                self.mouths.append(None)

    def landmark_detect(self, draw: bool=False):
        self.features = []
        for i in range(len(self.frames)):
            face = self.faces[i]
            if face is None:
                continue
            rect = dlib.rectangle(face[0], face[1], face[0] + face[2], face[1] + face[3])
            shape = self.landmark_predictor(self.frames_gray[i], rect)
            shape = face_utils.shape_to_np(shape)
            if draw:
                for (x, y) in shape:
                    cv2.circle(self.frames_gray[i], (x, y), 1, (0, 0, 255), -1)
            self.features.append(np.array((shape[51], shape[57], shape[48], shape[54])))

    def show_video(self):
        for frame in self.frames_gray:
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
