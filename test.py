from FeatureExtractor import FeatureExtractor
from VisemeClassifier import VisemeClassifier

fe = FeatureExtractor('lipreadtest.mp4')
fe.read_video()
fe.face_detect(draw=True)
fe.landmark_detect(draw=True)

vc = VisemeClassifier(fe.features)
