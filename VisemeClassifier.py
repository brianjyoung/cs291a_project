import numpy as np


class Viseme(object):
    def __init__(self, score: float):
        self.code = 0
        self.duration = 1
        pass

    def __eq__(self, other):
        return self.code == other.code


class VisemeClassifier(object):
    viseme_lut = {

    }

    def __init__(self, features: np.ndarray):
        self.visemes = []
        for feature in features:
            height = np.linalg.norm(feature[0] - feature[1])
            width = np.linalg.norm(feature[2] - feature[3])
            score = height / width
            viseme = Viseme(score)
            if not self.visemes or self.visemes[-1] != viseme:
                self.visemes.append(viseme)
            else:
                self.visemes[-1].duration += 1
