from FeatureExtractor import FeatureExtractor
import MSA

import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# DIRECTORY OF TRAINING DATA
files = os.listdir('training_dump')
data = np.empty((len(files), 60))
for i, file in enumerate(files):
    # Extract lip as 60x80 image
    file = os.path.join('training_dump', file)
    fe = FeatureExtractor.from_image(file)
    fe.face_detect()
    fe.landmark_detect()
    fe.crop_lips()
    print("File {} lips extracted.".format(i))

    # Convert lip image to 60-dimensional feature vector
    filters = np.empty((61, 4800))
    if fe.lips is None:
        continue
    filters[0] = fe.lips[0].flatten('F')
    for j in range(1, 61):
        filters[j] = MSA.multiscale_step(j+1, filters[j-1])
    differences = filters[1:] - filters[:-1]
    data[i] = np.sum(differences, 1)
    print("File {} complete.".format(i))

# Perform PCA on (normalized) training data
scaler = StandardScaler()
scaler.fit(data)
normalized = scaler.transform(data)
pca = PCA(0.95)
pca.fit(normalized)

# Now can apply PCA mapping like so:
training = pca.transform(normalized)

# Train/test logistic regression, need to get labels in some way; not finished
# logistic_regression = LogisticRegression(solver='lbfgs')
# logistic_regression.fit(training, LABELS)
# logistic_regression.predict(testing)