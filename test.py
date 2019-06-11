from FeatureExtractor import FeatureExtractor
import MSA

import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time


start_time = time.time()

# DIRECTORY OF TRAINING DATA
files = os.listdir('training_dump')
data = np.empty((len(files), 60))
for i, file in enumerate(files[:80]):
    # Extract lip as 60x80 image
    file = os.path.join('training_dump', file)
    fe = FeatureExtractor.from_image(file)
    fe.face_detect()
    fe.landmark_detect()
    fe.crop_lips()
    # print("File {} lips extracted.".format(i))

    # Convert lip image to 60-dimensional feature vector
    if fe.lips is None:
        print("File {} skipped, lips could not be found.".format(i))
        continue
    filters = MSA.multiscale_full(fe.lips[0].flatten('F'))
    differences = filters[1:] - filters[:-1]
    data[i] = np.sum(differences, 1)
    # print("File {} complete.".format(i))

pickle.dump(data, open('data.pk', 'wb'))
# Perform PCA on (normalized) training data
scaler = StandardScaler()
scaler.fit(data)
normalized = scaler.transform(data)
pca = PCA(0.95)
pca.fit(normalized)

# Now can apply PCA mapping like so:
training = pca.transform(normalized)

print("------{} seconds elapsed".format(time.time()-start_time))

# Train/test logistic regression, need to get labels in some way; not finished
# logistic_regression = LogisticRegression(solver='lbfgs')
# logistic_regression.fit(training, LABELS)
# logistic_regression.predict(testing)
