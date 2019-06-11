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

# READ FILE HERE
label_dict_train = {}
label_dict_test  =  {}

train_label_file = open('train_complete_file.txt','r')
for line in train_label_file.readlines():
    line_input = line.split(" ")
    label_dict_train[line_input[0]] = line_input[2].rstrip()
train_label_file.close()

test_label_file = open('test_complete_file.txt','r')
for line in test_label_file.readlines():
    line_input = line.split(" ")
    label_dict_test[line_input[0]] = line_input[2].rstrip()
test_label_file.close()

# DIRECTORY OF TRAINING DATA
files = os.listdir('train_data')
files = files[:10]
labels = []
data = np.empty((len(files), 60))
for i, file in enumerate(files):
    # Get label for file
    # Extract lip as 60x80 image
    file = os.path.join('train_data', file)
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
    labels.append(label_dict_train[file])
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

# Train logistic regression, need to get labels in some way; not finished
logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression.fit(training, labels)
logistic_regression.predict(training)


# TESTING
# logistic_regression.predict(testing)

