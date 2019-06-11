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
label_dict_test = {}

train_label_file = open('train_complete_file.txt', 'r')
for line in train_label_file.readlines():
    line_input = line.split(" ")
    label_dict_train[line_input[0]] = line_input[2].rstrip()
train_label_file.close()

test_label_file = open('test_complete_file.txt', 'r')
for line in test_label_file.readlines():
    line_input = line.split(" ")
    label_dict_test[line_input[0]] = line_input[2].rstrip()
test_label_file.close()

# Read training files
files = os.listdir('train_data')
files = files[:100]
training_labels = []
training_features = np.empty((len(files), 60))
for i, file in enumerate(files):
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
    training_labels.append(label_dict_train[file])
    training_features[i] = np.sum(differences, 1)
    # print("File {} complete.".format(i))

training_features = training_features[:len(training_labels), ]
print("------{} seconds elapsed".format(time.time()-start_time))

# Dump features from training data
pickle.dump(training_features, open('training_features.pk', 'wb'))

# Perform PCA on (normalized) training data
scaler = StandardScaler()
scaler.fit(training_features)
training_normalized = scaler.transform(training_features)
pca = PCA(0.95)
pca.fit(training_normalized)

# Now can apply PCA mapping like so:
training_transformed = pca.transform(training_normalized)

pickle.dump(training_features, open('training_transformed_after_PCA.pk', 'wb'))
# Train logistic regression
logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression.fit(training_transformed, training_labels)
training_predictions = logistic_regression.predict(training_transformed)


# Read test files
files = os.listdir('test_data')
files = files[:100]
test_labels = []
test_features = np.empty((len(files), 60))
for i, file in enumerate(files):
    # Get label for file
    test_labels.append(label_dict_test[file])
    # Extract lip as 60x80 image
    file = os.path.join('test_data', file)
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
    test_features[i] = np.sum(differences, 1)
    # print("File {} complete.".format(i))

# Apply PCA mapping on (normalized) test data
scaler.fit(test_features)
test_normalized = scaler.transform(test_features)
test_transformed = pca.transform(training_normalized)

# Test logistic regression
testing_predictions = logistic_regression.predict(test_transformed)
