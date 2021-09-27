import unittest
import numpy as np
import mcp
import csv
from statistics import mean



reader = csv.reader(open('train.csv', newline=''))
digit_train_data = list(reader)

temp = [r.pop(0) for r in digit_train_data]
numpyArray = np.array(temp)
transpose = numpyArray.T
digit_train_labels = transpose.tolist()

del digit_train_data[0]
digit_train_data = [list( map(int,i) ) for i in digit_train_data]
del digit_train_labels[0]
digit_train_labels = [int(i) for i in digit_train_labels]

digit_train_data = np.array(digit_train_data)
digit_train_labels = np.array(digit_train_labels)

#This is for the test data
reader = csv.reader(open('test.csv', newline=''))
digit_test_data = list(reader)

temp = [r.pop(0) for r in digit_test_data]
numpyArray = np.array(temp)
transpose = numpyArray.T
digit_test_labels = transpose.tolist()

del digit_test_data[0]
digit_test_data = [list( map(int,i) ) for i in digit_test_data]
del digit_test_labels[0]
digit_test_labels = [int(i) for i in digit_test_labels]

digit_test_data = np.array(digit_test_data)
digit_test_labels = np.array(digit_test_labels)

#This is for the handwriting data
reader = csv.reader(open('handwritingPixels.csv', newline=''))
handwritten_data = list(reader)

temp = [r.pop(0) for r in handwritten_data]
numpyArray = np.array(temp)
transpose = numpyArray.T
handwritten_labels = transpose.tolist()

del handwritten_data[0]
handwritten_data = [list( map(int,i) ) for i in handwritten_data]
del handwritten_labels[0]
handwritten_labels = [int(i) for i in handwritten_labels]

handwritten_data = np.array(handwritten_data)
handwritten_labels = np.array(handwritten_labels)

#Handwriting Pt 2
reader = csv.reader(open('casey_handwriting.csv', newline=''))
handwritten_data2 = list(reader)

temp = [r.pop(0) for r in handwritten_data2]
numpyArray = np.array(temp)
transpose = numpyArray.T
handwritten_labels2 = transpose.tolist()

del handwritten_data2[0]
handwritten_data2 = np.array(handwritten_data2).astype(float).astype(int)
handwritten_data2[handwritten_data2 == 1] = 255
handwritten_data2 = handwritten_data2.tolist()
handwritten_data2 = [list( map(int,i) ) for i in handwritten_data2]
del handwritten_labels2[0]
handwritten_labels2 = np.array(handwritten_labels2).astype(float).astype(int)
handwritten_labels2 = handwritten_labels2.tolist()
handwritten_labels2 = [int(i) for i in handwritten_labels2]

handwritten_data2 = np.array(handwritten_data2)
handwritten_labels2 = np.array(handwritten_labels2)


inputs = digit_train_data.shape[1]
n = digit_train_data.shape[0]

digit_train_targets = np.zeros(shape=(n,10))
i = 0
while i < n:
    digit_train_targets[i][digit_train_labels[i]] = 1
    i+=1


multiclass = mcp.MCP(inputs, 10)
multiclass.train(digit_train_data, digit_train_targets)
testaccuracy = 0
n = digit_train_data.shape[0]
test_predictions = []
for testIndex in range(n):
    result = multiclass.predict(digit_train_data[testIndex])
    actual = digit_train_labels[testIndex]
    if (result == actual):
        testaccuracy+=1
    test_predictions.append(result)
test_predictions = np.array(test_predictions).astype(int)

finalaccuracy = (testaccuracy / n) * 100.0
print("Accuracy of Multi-Class Perceptron is approximately {}%".format(finalaccuracy))
np.savetxt("predictedTestsPerceptron.csv", test_predictions.astype(int), fmt='%i', delimiter=",")


multiclass.train(handwritten_data, handwritten_labels)
multiclass.train(handwritten_data2, handwritten_labels2)
n = handwritten_data.shape[0]
handAcc = 0
for testIndex in range(n):
    result = multiclass.predict(handwritten_data[testIndex])
    actual = handwritten_labels[testIndex]
    if (result == actual):
        handAcc+=1
a = handwritten_data2.shape[0]
for testIndex in range(a):
    result = multiclass.predict(handwritten_data2[testIndex])
    actual = handwritten_labels2[testIndex]
    if (result == actual):
        handAcc+=1

finalaccuracy = (handAcc / (n+a)) * 100.0
print("Accuracy of Multi-Class Perceptron on handwriting is approximately {}%".format(finalaccuracy))

# CROSS VALIDATION

digitlength = digit_train_data.shape[0]
kfolds = 40
folds = np.split(digit_train_data, kfolds)
flabels = np.split(digit_train_targets, kfolds)
this_correct = 0
this_runs = 0
all_accuracies = []
new_mcp = mcp.MCP(inputs, 10)
for j in range(kfolds):
    for l in range (kfolds):
        if l != j:
            data = folds[l]
            targets = flabels[l]
            new_mcp.train(data, targets)
    for m in range(len(folds[j])):
        predictData = folds[j][m]
        realtarget = np.argmax(flabels[j][m])
        res = new_mcp.predict(predictData)
        if realtarget == res:
            this_correct+=1
        this_runs+=1
    acc = (this_correct / this_runs) * 100.0
    all_accuracies.append(acc)

best_fold = all_accuracies.index(max(all_accuracies))
best_accuracy = all_accuracies[best_fold]
best_data = folds[best_fold]
best_labels = flabels[best_fold]
total_accuracy = mean(all_accuracies)

print("Average Accuracy of Multi-Class Perceptron from Cross-Val is: {}%".format(total_accuracy))
print("Best Accuracy of Multi-Class Perceptron from Cross-Val is: {}%".format(best_accuracy))