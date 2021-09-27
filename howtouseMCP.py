import numpy as np
import mcp.py

# First create the model with the desired inputs and outputs (inputs, outputs)
mcpModel = mcp.MCP(784, 10)

# Then use the MCP's getData function with a csv file to get the data as a matrix and labels as an array
traindata, trainlabs = mcpModel.getData('train.csv')
# If used for training, reshape into an array of size (numinstances x numoutputs)
trainlabels = mcpModel.reshapeLabels(trainlabs)

# Use getData function to separate file into data and labels
testdata, testlabels = mcpModel.getData('test.csv')

# Use the train function with testing data and reshaped labels
mcpModel.train(traindata, trainlabels)

# Predict function does one instance at a time, so use a for loop to iterate through all the testing instances
# and calculate accuracy based off of that
testaccuracy = 0
n = testdata.shape[0]
for testIndex in range(n):
    # use predict with one instance of data at a time
    result = mcpModel.predict(testdata[testIndex])
    actual = testlabels[testIndex]

    # Compare the actual label and the predicted label, if equal increment the count by 1
    if (result == actual):
        testaccuracy+=1

# Get the total accuracy after testing
finalaccuracy = (testaccuracy / n) * 100.0

# Just to display accuracy
print("Accuracy of Multi-Class Perceptron is approximately {}%".format(finalaccuracy))