'''
    Compares perceptrons and multi-layer perceptrons (MLPs) using both artificial
    and real data sets.
'''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

## Perceptron

filePath_1 = "Data_Files\\sam_data\\000378875_1.csv"
filePath_2 = "Data_Files\\sam_data\\000378875_2.csv"
filePath_3 = "Data_Files\\sam_data\\000378875_3.csv"
filePath_4 = "Data_Files\\sam_data\\000378875_4.csv"

filePaths = [filePath_1, filePath_2, filePath_3, filePath_4]

learningRate = 0.1

finalAccuracy = []
finalWeights = []
finalThreshold = []
finalTestingData = []
finalTestingTargets = []

''' Custom perceptron Algorithm '''
def perceptronAlgorithm(data, target, learningRate):

    # Generates a list of weights with an initial value of zero
    weights = [0] * len(trainingData[0])
    weights = np.array(weights)

    threshold = 0

    epochs = 10

    runs = 0

    # Runs the perceptron algorithm n epochs
    while (runs < epochs):

        predictions = []
        count = 0

        # Calculates the output for each row
        for row in data:

            # Calculates the activation for each row
            activation = (row * weights).sum()

            # Predicts the output for each row and saves it to the predictions list
            if (activation <= threshold):
                output = 0
                predictions.append(output)
            else:
                output = 1
                predictions.append(output)

            # Calculates the weights and the threshold depending on the the output
            # and the target
            if (output < target[count]):
                weights = weights + row * learningRate
                threshold -= learningRate
            elif (output > target[count]):
                weights = np.subtract(weights, (row * learningRate))
                threshold += learningRate

            count += 1

        # Stops the loops when the predictions equals targets
        if (set(predictions) == set(target)):
            break

        runs += 1

    # Saves the weights and the threshold of the final epoch to finalWeights list
    # and finalThreshold
    finalWeights.append(np.round(weights, decimals=1))
    finalThreshold.append(threshold)

'''
    Calculates the accuracy using the weights and the threshold calculated by the
    custom perceptron algorithm
'''
def testingPerceptronAlgorithm(data, target, learningRate, weights, threshold):

    epochs = 1

    runs = 0
    accuracy = []

    # Runs the perceptron algorithm for 1 epoch
    while (runs < epochs):

        predictions = []

        # Calculates the output for each row
        for row in data:

            activation = (row * weights).sum()

            if (activation <= threshold):
                output = 0
                predictions.append(output)
            else:
                output = 1
                predictions.append(output)

        accuracy.append((accuracy_score(target, predictions)) * 100)

        runs += 1

    # Saves the accuracy in the finalAccuracy list
    finalAccuracy.append(max(accuracy))

for filePath in filePaths:

    # Reads a file of data
    rawDataset = np.genfromtxt(filePath, delimiter=",")

    # Separates the dataset into dataset list and targets list
    dataset = rawDataset[:,0:-1]
    targets = rawDataset[:,-1]

    # Splits the dataset list and the targets list into training data, training targets,
    # testing data and testing targets
    trainingData, testingData, trainingTargets, testingTargets = train_test_split(dataset, targets)

    # Saves the testing data and testing targets in global variables
    finalTestingData.append(testingData)
    finalTestingTargets.append(testingTargets)

    # Runs the perceptron algorithm with the training dataset and training targets
    # to determine the weights and the threshold
    perceptronAlgorithm(trainingData, trainingTargets, learningRate)

# Determines the accuracy of the algorithm with testing dataset and testing targets
for i in range(len(filePaths)):
    testingPerceptronAlgorithm(finalTestingData[i], finalTestingTargets[i], learningRate, finalWeights[i], finalThreshold[i])

# Prints the accuracy, weights and the threshold foe each data file
for x in range(len(filePaths)):
    print(filePaths[x].split("\\")[-1] + ": " + str(round(finalAccuracy[x], 2)) + "% W: " + str(finalWeights[x]) + " T: " + str(round(finalThreshold[x], 1)))

## Multi-Layer Perceptron

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

filePath_5 = "Data_Files\\TUANDROMD.csv"

filePaths.append(filePath_5)

# Runs a MLP for each file in the filePaths list
for filePath in filePaths:

    # Removes the first row of the dataset if the file is TUANDROMD.csv
    if (filePath == "Data_Files\\TUANDROMD.csv"):
        rawDataset = np.genfromtxt(filePath, delimiter=",", skip_header=1)
    else:
        rawDataset = np.genfromtxt(filePath, delimiter=",")

    dataset = rawDataset[:, 0:-1]
    targets = rawDataset[:,-1]

    # Splits the dataset and targets into training data, trainging targets,
    # testing data and testing targets
    trainingData, testingData, trainingTargets, testingTargets = train_test_split(dataset, targets)

    # Converts the lists into arrays
    trainingData = np.array(trainingData)
    testingData = np.array(testingData)
    trainingTargets = np.array(trainingTargets)
    testingTargets = np.array(testingTargets)

    # Shuffles the datasets and the targets
    i = np.arange( trainingData.shape[0] )
    np.random.shuffle(i)
    trainingData = trainingData[i]
    trainingTargets = trainingTargets[i]

    k = np.arange( testingData.shape[0] )
    np.random.shuffle(k)
    testingData = testingData[k]
    testingTargets = testingTargets[k]

    # Trains the multilayer perceptron
    multilayerPerceptron = MLPClassifier(hidden_layer_sizes=250, learning_rate_init=0.02, tol=0.04, n_iter_no_change=12)
    multilayerPerceptron = multilayerPerceptron.fit(trainingData, trainingTargets)

    # Gets the accuracy of the multilayer perceptron using testing data and testing
    # targets
    multilayerPerceptronPrediction = multilayerPerceptron.predict(testingData)
    multilayerPerceptronAccuracy = accuracy_score(testingTargets, multilayerPerceptronPrediction) * 100

    # Trains the decision tree classifier
    decisionTreeClassifier = DecisionTreeClassifier()
    decisionTreeClassifier = decisionTreeClassifier.fit(trainingData, trainingTargets)

    # Gets the accuracy of the decission tree classifier
    decisionTreeClassifierPrediction = decisionTreeClassifier.predict(testingData)
    decisionTreeClassifierAccuracy = accuracy_score(testingTargets, decisionTreeClassifierPrediction) * 100

    # Prints info for each file of data
    print("")
    print("File: ", filePath.split("\\")[-1])
    print("Decision Tree: ", round(decisionTreeClassifierAccuracy, 1), "% Accuracy")
    print("MLP: hidden layers = ", multilayerPerceptron.n_layers_, ", LR = 0.02, tol = 0.04")
    print(round(multilayerPerceptronAccuracy, 1), "% Accuracy ", multilayerPerceptron.n_iter_, " iterations")
