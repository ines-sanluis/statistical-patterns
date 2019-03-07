import sys
import math
import numpy as np

global n_classes
global n_features
global n_objects

def getFileNames(train_file, test_file, output_file):
    if len(sys.argv) == 1:
        print("Using default values. To select your own run the program using: \n\tpython <script> <training file> <testing file> <output file>")
    else:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        output_file = sys.argv[3]
    print("Training file: ", train_file)
    print("Testing file: ", test_file)
    print("Output file: ", output_file)

def readTrainFile(file, classes, features):
    global n_classes
    global n_features
    global n_objects
    f = open(file, "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    n_objects = int(header.split()[2])
    for i in range(n_objects):
        feature = []
        line = f.readline()
        classes.append(line.split()[0])
        for j in range(n_features):
            feature.append(line.split()[j+1])
        features.append(feature)
    f.close()

def readTestFile(file, classes, features):
    global n_classes
    global n_features
    global n_objects
    f = open(file, "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    n_objects = int(header.split()[2])
    for i in range(n_objects):
        feature = []
        line = f.readline()
        classes.append(line.split()[0])
        for j in range(1, n_features+1):
            feature.append(line.split()[j])
        features.append(feature)
    f.close()

def calculateValues(mean_values, desviations, features):
    for j in range(n_features):
        mean_values.append(0)
        for i in range(n_objects):
            mean_values[j] = mean_values[j] + float(features[i][j])
        mean_values[j] = mean_values[j] / n_objects
        desviations.append(0)
        for i in range(n_objects):
            desviations[j] = desviations[j] + (float(features[i][j])-mean_values[j])**2
        desviations[j] = math.sqrt(desviations[j] / n_objects)

def numberElementsClass(file, classes):
    number = np.zeros(n_classes)
    types = list(set(classes))
    f = open(file, "r")
    header = f.readline()
    for i in range(n_objects):
         line = f.readline()
         clase = int(line.split()[0])
         number[clase - 1] = number[clase - 1]+1
    f.close()
    return number

def calculateGravityCenter(classes, features, elements_for_class):
    p = [[0.0 for i in range(n_features)] for j in range(n_classes)]
    for i in range(n_objects):
        clase = int(classes[i]) - 1
        for j in range(n_features):
            p[clase][j] = p[clase][j] + float(features[i][j])
    for i in range(n_classes):
        for j in range(n_features):
            p[i][j] = p[i][j] / elements_for_class[i]
    return p

def printMatrix(matrix):
    for i in range(n_classes):
        for j in range(n_features):
            print(matrix[i][j], end=" ")
        print("")

def standardise(p, mean_values, desviations):
    for i in range(n_classes):
        for j in range(n_features):
            p[i][j] = (p[i][j] - mean_values[j])/desviations[j];

def train(file, mean_values, desviations):
    classes = []
    features = []
    readTrainFile(file, classes, features)
    calculateValues(mean_values, desviations, features)
    elements_for_class = numberElementsClass(file, classes)
    p = calculateGravityCenter(classes, features, elements_for_class)
    print("Class gravity centers before standardisation: ")
    printMatrix(p)
    standardise(p, mean_values, desviations)
    print("Class gravity centers after standardisation: ")
    printMatrix(p)
    return p

def test(file, mean_values, desviations, p):
    classes = []
    features = []
    readTestFile(file, classes, features)
    error = 0
    for i in range(n_objects):
        for j in range(n_features):
            features[i][j] = (float(features[i][j])-mean_values[j])/desviations[j]
        dmin = 10e10;
        for k in range(n_classes):
            d = 0
            for j in range(n_features):
                d = d + (features[i][j]-p[k][j])**2
                if d < dmin:
                    dmin = d
                    assigned_class = k + 1
        if int(classes[i]) != assigned_class: error = error + 1;
        print(i+1, classes[i], assigned_class)
    print("Error rate ", round(100*error /360, 1), "%")
    print("Error rate ", round(100*error /600, 1), "%")


def main():
    train_file =  "trn.txt"
    test_file = "tst.txt"
    output_file = "result.txt"
    getFileNames(train_file, test_file, output_file)
    mean_values = []
    desviations = []
    p = train(train_file, mean_values, desviations)
    test(test_file, mean_values, desviations, p)

main()
