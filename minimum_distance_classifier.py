import sys
import math
import numpy as np

global n_classes
global n_features
global n_objects

def readFile(file, classes, features):
    global n_classes, n_features, n_objects
    f = open(file, "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    n_objects = int(header.split()[2])
    for i in range(n_objects):
        feature = []
        line = f.readline()
        classes.append(line.split()[0])
        for j in range(n_features): feature.append(line.split()[j+1])
        features.append(feature)
    f.close()

def calculateValues(mean_values, desviations, features):
    for j in range(n_features):
        mean_values.append(0)
        for i in range(n_objects): mean_values[j] = mean_values[j] + float(features[i][j])
        mean_values[j] = mean_values[j] / n_objects
        desviations.append(0)
        for i in range(n_objects): desviations[j] = desviations[j] + (float(features[i][j])-mean_values[j])**2
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
        for j in range(n_features): p[clase][j] = p[clase][j] + float(features[i][j])
    for i in range(n_classes):
        for j in range(n_features): p[i][j] = p[i][j] / elements_for_class[i]
    return p

def printMatrix(output, matrix):
    for i in range(n_classes):
        for j in range(n_features):
            decimal = "%.3f" % matrix[i][j]
            output.write(str(decimal)+"\t")
        output.write("\n")

def standardise(p, mean_values, desviations):
    for i in range(n_classes):
        for j in range(n_features):
            p[i][j] = (p[i][j] - mean_values[j])/desviations[j];

def train(file, mean_values, desviations, output):
    classes = []
    features = []
    readFile(file, classes, features)
    calculateValues(mean_values, desviations, features)
    elements_for_class = numberElementsClass(file, classes)
    p = calculateGravityCenter(classes, features, elements_for_class)
    output.write("Class gravity centers before standardisation:\n")
    printMatrix(output, p)
    standardise(p, mean_values, desviations)
    output.write("\nClass gravity centers after standardisation:\n")
    printMatrix(output, p)
    return p

def test(file, mean_values, desviations, p, output):
    f = open(file, "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    n_objects = int(header.split()[2])
    error = 0
    output.write("\nResults of classification:\n")
    output.write("Object\tTrue class\tAssigned class\n")
    for i in range(n_objects):
        feature = []
        line = f.readline()
        real_class = int(line.split()[0])
        for j in range(n_features): feature.append(float(line.split()[j+1]))
        for j in range(n_features): feature[j] = (feature[j]-mean_values[j])/desviations[j]
        dmin=10e10
        distances = [0.0 for i in range(n_classes)]
        for k in range(n_classes):
            for z in range(n_features):
                distances[k] = distances[k] + (feature[z]-p[k][z])**2
        dmin = min(distances)
        assigned_class = distances.index(dmin) + 1
        if i < 9: output.write(" "+str(i+1)+"\t\t\t\t\t"+str(real_class)+"\t\t\t\t\t"+str(assigned_class)+"\n")
        else: output.write(str(i+1)+"\t\t\t\t\t"+str(real_class)+"\t\t\t\t\t"+str(assigned_class)+"\n")
        if (real_class != assigned_class): error = error+1
    print("Number of errors: ", error)
    error = (100*error)/n_objects
    print("Error rate: %.1f" % error)
    output.write("\nError rate: "+str("%.1f" % error))

def main():
    train_file =  input("Enter train file: ")
    test_file = input("Enter test file: ")
    output_file = input("Enter output file: ")
    output = open(output_file, "w")

    mean_values = []
    desviations = []
    p = train(train_file, mean_values, desviations, output)
    test(test_file, mean_values, desviations, p, output)

    output.close()

main()
