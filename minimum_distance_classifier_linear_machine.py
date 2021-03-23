import sys
import math
import numpy as np
# Minimum distance classifier as a linear machine

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

def elementsClass(file, classes):
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

def gravityCenters(classes, features, elements_for_class):
    p = [[0.0 for i in range(n_features)] for j in range(n_classes)]
    for i in range(n_objects):
        clase = int(classes[i]) - 1
        for j in range(n_features): p[clase][j] = p[clase][j] + float(features[i][j])
    for i in range(n_classes):
        for j in range(n_features): p[i][j] = p[i][j] / elements_for_class[i]
    return p

def printGravityCenters(output_file, gravity_centers):
    index = 1
    for i in gravity_centers:
        output_file.write("P"+str(index)+"\t")
        for j in i: output_file.write(str("%.3f" % j)+"\t")
        index = index + 1
        output_file.write("\n")

def printWeights(output_file, weights):
    index = 1
    for i in weights:
        output_file.write("w"+str(index)+"\t")
        for j in i: output_file.write(str("%.3f" % j)+"\t")
        index = index + 1
        output_file.write("\n")

def printValues(output_file, mean_values, desviations):
    output_file.write("\nmv\t")
    for i in mean_values: output_file.write(str("%.3f" % i)+"\t")
    output_file.write("\nsd\t")
    for i in desviations: output_file.write(str("%.3f" % i)+"\t")
    output_file.write("\n")

def standardise(p, mean_values, desviations):
    for i in range(n_classes):
        for j in range(n_features):
            p[i][j] = (p[i][j] - mean_values[j])/desviations[j];

def calculateValues(mean_values, desviations, features):
    for j in range(n_features):
        mean_values.append(0)
        for i in range(n_objects): mean_values[j] = mean_values[j] + float(features[i][j])
        mean_values[j] = mean_values[j] / n_objects
        desviations.append(0)
        for i in range(n_objects): desviations[j] = desviations[j] + (float(features[i][j])-mean_values[j])**2
        desviations[j] = math.sqrt(desviations[j] / n_objects)

def calculateWeights(output_file, gravity_centers, mean_values, desviations):
    w = [[0.0 for i in range(n_features+1)] for j in range(n_classes)]
    for i in range(n_classes):
        for j in range(n_features): w[i][j] = 2*gravity_centers[i][j]
        for k in range(n_features): w[i][j+1] = w[i][j+1] + gravity_centers[i][k]**2
        w[i][j+1] = w[i][j+1]*-1
    output_file.write("\nWeights before standardisation:\n")
    printWeights(output_file, w)
    for i in range(n_classes):
        suma = 0.0
        for j in range(n_features):
            w[i][j] = w[i][j] / desviations[j]
            suma = suma + mean_values[j]*w[i][j]
        w[i][j+1] = w[i][j+1] - suma
    output_file.write("\nWeights after standardisation:\n")
    printWeights(output_file, w)
    return w

def printStatistics(output_file, matrix):
    #Confussion matrix
    output_file.write("\nConfussion matrix:\n")
    output_file.write("\t")
    for i in range(n_classes): output_file.write("     "+str(i+1)+"\t")
    output_file.write("\n")
    for i in range(n_classes):
        output_file.write(str(i+1)+"\t")
        for j in range(n_classes): output_file.write("   "+str("%.1f" % matrix[i][j])+"\t")
        output_file.write("\n")
    #Probabilities a priori
    output_file.write("\nProbabilities a priori:\n")
    output_file.write("\t")
    for i in range(n_classes): output_file.write("     "+str(i+1)+"\t")
    output_file.write("\n")
    for i in range(n_classes):
        total = sum(matrix[i])
        output_file.write(str(i+1)+"\t")
        for j in range(n_classes):
            result = matrix[i][j] / total
            output_file.write(str("%.4f" % result)+"\t")
        output_file.write("\n")
    #Probabilities a posteriori
    output_file.write("\nProbabilities a posteriori:\n")
    output_file.write("\t")
    for i in range(n_classes): output_file.write("     "+str(i+1)+"\t")
    output_file.write("\n")
    column = [sum([row[i] for row in matrix]) for i in range(0,len(matrix[0]))] #columns sum
    for i in range(n_classes):
        output_file.write(str(i+1)+"\t")
        for j in range(n_classes):
            result = matrix[j][i] / column[i]
            output_file.write(str("%.4f" % result)+"\t")
        output_file.write("\n")

def test(output_file, test_file, weights):
    f = open(test_file, "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    n_objects = int(header.split()[2])
    error = 0
    cm = [[0 for i in range(n_classes)] for j in range(n_classes)]
    output_file.write("\nObject\t True class \t Assigned class\n")
    for i in range(n_objects):
        feature = []
        line = f.readline()
        real_class = int(line.split()[0])
        for j in range(n_features): feature.append(float(line.split()[j+1]))
        g = []
        for j in range(n_classes):
            g.append(0.0)
            for k in range(n_features): g[j] = g[j] + feature[k]*weights[j][k]
            g[j] = g[j] + weights[j][k+1]
        assigned_class = g.index(max(g)) + 1
        # print("Max ", str(round(max(g), 2)))
        if i < 9: output_file.write(" "+str(i+1)+"\t\t\t\t\t"+str(real_class)+"\t\t\t\t\t\t"+str(assigned_class)+"\n")
        else: output_file.write(str(i+1)+"\t\t\t\t\t"+str(real_class)+"\t\t\t\t\t\t"+str(assigned_class)+"\n")
        if(assigned_class != real_class): error = error + 1
        cm[real_class-1][assigned_class-1] = cm[real_class-1][assigned_class-1]+1
    error = (100*error)/n_objects
    output_file.write("\nError rate: "+str("%.1f" % error)+" %\n")
    printStatistics(output_file, cm)
    f.close()

def train(output_file, train_file, classes, features):
    elements = elementsClass(train_file, classes)
    p = gravityCenters(classes, features, elements)
    output_file.write("Class gravity centers before standardisation:\n")
    printGravityCenters(output_file, p)
    mean_values = []
    desviations = []
    calculateValues(mean_values, desviations, features)
    printValues(output_file, mean_values, desviations)
    standardise(p, mean_values, desviations)
    output_file.write("\nClass gravity centers after standardisation:\n")
    printGravityCenters(output_file, p)
    return calculateWeights(output_file, p, mean_values, desviations)

def main():
    train_file =  input("Enter train file: ")
    test_file = input("Enter test file: ")
    output_file = input("Enter output file: ")
    output = open(output_file, "w")

    classes = []
    features = []
    readFile(train_file, classes, features)
    weights = train(output, train_file, classes, features)
    test(output, test_file, weights)
    output.close()

main()
