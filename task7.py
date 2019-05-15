import sys
import math
import numpy as np

def main():
    print(' Programm for class areas determination.')
    train_file =  input("Training set file: ")
    output_file = input("Output set file: ")
    output = open(output_file, "w")

    f = open(train_file,  "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    n_objects = int(header.split()[2])
    classes = []
    features = []
    for i in range(n_objects):
        feature = []
        line = f.readline()
        classes.append(int(line.split()[0]))
        for j in range(n_features): feature.append(float(line.split()[j+1]))
        features.append(feature)
    f.close()

    output.write("Number of classes: ")
    unique_classes = np.unique(np.array(classes))
    for i in unique_classes: output.write(str(classes.count(i))+"\t")
    output.write("\n")

    mean_values = []
    desviations = []
    for i in range(n_features):
        mean_values.append(0.0)
        for j in range(n_objects): mean_values[i] = mean_values[i] + features[j][i]
        mean_values[i] = mean_values[i] / n_objects
        desviations.append(0.0)
        for j in range(n_objects): desviations[i] = desviations[i] + math.pow(features[j][i] - mean_values[i], 2)
        desviations[i] = math.sqrt(desviations[i] / n_objects)
        if desviations[i] > 0.0001:
            for j in range(n_objects): features[j][i] = (features[j][i] - mean_values[i]) / desviations[i]

    e = []
    for i in range(n_classes):
        e.append(-10e10)
        for j in range(n_objects): #first object
            if classes[j] == (i+1):
                dmin = 10e10
                for k in range(n_objects): #second object
                    if (j != k) and (classes[k] == (i+1)):
                        d = 0.0
                        for l in range(n_features): d = d + math.pow(features[j][l] - features[k][l], 2)
                        d = math.sqrt(d)
                        if d < dmin: dmin = d
                if dmin > e[i]: e[i] = dmin
    output.write("Object\tClass\t")
    for j in range(n_classes): output.write(" A"+str(j+1)+"\t")
    output.write("\n")

    fraction = 0.0
    f = [0 for k in range(n_classes)]
    for i in range(n_objects):
        for j in range(n_classes):
            dmin = 10e10
            f[j] = 0
            for k in range(n_objects):
                if classes[k] == (j+1):
                    d = 0.0
                    for l in range(n_features): d = d + math.pow(features[i][l] - features[k][l], 2)
                    d = math.sqrt(d)
                    if d < dmin: dmin = d
            if dmin <= e[j]: f[j] = 1
        if sum(f) > 1: fraction = fraction + 1.0
        if i < 9: output.write("\t "+str(i+1)+"\t\t"+str(classes[i])+"\t\t\t ")
        else: output.write("\t"+str(i+1)+"\t\t"+str(classes[i])+"\t\t\t ")
        for j in range(n_classes): output.write(str(f[j])+"\t\t")
        output.write("\n")

    fraction = fraction / n_objects
    output.write("\nFraction: "+str("%.4f" % fraction))
    output.close()

main()
