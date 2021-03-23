import sys
import math
import numpy as np
from decimal import Decimal
from operator import add

global n_classes
global file_size
global n_features
global n_objects
global selected

# Training a linear classifier based on error correction algorithm

def readFile(file, classes, features):
    global n_classes, n_features, n_objects, file_size, selected
    f = open(file,  "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    file_size = int(header.split()[2])
    selected = "all"
    if n_classes > 2:
        selected = input("Select classes: ")
        n_objects = 0
        flag = True
    for i in range(file_size):
        feature = []
        line = f.readline()
        real_class = line.split()[0]
        if (selected == "all") or (selected.find(real_class) != -1):
            if(flag): n_objects = n_objects + 1
            classes.append(line.split()[0])
            for j in range(n_features): feature.append(line.split()[j+1])
            features.append(feature)
    f.close()

def code_object(features, classes):
    global n_classes, n_features
    #Transform into float
    f = []
    c = []
    for i in features:
        list = []
        for j in i: list.append(float(j))
        f.append(list)
    for i in classes: c.append(float(i))
    #Transform data
    X = []
    for patron in range(len(f)):
        x = [0.0 for i in range(n_features+1)]
        for i in range(n_features):
            if(c[patron] == 1): x[i] = f[patron][i]
            else: x[i] = - float(f[patron][i])
        if(c[patron] == 1): x[i+1] = 1
        else: x[i+1] = -1
        X.append(x)
    return X

def train(output, data, iterations):
    global n_features
    v = [0.0 for i in range(n_features+1)]
    new_v = [0.0 for i in range(n_features+1)]
    #bucle
    corrections = 0
    scalars = []
    nIterations = 0
    for i in range(iterations):
        for y in data:
            nIterations += 1
            scalar = 0
            for i in range(n_features+1): scalar += y[i]*v[i]
            if(scalar <= 0):
                v = list(map(add, v, y))
                corrections = corrections + 1
            scalars.append(scalar)
    output.write("\nTraining set size: " + str(n_objects))
    output.write("\nNumber of corrections: " + str(corrections))
    output.write("\nWeight of discriminant function:\n")
    for i in v: output.write(str("\t\t%.3f" % i)+"\n")
    return v

def well_classified(output, file, v):
    global selected, file_size
    f = open(file, "r")
    header = f.readline()
    well_classified = 0
    for i in range(file_size):
        discriminant = 0
        line = f.readline()
        real_class = int(line.split()[0])
        assigned_class = 0
        if (selected == "all") or (selected.find(str(real_class)) != -1):
            for j in range(n_features):
                feature = float(line.split()[j+1])
                discriminant = discriminant + (v[j]*feature)
            discriminant = discriminant + v[j+1]
            if discriminant >= 0: assigned_class = 1
            else: assigned_class = 2
            if assigned_class == real_class: well_classified = well_classified + 1
    f.close()
    output.write("\nNumber of correct decisions: " + str(well_classified))

def main():
    train_file =  input("Enter train file: ")
    output_file = input("Enter output file: ")
    iterations = int(input("Enter number of iterations: "))
    output = open(output_file, "w")

    classes = []
    features = []
    readFile(train_file, classes, features)
    data = code_object(features, classes)
    v = train(output, data, iterations)
    well_classified(output, train_file, v)
    output.close()

main()
