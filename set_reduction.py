import sys
import math
import numpy as np

# Reerence set reduction using Tomeks algorithm.
def main():
    print("Reference set reduction by Tomeks algorithm.")
    train_file =  input("Reference set file: ")
    output_file = input("Reduced set file: ")
    list_pairs = input("Enter list of pairs: ")

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

    output = open(list_pairs, "w")
    selected = [0.0 for i in range(n_objects)]
    for i in range(n_objects - 1):
        j = i + 1
        for j in range(n_objects):
            if classes[i] != classes[j]:
                mean = []
                a = 0
                for k in range(n_features): a = a + math.pow(features[i][k] - features[j][k], 2)
                a = math.sqrt(a) / 2.0
                for k in range(n_features): mean.append((features[i][k] + features[j][k]) / 2.0)
                flag = True
                object = 0
                while (flag == True) and (object < n_objects):
                    r = 0
                    for k in range(n_features): r = r + math.pow(features[object][k] - mean[k], 2)
                    r = math.sqrt(r)
                    if (r < a) and (object != i) and (object != j): flag = False
                    object = object + 1
                if flag:
                    selected[i] = 1
                    selected[j] = 1
                    output.write(str(selected.count(1)/2)+"\t"+str(i+1)+"\t"+str(j+1)+"\n")
    output.close()

    output = open(output_file, "w")
    output.write(str(n_classes)+"\t"+str(n_features)+"\t"+str(selected.count(1))+"\n")
    for i in range(n_objects):
        if selected[i] == 1:
            output.write("\n"+str(classes[i])+"\t")
            for j in range(n_features): output.write(str("%.4f" % features[i][j])+"\t")
    for i in range(n_objects):
        if selected[i] == 1: output.write("\n"+str(i+1))
    output.close()

main()
