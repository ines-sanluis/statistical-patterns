import sys
import math
import numpy as np

def printStatistics(output, error, matrix):
    #Error rate
    output.write("\nError rate: "+str("%.1f" % error)+" %\n")
    #Confussion matrix
    n_classes = len(matrix)
    output.write("\nConfussion matrix:\n")
    output.write("\t")
    for i in range(n_classes): output.write("     "+str(i+1)+"\t")
    output.write("\n")
    for i in range(n_classes):
        output.write(str(i+1)+"\t")
        for j in range(n_classes):
            output.write("   "+str(matrix[i][j])+"\t")
        output.write("\n")
    #Probabilities a priori
    output.write("\nProbabilities a priori:\n")
    output.write("\t")
    for i in range(n_classes): output.write("     "+str(i+1)+"\t")
    output.write("\n")
    for i in range(n_classes):
        total = sum(matrix[i])
        output.write(str(i+1)+"\t")
        for j in range(n_classes):
            result = matrix[i][j] / total
            output.write(str("%.4f" % result)+"\t")
        output.write("\n")
    #Probabilities a posteriori
    output.write("\nProbabilities a posteriori:\n")
    output.write("\t")
    for i in range(n_classes): output.write("     "+str(i+1)+"\t")
    output.write("\n")
    column = [sum([row[i] for row in matrix]) for i in range(0,len(matrix[0]))] #columns sum
    for i in range(n_classes):
        output.write(str(i+1)+"\t")
        for j in range(n_classes):
            result = matrix[j][i] / column[i]
            output.write(str("%.4f" % result)+"\t")
        output.write("\n")

def main():
    train_file =  input("Enter train file: ")
    test_file = input("Enter test file: ")
    output_file = input("Enter output file: ")
    output = open(output_file, "w")
    classes = []
    data = []
    mean_values = []
    desviations = []
    #Read train file
    f = open(train_file, "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    n_objects_trn = int(header.split()[2])
    for i in range(n_objects_trn):
        feature = []
        line = f.readline()
        classes.append(line.split()[0])
        for j in range(n_features): feature.append(line.split()[j+1])
        data.append(feature)
    f.close()
    #Modify data
    for j in range(n_features):
        mean_values.append(0)
        for i in range(n_objects_trn):
            mean_values[j] = mean_values[j] + float(data[i][j])
        mean_values[j] = mean_values[j] / n_objects_trn
        desviations.append(0)
        for i in range(n_objects_trn):
            desviations[j] = desviations[j] + (float(data[i][j])-mean_values[j])**2
        desviations[j] = math.sqrt(desviations[j] / n_objects_trn)
    for i in range(n_objects_trn):
        for j in range(n_features):
            if desviations[j] > 0.0001:
                data[i][j] = (float(data[i][j]) - mean_values[j]) / desviations[j];
    #Initialize values
    errors = 0
    cm = [[0 for i in range(n_classes)] for j in range(n_classes)]
    #Read test file
    f = open(test_file, "r")
    header = f.readline()
    n_classes = int(header.split()[0])
    n_features = int(header.split()[1])
    n_objects_tst = int(header.split()[2])
    #Test
    output.write("\nResults of classification:\n")
    output.write("Object,\tTrue class,\tAssigned class\n")
    for i in range(n_objects_tst):
        feature = []
        line = f.readline()
        real_class = int(line.split()[0])
        for j in range(n_features): feature.append(float(line.split()[j+1]))
        for j in range(n_features):
            if desviations[j] > 0.001:
                feature[j] = (feature[j] - mean_values[j]) / desviations[j]
        dmin = 10e10;
        distances = [0.0 for i in range(n_objects_trn)]
        for k in range(n_objects_trn):
            for j in range(n_features):
                distances[k] = distances[k] + (float(data[k][j] - feature[j]))**2
        dmin = min(distances)
        assigned_class = int(classes[distances.index(dmin)]) #+1
        if(assigned_class != real_class): errors = errors + 1
        cm[real_class-1][assigned_class-1] = cm[real_class-1][assigned_class-1] + 1;
        output.write(str(i+1)+"\t\t\t\t\t"+str(real_class)+"\t\t\t\t\t\t\t"+str(assigned_class)+"\n")
    #Print results
    d = 100.0 * errors / n_objects_tst
    printStatistics(output, d, cm)
    output.close()
main()
