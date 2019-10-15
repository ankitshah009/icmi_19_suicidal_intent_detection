import csv
import numpy as np
import pdb
from krippendorff import *
from sklearn.metrics import cohen_kappa_score

fname = 'Round_1.csv'
data_1 = []
with open(fname) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data_1.append(row)

data_1 = data_1[1:]
data_1 = np.array(data_1)

fname = 'Round_2.csv'
flag = True #header
data_2 = []
with open(fname) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data_2.append(row)

data_2 = data_2[1:]
data_2 = np.array(data_2)
scores3 = []

row_ids = ['Depression', 'Self-harm', 'Eye gaze', 'Distress (0) vs. Suicide (1)', 'Age', 'Short-term/Chronic distress', 'Hopelessness', 'Impulsiveness', 'Anhedonia', 'Lability', 'Guilt']
unordered_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10]
ordered_ids = [0, 1]

##########################################################################
#Krippendorff Alpha computation
#########################################################################
'''
    Arguements needed for Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
'''
#Unordered variables
data = []
#coder1 = {}
#coder2 = {}
for a_id in unordered_ids:
    data = [data_1[:,a_id], data_2[:,a_id]]
    alpha_or = krippendorff_alpha(data, metric=nominal_metric, convert_items=str)
    print("Krippendorff_Alpha for " + row_ids[a_id] + " is: " + str(alpha_or))
    cohen_k = cohen_kappa_score(data_1[:,a_id], data_2[:,a_id])
    print("Cohen's Kappa for " + row_ids[a_id] + " is: " + str(cohen_k))

    #coder1[row_ids[a_id]] = data_1[:,a_id]
    #coder2[row_ids[a_id]] = data_2[:,a_id]

#data.append(coder1)
#data.append(coder2)
#alpha_or = krippendorff_alpha(data, metric=nominal_metric, convert_items=int)   #For unordered variables



#Ordered variables
data = []
#coder1 = {}
#coder2 = {}
for a_id in ordered_ids:
    data = [data_1[:,a_id], data_2[:,a_id]]
    alpha_un = krippendorff_alpha(data, metric=interval_metric, convert_items=float)
    print("Krippendorff_Alpha for " + row_ids[a_id] + " is: " + str(alpha_un))
    cohen_k = cohen_kappa_score(data_1[:,a_id], data_2[:,a_id])
    print("Cohen's Kappa for " + row_ids[a_id] + " is: " + str(cohen_k))
    #coder1[row_ids[a_id]] = data_1[:,a_id]
    #coder2[row_ids[a_id]] = data_2[:,a_id]

#alpha_un = krippendorff_alpha(data, metric=interval_metric, convert_items=float)   #For ordered variables
pdb.set_trace()






#pdb.set_trace()
unordered_scores = []
for a_id in unordered_ids:
    score = np.sum(data_1[:,a_id] == data_2[:, a_id])/ float(data_1.shape[0])
    unordered_scores.append(score)
    print("IRR for " + row_ids[a_id] +" is "+ str(score))



scores_5 = [1, 0.972, 0.888, 0.75, 0.30, 0]
scores_3 = [1, 0.889, 0.556, 0]
ordered_scores = []
for a_id in ordered_ids:
    if a_id == 0:
            scores = []
            for i in range(data_1.shape[0]):
                scores3.append(scores_5[np.abs(int(data_1[i, a_id]) - int(data_2[i, a_id]))])
            scores3 = np.array(scores3)
            fin_score = np.sum(scores3) / float(data_1.shape[0])
            ordered_scores.append(fin_score)
            print("Quadratic weighted ordered IRR for " + row_ids[a_id] +" is "+ str(fin_score))

    if a_id == 1:

            scores3 = []
            for i in range(data_1.shape[0]):
                scores3.append(scores_3[np.abs(int(data_1[i, a_id]) - int(data_2[i, a_id]))])
            scores3 = np.array(scores3)
            fin_score = np.sum(scores3) / float(data_1.shape[0])
            ordered_scores.append(fin_score)
            print("Quadratic weighted ordered IRR for " + row_ids[a_id] +" is "+ str(fin_score))

    
