import numpy as np

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
import pdb

import warnings
warnings.filterwarnings("ignore")


        

data = np.load('/media/bighdd7/vasu/depression_detection_11776/data/all_features.npy').item()
vid_ids = data.keys()
#[37, 52, 74, 77, 24, 60, 44, 70, 71, 17, 19, 56, 55, 3, 28, 18, 67, 39, 85, 73, 5, 86, 87, 26, 83, 10, 30, 38, 59, 4, 32, 1, 76, 64, 89, 21, 79, 6, 45, 81, 7, 22, 14, 23, 72, 57, 46, 12, 8, 68, 82, 58, 61, 47, 40, 34, 13, 50, 48, 27, 78, 69, 43, 90, 84, 29, 62, 2, 16, 35, 53, 80, 25, 15, 65, 63, 54, 31, 51, 49, 42, 66, 75, 41, 33, 11, 20, 88, 9, 36]

feats = data[1].keys()
#['F1', 'F2', 'F3', 'F1_bandwidth', 'F2_bandwidth', 'F3_bandwidth', 'loudness', 'spectral_flux', 'voiced_segments', 'unvoiced_segments', 'F0', 'liwc', 'bert', 'pose', 'openface', 'label']


#F0 and openface are time series
X_data = []
Y_labels = []
for vid in vid_ids:
        print(vid)
        if vid == 36:
            print("Skipping video 36. Lacks acousitc and visual feats")
            continue
        Y_labels.append(data[vid]['label'])
        feat_vec = []

        #Add acoustic features
        feat_vec.append(data[vid]['F1']) #scalar
        feat_vec.append(data[vid]['F2']) #scalar
        feat_vec.append(data[vid]['F3']) #scalar
        feat_vec.append(data[vid]['F1_bandwidth']) #scalar
        feat_vec.append(data[vid]['F2_bandwidth']) #scalar
        feat_vec.append(data[vid]['F3_bandwidth']) #scalar
        feat_vec.append(data[vid]['loudness']) #scalar
        feat_vec.append(data[vid]['spectral_flux']) #scalar
        feat_vec.append(data[vid]['voiced_segments']) #scalar
        feat_vec.append(data[vid]['unvoiced_segments']) #scalar

        #F0 is time series of scalars (later think of LSTM and if it can be used)
        #Convert to features as follows
        #1) Average of the non-zero portions 2) max amplitude of non zeros portions 3)Average length of zero segments  4) Ratio of zero to non zero segments
        F0_data = np.array(data[vid]['F0'])
        avg_F0_data = np.mean(F0_data[F0_data != 0.0])
        max_F0_data = np.max(F0_data)
        n_zero_seg = 0
        n_nonzero_seg = 0
        flag = 0
        n_zeros = np.sum(F0_data == 0.0)
        i = 0
        while i < len(F0_data):
            if i == 0:
                if F0_data[i] == 0.0:
                    flag = 0
                    n_zero_seg += 1
                else:
                    flag = 1
                    n_nonzero_seg += 1

            elif flag == 0 and F0_data[i] != 0.0:
                flag = 1
                n_nonzero_seg += 1

            elif flag == 1 and F0_data[i] == 0.0:
                flag = 0
                n_zero_seg += 1

            i += 1

        avg_zero_seg = n_zeros / n_zero_seg 
        ratio_zero_nzero = float(n_zero_seg) / float(n_nonzero_seg)

        feat_vec += [avg_F0_data, max_F0_data, avg_zero_seg, ratio_zero_nzero ]

        audio_feat_len = len(feat_vec)
        #Add verbal features
        feat_vec += data[vid]['liwc'].tolist() #16 dim
        feat_vec += data[vid]['bert'].tolist() #768 dim (sentiment weighted word average embedding)

        verbal_feat_len = len(feat_vec) - audio_feat_len

        #Add visual features
        feat_vec += data[vid]['pose'] #25 dim
        #feat_vec += data[vid]['openface'] #1502 dim?? time series??
        ###################################################################
        #Openface processing
        face = data[vid]['openface']
        face_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 676, 678, 683, 686, 687] #, 693, 695, 700, 703, 704] #################### AU 1, 4, 10, 15 and 17 along with eye gaze x,y,z co-ordinates of both eyes and gaze angle x and y
        feat_dict = {}
        for r_num in range(len(face)):
            if r_num == 0:
                rnames = face[r_num][0].strip().split(', ')
                for ind in face_indices:
                    feat_dict[rnames[ind]] = []

            else:
                rvals = face[r_num][0].strip().split(', ')
                rvals = [float(x) for x in rvals]
                for ind in face_indices:
                    feat_dict[rnames[ind]].append(rvals[ind])               


        for ind in face_indices:
            key = rnames[ind] 
            feat_vec += [np.mean(np.array(feat_dict[key]))]
            feat_vec += [np.std(np.array(feat_dict[key]))] #For gaze and AU based features also add the standard deviation of the values to track deviations
        ###################################################################

        visual_feat_len = len(feat_vec) - audio_feat_len - verbal_feat_len # 25(pose) + 28(face) = 53 dim 



        #Add to main data matrix
        X_data.append(feat_vec)

#Convert to numpy
X_data = np.array(X_data)
from sklearn import preprocessing
X_data = preprocessing.scale(X_data) #Mean normalize and std deviation scale

Y_labels = np.array(Y_labels)

##########################################################################
#Process manual features
##########################################################################
manual_feats = '/media/bighdd7/vasu/depression_detection_11776/project_files/dataset/merged.csv'
order = [75, 77, 12, 49, 7, 50, 39, 20, 23, 76, 58, 64, 34, 8, 71, 37, 29, 74, 3, 45, 40, 48, 2, 62, 68, 81, 82, 54, 69, 67, 5, 30, 46, 59, 70, 72, 10, 87, 52, 32, 33, 41, 90, 55, 85, 11, 21, 44, 89, 79, 31, 14, 27, 4, 57, 25, 51, 47, 88, 43, 9, 73, 84, 18, 38, 35, 78, 13, 28, 80, 42, 36, 56, 16, 6, 1, 83, 66, 17, 60, 26, 61, 53, 22, 63, 65, 15, 19, 24, 86]

import csv

cntr = -1
ethnic_dict = {}
ethnic_cnt = 0
manual_feat_dict = {}
with open(manual_feats) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        
        if cntr == -1:
            cntr = 0
            continue   #Header row ignore

        vid_id = order[cntr]
        #label = data[vid_id]['label']
        ethnicity = row[6]
        if ethnicity in ethnic_dict.keys():
            row[6] = ethnic_dict[ethnicity]
        else:
            ethnic_dict[ethnicity] = ethnic_cnt
            ethnic_cnt += 1
            row[6] = ethnic_dict[ethnicity]

        row = [int(x) for x in row] #Convert features to ints (problem with ethnicity)
        del(row[3]) #Remove target variable from list
        manual_feat_dict[vid_id] = row
        cntr += 1

        #pdb.set_trace()


X_manual_data = []
for vid in vid_ids:
    print(vid)
    if vid == 36:
            print("Skipping video 36. Lacks acousitc and visual feats")
            continue
    X_manual_data.append(manual_feat_dict[vid]) #Arrange in same order as automatic data

X_manual_data = np.array(X_manual_data)

print("Dimensions of  manual data matrix are: " + str(X_manual_data.shape))





############################################################################

print("Dimensions of data matrix are: " + str(X_data.shape))

pdb.set_trace()
#Baselines
'''
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

#Uncomment to run SVm fits

#1. All features concatenate and use linear SVM
clf1 = SVC(kernel='linear')
clf1.fit(X_data, Y_labels)
train_acc1 = clf1.score(X_data, Y_labels)
print("Linear SVM with all features has accuracy: "+ str(train_acc1))
out1 = clf1.predict(X_data)
print(classification_report(Y_labels, out1))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5)
print("Linear SVM with all features has 5 fold CV accuracy: " + str(scores1) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores1)))
scores1_p = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='precision')
print("Linear SVM with all features has 5 fold CV prec: " + str(scores1_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores1_p)))
scores1_r = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='recall')
print("Linear SVM with all features has 5 fold CV recall: " + str(scores1_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores1_r)))
scores1_f1 = (2 * scores1_p * scores1_r )/ (scores1_p + scores1_r)
print("Linear SVM with all features has 5 fold CV f1: " + str(scores1_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores1_f1)))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='roc_auc')
print("Linear SVM with all features has 5 fold CV AUC: " + str(scores1) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores1)))

clf2 = SVC(kernel='rbf')
clf2.fit(X_data, Y_labels)
train_acc2 = clf2.score(X_data, Y_labels)
print(" SVM with RBF with all features has accuracy: "+ str(train_acc2))
out2 = clf2.predict(X_data)
print(classification_report(Y_labels, out2))
scores2 = cross_val_score(clf2, X_data, Y_labels, cv=5)
print("RBF SVM with all features has 5 fold CV accuracy: " + str(scores2) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores2)))
scores2_p = cross_val_score(clf2, X_data, Y_labels, cv=5, scoring='precision')
print("RBF SVM with all features has 5 fold CV prec: " + str(scores2_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores2_p)))
scores2_r = cross_val_score(clf2, X_data, Y_labels, cv=5, scoring='recall')
print("RBF SVM with all features has 5 fold CV recall: " + str(scores2_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores2_r)))
scores2_f1 = (2 * scores2_p * scores2_r )/ (scores2_p + scores2_r)
print("RBF SVM with all features has 5 fold CV f1: " + str(scores2_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores2_f1)))
scores2 = cross_val_score(clf2, X_data, Y_labels, cv=5, scoring='roc_auc')
print("RBF SVM with all features has 5 fold CV AUC: " + str(scores2) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores2)))

#2. separate features (vocal, visual, verbal) and pairs of these with linear SVM
#Acoustic
clf3 = SVC(kernel='linear')
clf3.fit(X_data[:, :audio_feat_len ], Y_labels)
train_acc3 = clf3.score(X_data[:, :audio_feat_len ], Y_labels)
print("Linear SVM with acoustic features has accuracy: "+ str(train_acc3))
out3 = clf3.predict(X_data[:, :audio_feat_len ])
print(classification_report(Y_labels, out3))
scores3 = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5)
print("Linear SVM with acoustic features has 5 fold CV accuracy: " + str(scores3) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores3)))
scores3_p = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='precision')
print("Linear SVM with acoustic features has 5 fold CV prec: " + str(scores3_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores3_p)))
scores3_r = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='recall')
print("Linear SVM with acoustic features has 5 fold CV recall: " + str(scores3_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores3_r)))
scores3_f1 = (2 * scores3_p * scores3_r )/ (scores3_p + scores3_r)
print("Linear SVM with acoustic features has 5 fold CV f1: " + str(scores3_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores3_f1)))
scores3 = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("Linear SVM with acoustic features has 5 fold CV AUC: " + str(scores3) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores3)))

#Verbal
clf4 = SVC(kernel='linear')
clf4.fit(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels)
train_acc4 = clf4.score(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels)
print("Linear SVM with verbal features has accuracy: "+ str(train_acc4))
out4 = clf4.predict(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ])
print(classification_report(Y_labels, out4))
scores4 = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5)
print("Linear SVM with verbal features has 5 fold CV accuracy: " + str(scores4) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores4)))
scores4_p = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='precision')
print("Linear SVM with verbal features has 5 fold CV prec: " + str(scores4_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores4_p)))
scores4_r = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='recall')
print("Linear SVM with verbal features has 5 fold CV recall: " + str(scores4_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores4_r)))
scores4_f1 = (2 * scores4_p * scores4_r )/ (scores4_p + scores4_r)
print("Linear SVM with verbal features has 5 fold CV f1: " + str(scores4_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores4_f1)))
scores4 = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("Linear SVM with verbal features has 5 fold CV AUC: " + str(scores4) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores4)))

#Visual
clf5 = SVC(kernel='linear')
clf5.fit(X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels)
train_acc5 = clf5.score(X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels)
print("Linear SVM with visual features has accuracy: "+ str(train_acc5))
out5 = clf5.predict(X_data[:,  audio_feat_len + verbal_feat_len : ])
print(classification_report(Y_labels, out5))
scores5 = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5)
print("Linear SVM with visual features has 5 fold CV accuracy: " + str(scores5) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores5)))
scores5_p = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='precision')
print("Linear SVM with visual features has 5 fold CV prec: " + str(scores5_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores5_p)))
scores5_r = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='recall')
print("Linear SVM with visual features has 5 fold CV recall: " + str(scores5_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores5_r)))
scores5_f1 = (2 * scores5_p * scores5_r )/ (scores5_p + scores5_r)
print("Linear SVM with visual features has 5 fold CV f1: " + str(scores5_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores5_f1)))
scores5 = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='roc_auc')
print("Linear SVM with visual features has 5 fold CV AUC: " + str(scores5) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores5)))

#Acoustic (RBF)
clf6 = SVC(kernel='rbf')
clf6.fit(X_data[:, :audio_feat_len ], Y_labels)
train_acc6 = clf6.score(X_data[:, :audio_feat_len ], Y_labels)
print("RBF SVM with acoustic features has accuracy: "+ str(train_acc6))
out6 = clf6.predict(X_data[:, :audio_feat_len ])
print(classification_report(Y_labels, out6))
scores6 = cross_val_score(clf6, X_data[:, :audio_feat_len ], Y_labels, cv=5)
print("RBF SVM with acoustic features has 5 fold CV accuracy: " + str(scores6) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores6)))
scores6_p = cross_val_score(clf6, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='precision')
print("RBF  SVM with acoustic features has 5 fold CV prec: " + str(scores6_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores6_p)))
scores6_r = cross_val_score(clf6, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='recall')
print("RBF SVM with acoustic features has 5 fold CV recall: " + str(scores6_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores6_r)))
scores6_f1 = (2 * scores6_p * scores6_r )/ (scores6_p + scores6_r)
print("RBF  SVM with acoustic features has 5 fold CV f1: " + str(scores6_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores6_f1)))
scores6 = cross_val_score(clf6, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("RBF  SVM with acoustic features has 5 fold CV AUC: " + str(scores6) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores6)))

#Verbal(RBF)
clf7 = SVC(kernel='rbf')
clf7.fit(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels)
train_acc7 = clf7.score(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels)
print("RBF SVM with verbal features has accuracy: "+ str(train_acc7))
out7 = clf7.predict(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ])
print(classification_report(Y_labels, out7))
scores7 = cross_val_score(clf7, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5)
print("RBF SVM with verbal features has 5 fold CV accuracy: " + str(scores7) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores7)))
scores7_p = cross_val_score(clf7, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='precision')
print("RBF SVM with verbal features has 5 fold CV prec: " + str(scores7_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores7_p)))
scores7_r = cross_val_score(clf7, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='recall')
print("RBF SVM with verbal features has 5 fold CV recall: " + str(scores7_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores7_r)))
scores7_f1 = (2 * scores7_p * scores7_r )/ (scores7_p + scores7_r)
print("RBF SVM with verbal features has 5 fold CV f1: " + str(scores7_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores7_f1)))
scores7 = cross_val_score(clf7, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("RBF SVM with verbal features has 5 fold CV AUC: " + str(scores7) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores7)))

#Visual(RBF)
clf8 = SVC(kernel='rbf')
clf8.fit(X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels)
train_acc8 = clf8.score(X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels)
print("RBF SVM with visual features has accuracy: "+ str(train_acc8))
out8 = clf8.predict(X_data[:,  audio_feat_len + verbal_feat_len : ])
print(classification_report(Y_labels, out8))
scores8 = cross_val_score(clf8, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5)
print("RBF SVM with visual features has 5 fold CV accuracy: " + str(scores8) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores8)))
scores8_p = cross_val_score(clf8, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='precision')
print("RBF SVM with visual features has 5 fold CV prec: " + str(scores8_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores8_p)))
scores8_r = cross_val_score(clf8, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='recall')
print("RBF SVM with visual features has 5 fold CV recall: " + str(scores8_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores8_r)))
scores8_f1 = (2 * scores8_p * scores8_r )/ (scores8_p + scores8_r)
print("RBF SVM with visual features has 5 fold CV f1: " + str(scores8_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores8_f1)))
scores8 = cross_val_score(clf8, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='roc_auc')
print("RBF SVM with visual features has 5 fold CV AUC: " + str(scores8) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores8)))

#Acoustic and verbal
clf9 = SVC(kernel='linear')
clf9.fit(X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels)
train_acc9 = clf9.score(X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels)
print("Linear SVM with acoustic + verbal features has accuracy: "+ str(train_acc9))
out9 = clf9.predict(X_data[:, :audio_feat_len + verbal_feat_len ])
print(classification_report(Y_labels, out9))
scores9 = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5)
print("Linear SVM with acoustic+verbal features has 5 fold CV accuracy: " + str(scores9) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores9)))
scores9_p = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='precision')
print("Linear SVM with acoustic+verbal features has 5 fold CV prec: " + str(scores9_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores9_p)))
scores9_r = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='recall')
print("Linear SVM with acoustic+verbal features has 5 fold CV recall: " + str(scores9_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores9_r)))
scores9_f1 = (2 * scores9_p * scores9_r )/ (scores9_p + scores9_r)
print("Linear SVM with acoustic+verbal features has 5 fold CV f1: " + str(scores9_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores9_f1)))
scores9 = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("Linear SVM with acoustic+verbal features has 5 fold CV AUC: " + str(scores9) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores9)))

#Visual and verbal
clf10 = SVC(kernel='linear')
clf10.fit(X_data[:, audio_feat_len:  ], Y_labels)
train_acc10 = clf10.score(X_data[:, audio_feat_len: ], Y_labels)
print("Linear SVM with visual + verbal features has accuracy: "+ str(train_acc10))
out10 = clf10.predict(X_data[:, audio_feat_len: ])
print(classification_report(Y_labels, out10))
scores10 = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5)
print("Linear SVM with visual+verbal features has 5 fold CV accuracy: " + str(scores10) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores10)))
scores10_p = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='precision')
print("Linear SVM with visual+verbalfeatures has 5 fold CV prec: " + str(scores10_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores10_p)))
scores10_r = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='recall')
print("Linear SVM with visual+verbalfeatures has 5 fold CV recall: " + str(scores10_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores10_r)))
scores10_f1 = (2 * scores10_p * scores10_r )/ (scores10_p + scores10_r)
print("Linear SVM with visual+verbal features has 5 fold CV f1: " + str(scores10_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores10_f1)))
scores10 = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='roc_auc')
print("Linear SVM with visual+verbal features has 5 fold CV AUC: " + str(scores10) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores10)))

#Visual and Acoustic
X_joint_data = np.concatenate((X_data[:, :audio_feat_len ], X_data[:,  audio_feat_len + verbal_feat_len : ]), axis=1)
clf11 = SVC(kernel='linear')
clf11.fit(X_joint_data, Y_labels)
train_acc11 = clf11.score(X_joint_data, Y_labels)
print("Linear SVM with acoustic + visual features has accuracy: "+ str(train_acc11))
out11 = clf11.predict(X_joint_data)
print(classification_report(Y_labels, out11))
scores11 = cross_val_score(clf11, X_joint_data, Y_labels, cv=5)
print("Linear SVM with visual+acoustic features has 5 fold CV accuracy: " + str(scores11) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores11)))
scores11_p = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='precision')
print("Linear SVM with visual+acoustic features has 5 fold CV prec: " + str(scores11_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores11_p)))
scores11_r = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='recall')
print("Linear SVM with visual+acoustic features has 5 fold CV recall: " + str(scores11_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores11_r)))
scores11_f1 = (2 * scores11_p * scores11_r )/ (scores11_p + scores11_r)
print("Linear SVM with visual+acoustic features has 5 fold CV f1: " + str(scores11_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores11_f1)))
scores11 = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='roc_auc')
print("Linear SVM with visual+acoustic features has 5 fold CV AUC: " + str(scores11) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores11)))





####################################################################################################################################################################
#Use Random Forests
###################################################################################################################################################################
from sklearn.ensemble import RandomForestClassifier
#1. All features concatenate and use linear SVM
clf1 = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight="balanced")
clf1.fit(X_data, Y_labels)
train_acc1 = clf1.score(X_data, Y_labels)
print("Random Forest with all features has accuracy: "+ str(train_acc1))
out1 = clf1.predict(X_data)
print(classification_report(Y_labels, out1))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5)
print("Random Forest with all features has 5 fold CV accuracy: " + str(scores1) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores1)))
scores1_p = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='precision')
print("Random Forest with all features has 5 fold CV prec: " + str(scores1_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores1_p)))
scores1_r = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='recall')
print("Random Forest with all features has 5 fold CV recall: " + str(scores1_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores1_r)))
scores1_f1 = (2 * scores1_p * scores1_r )/ (scores1_p + scores1_r)
print("Random Forest with all features has 5 fold CV f1: " + str(scores1_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores1_f1)))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='roc_auc')
print("Random Forest with all features has 5 fold CV AUC: " + str(scores1) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores1)))


#2. separate features (vocal, visual, verbal) and pairs of these with linear SVM
#Acoustic
clf3 = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight="balanced")
clf3.fit(X_data[:, :audio_feat_len ], Y_labels)
train_acc3 = clf3.score(X_data[:, :audio_feat_len ], Y_labels)
print("Random Forest with acoustic features has accuracy: "+ str(train_acc3))
out3 = clf3.predict(X_data[:, :audio_feat_len ])
print(classification_report(Y_labels, out3))
scores3 = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5)
print("Random Forest with acoustic features has 5 fold CV accuracy: " + str(scores3) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores3)))
scores3_p = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='precision')
print("Random Forest with acoustic features has 5 fold CV prec: " + str(scores3_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores3_p)))
scores3_r = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='recall')
print("Random Forest with acoustic features has 5 fold CV recall: " + str(scores3_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores3_r)))
scores3_f1 = (2 * scores3_p * scores3_r )/ (scores3_p + scores3_r)
print("Random Forest with acoustic features has 5 fold CV f1: " + str(scores3_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores3_f1)))
scores3 = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("Random Forest with acoustic features has 5 fold CV AUC: " + str(scores3) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores3)))

#Verbal
clf4 = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight="balanced")
clf4.fit(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels)
train_acc4 = clf4.score(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels)
print("Random Forest with verbal features has accuracy: "+ str(train_acc4))
out4 = clf4.predict(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ])
print(classification_report(Y_labels, out4))
scores4 = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5)
print("Random Forest with verbal features has 5 fold CV accuracy: " + str(scores4) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores4)))
scores4_p = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='precision')
print("Random Forest with verbal features has 5 fold CV prec: " + str(scores4_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores4_p)))
scores4_r = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='recall')
print("Random Forest with verbal features has 5 fold CV recall: " + str(scores4_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores4_r)))
scores4_f1 = (2 * scores4_p * scores4_r )/ (scores4_p + scores4_r)
print("Random Forest with verbal features has 5 fold CV f1: " + str(scores4_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores4_f1)))
scores4 = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("Random Forest with verbal features has 5 fold CV AUC: " + str(scores4) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores4)))

#Visual
clf5 = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight="balanced")
clf5.fit(X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels)
train_acc5 = clf5.score(X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels)
print("Random Forest with visual features has accuracy: "+ str(train_acc5))
out5 = clf5.predict(X_data[:,  audio_feat_len + verbal_feat_len : ])
print(classification_report(Y_labels, out5))
scores5 = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5)
print("Random Forestwith visual features has 5 fold CV accuracy: " + str(scores5) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores5)))
scores5_p = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='precision')
print("Random Forest with visual features has 5 fold CV prec: " + str(scores5_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores5_p)))
scores5_r = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='recall')
print("Random Forest with visual features has 5 fold CV recall: " + str(scores5_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores5_r)))
scores5_f1 = (2 * scores5_p * scores5_r )/ (scores5_p + scores5_r)
print("Random Forest with visual features has 5 fold CV f1: " + str(scores5_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores5_f1)))
scores5 = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='roc_auc')
print("Random Forest with visual features has 5 fold CV AUC: " + str(scores5) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores5)))
#pdb.set_trace()

#Ensemble the individual predictions
out_ensemble = out3 + out4 + out5 #Acoustic + verbal + visual

out_ensemble[out_ensemble < 2] = 0 #Take best 2 out of 3 voting
out_ensemble[out_ensemble >= 2] = 1
print("Ensemble of individual predictions from each modality: ")
print(classification_report(Y_labels, out_ensemble, labels = np.array([0, 1])))



#Acoustic and verbal
clf9 = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight="balanced")
clf9.fit(X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels)
train_acc9 = clf9.score(X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels)
print("Random Forest with acoustic + verbal features has accuracy: "+ str(train_acc9))
out9 = clf9.predict(X_data[:, :audio_feat_len + verbal_feat_len ])
print(classification_report(Y_labels, out9))
scores9 = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5)
print("Random Forest with acoustic+verbal features has 5 fold CV accuracy: " + str(scores9) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores9)))
scores9_p = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='precision')
print("Random Forest with acoustic+verbal features has 5 fold CV prec: " + str(scores9_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores9_p)))
scores9_r = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='recall')
print("Random Forest with acoustic+verbal features has 5 fold CV recall: " + str(scores9_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores9_r)))
scores9_f1 = (2 * scores9_p * scores9_r )/ (scores9_p + scores9_r)
print("Random Forest with acoustic+verbal features has 5 fold CV f1: " + str(scores9_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores9_f1)))
scores9 = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("Random Forest with acoustic+verbal features has 5 fold CV AUC: " + str(scores9) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores9)))

#Visual and verbal
clf10 = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight="balanced")
clf10.fit(X_data[:, audio_feat_len:  ], Y_labels)
train_acc10 = clf10.score(X_data[:, audio_feat_len: ], Y_labels)
print("Random Forestwith visual + verbal features has accuracy: "+ str(train_acc10))
out10 = clf10.predict(X_data[:, audio_feat_len: ])
print(classification_report(Y_labels, out10))
scores10 = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5)
print("Random Forest with visual+verbal features has 5 fold CV accuracy: " + str(scores10) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores10)))
cores10_p = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='precision')
print("Random Forest with visual+verbal features has 5 fold CV prec: " + str(scores10_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores10_p)))
scores10_r = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='recall')
print("Random Forest with visual+verbal features has 5 fold CV recall: " + str(scores10_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores10_r)))
scores10_f1 = (2 * scores10_p * scores10_r )/ (scores10_p + scores10_r)
print("Random Forest with visual+verbal features has 5 fold CV f1: " + str(scores10_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores10_f1)))
scores10 = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='roc_auc')
print("Random Forest with visual+verbal features has 5 fold CV AUC: " + str(scores10) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores10)))

#Visual and Acoustic
X_joint_data = np.concatenate((X_data[:, :audio_feat_len ], X_data[:,  audio_feat_len + verbal_feat_len : ]), axis=1)
clf11 = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight="balanced")
clf11.fit(X_joint_data, Y_labels)
train_acc11 = clf11.score(X_joint_data, Y_labels)
print("Random Forest with visual+acoustic features has accuracy: "+ str(train_acc11))
out11 = clf11.predict(X_joint_data)
print(classification_report(Y_labels, out11))
scores11 = cross_val_score(clf11, X_joint_data, Y_labels, cv=5)
print("Random Forest with visual+acoustic features has 5 fold CV accuracy: " + str(scores11) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores11)))
cores11_p = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='precision')
print("Random Forest with visual+acoustic features has 5 fold CV prec: " + str(scores11_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores11_p)))
scores11_r = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='recall')
print("Random Forest with visual+acoustic features has 5 fold CV recall: " + str(scores11_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores11_r)))
scores11_f1 = (2 * scores11_p * scores11_r )/ (scores11_p + scores11_r)
print("Random Forest with visual+acoustic features has 5 fold CV f1: " + str(scores11_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores11_f1)))
scores11 = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='roc_auc')
print("Random Forest with visual+acoustic features has 5 fold CV AUC: " + str(scores11) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores11)))



#####################################################################################################
#XGBoost
from xgboost import XGBClassifier
#####################################################################################################

#1. All features concatenate 
clf1 =  XGBClassifier()
clf1.fit(X_data, Y_labels)
train_acc1 = clf1.score(X_data, Y_labels)
print("XGBoost with all features has accuracy: "+ str(train_acc1))
out1 = clf1.predict(X_data)
out1 = [round(value) for value in out1]
print(classification_report(Y_labels, out1))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5)
print("XGBoost with all features has 5 fold CV accuracy: " + str(scores1) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores1)))
scores1_p = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='precision')
print("XGBoost with all features has 5 fold CV prec: " + str(scores1_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores1_p)))
scores1_r = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='recall')
print("XGBoost with all features has 5 fold CV recall: " + str(scores1_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores1_r)))
scores1_f1 = (2 * scores1_p * scores1_r )/ (scores1_p + scores1_r)
print("XGBoost with all features has 5 fold CV f1: " + str(scores1_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores1_f1)))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='roc_auc')
print("XGBoost with all features has 5 fold CV AUC: " + str(scores1) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores1)))


#2. separate features (vocal, visual, verbal) and pairs of these with linear SVM
#Acoustic
clf3 = XGBClassifier()
clf3.fit(X_data[:, :audio_feat_len ], Y_labels)
train_acc3 = clf3.score(X_data[:, :audio_feat_len ], Y_labels)
print("XGBoost with acoustic features has accuracy: "+ str(train_acc3))
out3 = clf3.predict(X_data[:, :audio_feat_len ])
out3 = [round(value) for value in out3]
print(classification_report(Y_labels, out3))
scores3 = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5)
print("XGBoost with acoustic features has 5 fold CV accuracy: " + str(scores3) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores3)))
scores3_p = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='precision')
print("XGBoost with acoustic features has 5 fold CV prec: " + str(scores3_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores3_p)))
scores3_r = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='recall')
print("XGBoost with acoustic features has 5 fold CV recall: " + str(scores3_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores3_r)))
scores3_f1 = (2 * scores3_p * scores3_r )/ (scores3_p + scores3_r)
print("XGBoost with acoustic features has 5 fold CV f1: " + str(scores3_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores3_f1)))
scores3 = cross_val_score(clf3, X_data[:, :audio_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("XGBoost with acoustic features has 5 fold CV AUC: " + str(scores3) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores3)))

#Verbal
clf4 = XGBClassifier()
clf4.fit(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels)
train_acc4 = clf4.score(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels)
print("XGBoost with verbal features has accuracy: "+ str(train_acc4))
out4 = clf4.predict(X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ])
out4 = [round(value) for value in out4]
print(classification_report(Y_labels, out4))
scores4 = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5)
print("XGBoost with verbal features has 5 fold CV accuracy: " + str(scores4) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores4)))
scores4_p = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='precision')
print("XGBoost with verbal features has 5 fold CV prec: " + str(scores4_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores4_p)))
scores4_r = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='recall')
print("XGBoost with verbal features has 5 fold CV recall: " + str(scores4_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores4_r)))
scores4_f1 = (2 * scores4_p * scores4_r )/ (scores4_p + scores4_r)
print("XGBoost with verbal features has 5 fold CV f1: " + str(scores4_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores4_f1)))
scores4 = cross_val_score(clf4, X_data[:, audio_feat_len : audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("XGBoost with verbal features has 5 fold CV AUC: " + str(scores4) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores4)))

#Visual
clf5 = XGBClassifier()
clf5.fit(X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels)
train_acc5 = clf5.score(X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels)
print("XGBoost with visual features has accuracy: "+ str(train_acc5))
out5 = clf5.predict(X_data[:,  audio_feat_len + verbal_feat_len : ])
out5 = [round(value) for value in out5]
print(classification_report(Y_labels, out5))
scores5 = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5)
print("Random Forestwith visual features has 5 fold CV accuracy: " + str(scores5) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores5)))
scores5_p = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='precision')
print("XGBoost with visual features has 5 fold CV prec: " + str(scores5_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores5_p)))
scores5_r = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='recall')
print("XGBoost with visual features has 5 fold CV recall: " + str(scores5_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores5_r)))
scores5_f1 = (2 * scores5_p * scores5_r )/ (scores5_p + scores5_r)
print("XGBoost with visual features has 5 fold CV f1: " + str(scores5_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores5_f1)))
scores5 = cross_val_score(clf5, X_data[:,  audio_feat_len + verbal_feat_len : ], Y_labels, cv=5, scoring='roc_auc')
print("XGBoost with visual features has 5 fold CV AUC: " + str(scores5) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores5)))
#pdb.set_trace()



#Acoustic and verbal
clf9 = XGBClassifier()
clf9.fit(X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels)
train_acc9 = clf9.score(X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels)
print("XGBoost with acoustic + verbal features has accuracy: "+ str(train_acc9))
out9 = clf9.predict(X_data[:, :audio_feat_len + verbal_feat_len ])
out9 = [round(value) for value in out9]
print(classification_report(Y_labels, out9))
scores9 = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5)
print("XGBoost with acoustic+verbal features has 5 fold CV accuracy: " + str(scores9) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores9)))
scores9_p = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='precision')
print("XGBoost with acoustic+verbal features has 5 fold CV prec: " + str(scores9_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores9_p)))
scores9_r = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='recall')
print("XGBoost with acoustic+verbal features has 5 fold CV recall: " + str(scores9_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores9_r)))
scores9_f1 = (2 * scores9_p * scores9_r )/ (scores9_p + scores9_r)
print("XGBoost with acoustic+verbal features has 5 fold CV f1: " + str(scores9_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores9_f1)))
scores9 = cross_val_score(clf9, X_data[:, :audio_feat_len + verbal_feat_len ], Y_labels, cv=5, scoring='roc_auc')
print("XGBoost with acoustic+verbal features has 5 fold CV AUC: " + str(scores9) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores9)))

#Visual and verbal
clf10 = XGBClassifier()
clf10.fit(X_data[:, audio_feat_len:  ], Y_labels)
train_acc10 = clf10.score(X_data[:, audio_feat_len: ], Y_labels)
print("XGBoost with visual + verbal features has accuracy: "+ str(train_acc10))
out10 = clf10.predict(X_data[:, audio_feat_len: ])
out10 = [round(value) for value in out10]
print(classification_report(Y_labels, out10))
scores10 = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5)
print("XGBoost with visual+verbal features has 5 fold CV accuracy: " + str(scores10) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores10)))
scores10_p = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='precision')
print("XGBoost with visual+verbal features has 5 fold CV prec: " + str(scores10_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores10_p)))
scores10_r = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='recall')
print("XGBoost with visual+verbal features has 5 fold CV recall: " + str(scores10_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores10_r)))
scores10_f1 = (2 * scores10_p * scores10_r )/ (scores10_p + scores10_r)
print("XGBoost with visual+verbal features has 5 fold CV f1: " + str(scores10_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores10_f1)))
scores10 = cross_val_score(clf10, X_data[:, audio_feat_len: ], Y_labels, cv=5, scoring='roc_auc')
print("XGBoost with visual+verbal features has 5 fold CV AUC: " + str(scores10) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores10)))

#Visual and Acoustic
X_joint_data = np.concatenate((X_data[:, :audio_feat_len ], X_data[:,  audio_feat_len + verbal_feat_len : ]), axis=1)
clf11 = XGBClassifier()
clf11.fit(X_joint_data, Y_labels)
train_acc11 = clf11.score(X_joint_data, Y_labels)
print("XGBoost with visual+acoustic features has accuracy: "+ str(train_acc11))
out11 = clf11.predict(X_joint_data)
out11 = [round(value) for value in out11]
print(classification_report(Y_labels, out11))
scores11 = cross_val_score(clf11, X_joint_data, Y_labels, cv=5)
print("XGBoost with visual+acoustic features has 5 fold CV accuracy: " + str(scores11) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores11)))
scores11_p = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='precision')
print("XGBoost with visual+acoustic features has 5 fold CV prec: " + str(scores11_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores11_p)))
scores11_r = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='recall')
print("XGBoost with visual+acoustic features has 5 fold CV recall: " + str(scores11_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores11_r)))
scores11_f1 = (2 * scores11_p * scores11_r )/ (scores11_p + scores11_r)
print("XGBoost with visual+acoustic features has 5 fold CV f1: " + str(scores11_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores11_f1)))
scores11 = cross_val_score(clf11, X_joint_data, Y_labels, cv=5, scoring='roc_auc')
print("XGBoost with visual+acoustic features has 5 fold CV AUC: " + str(scores11) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores11)))


'''
############################################################################################
#Classifiers for Manusal annotated data
############################################################################################
X_data = X_manual_data
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


from sklearn.svm import SVC


#Uncomment to run SVm fits

#1. All features concatenate and use linear SVM
clf1 = SVC(kernel='linear')
clf1.fit(X_data, Y_labels)
train_acc1 = clf1.score(X_data, Y_labels)
print("Linear SVM with all manual features has accuracy: "+ str(train_acc1))
out1 = clf1.predict(X_data)
print(classification_report(Y_labels, out1))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5)
print("Linear SVM with all manual features has 5 fold CV accuracy: " + str(scores1) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores1)))
scores1_p = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='precision')
print("Linear SVM with all manual features has 5 fold CV prec: " + str(scores1_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores1_p)))
scores1_r = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='recall')
print("Linear SVM with all manual features has 5 fold CV recall: " + str(scores1_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores1_r)))
scores1_f1 = (2 * scores1_p * scores1_r )/ (scores1_p + scores1_r)
print("Linear SVM with all manual features has 5 fold CV f1: " + str(scores1_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores1_f1)))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='roc_auc')
print("Linear SVM with all manual features has 5 fold CV AUC: " + str(scores1) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores1)))


from sklearn.ensemble import RandomForestClassifier
#1. All features concatenate and use linear SVM
clf1 = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight="balanced")
clf1.fit(X_data, Y_labels)
train_acc1 = clf1.score(X_data, Y_labels)
print("Random Forest with all manual features has accuracy: "+ str(train_acc1))
out1 = clf1.predict(X_data)
print(classification_report(Y_labels, out1))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5)
print("Random Forest with all manual features has 5 fold CV accuracy: " + str(scores1) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores1)))
scores1_p = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='precision')
print("Random Forest with all manual features has 5 fold CV prec: " + str(scores1_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores1_p)))
scores1_r = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='recall')
print("Random Forest with all manual features has 5 fold CV recall: " + str(scores1_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores1_r)))
scores1_f1 = (2 * scores1_p * scores1_r )/ (scores1_p + scores1_r)
print("Random Forest with all manual features has 5 fold CV f1: " + str(scores1_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores1_f1)))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='roc_auc')
print("Random Forest with all manual features has 5 fold CV AUC: " + str(scores1) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores1)))


from xgboost import XGBClassifier
#1. All features concatenate 
clf1 =  XGBClassifier()
clf1.fit(X_data, Y_labels)
train_acc1 = clf1.score(X_data, Y_labels)
print("XGBoost with all manual features has accuracy: "+ str(train_acc1))
out1 = clf1.predict(X_data)
out1 = [round(value) for value in out1]
print(classification_report(Y_labels, out1))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5)
print("XGBoost with all manual features has 5 fold CV accuracy: " + str(scores1) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores1)))
scores1_p = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='precision')
print("XGBoost with all manual features has 5 fold CV prec: " + str(scores1_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores1_p)))
scores1_r = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='recall')
print("XGBoost with all manual features has 5 fold CV recall: " + str(scores1_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores1_r)))
scores1_f1 = (2 * scores1_p * scores1_r )/ (scores1_p + scores1_r)
print("XGBoost with all features has 5 fold CV f1: " + str(scores1_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores1_f1)))
scores1 = cross_val_score(clf1, X_data, Y_labels, cv=5, scoring='roc_auc')
print("XGBoost with all manual features has 5 fold CV AUC: " + str(scores1) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores1)))


