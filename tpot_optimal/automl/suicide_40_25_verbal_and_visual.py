import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

X_data=np.load('X_data.npy')
Y_labels=np.load('Y_labels.npy')
audio_feat_len=14
verbal_feat_len=784
verbal_and_visual = X_data[:, audio_feat_len: ]
training_features,testing_features,training_target,testing_target=train_test_split(verbal_and_visual,Y_labels,train_size=0.75,test_size=0.25)
# NOTE: Make sure that the class is labeled 'target' in the data file

# Average CV score on the training set was:0.831868131868
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.005),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.5, min_samples_leaf=9, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
###Best pipeline: RandomForestClassifier(VarianceThreshold(input_matrix, threshold=0.005), bootstrap=False, criterion=entropy, max_features=0.5, min_samples_leaf=9, min_samples_split=12, n_estimators=100)
#0.4782608695652174
print(classification_report(testing_target, results))

scores1 = cross_val_score(exported_pipeline, X_data, Y_labels, cv=5)
print("all features has 5 fold CV accuracy: " + str(scores1) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores1)))
scores1_p = cross_val_score(exported_pipeline, X_data, Y_labels, cv=5, scoring='precision')
print("all features has 5 fold CV prec: " + str(scores1_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores1_p)))
scores1_r = cross_val_score(exported_pipeline, X_data, Y_labels, cv=5, scoring='recall')
print("all features has 5 fold CV recall: " + str(scores1_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores1_r)))
scores1_f1 = (2 * scores1_p * scores1_r )/ (scores1_p + scores1_r)
print("all features has 5 fold CV f1: " + str(scores1_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores1_f1)))
scores1 = cross_val_score(exported_pipeline, X_data, Y_labels, cv=5, scoring='roc_auc')
print("all features has 5 fold CV AUC: " + str(scores1) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores1)))
